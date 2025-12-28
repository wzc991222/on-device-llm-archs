import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
from einops import rearrange, repeat
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts as checkpoint_contexts,
)
import functools
import os
import torch.distributed as dist

flex_attention = torch.compile(flex_attention, dynamic=False)
from utils import GroupedGEMM, grouped_gemm_func_original
from config import Config

grouped_gemm = GroupedGEMM.apply


def grouped_gemm_func(
    x: torch.Tensor,
    block: torch.Tensor,
    index: torch.Tensor,
    score: torch.Tensor,
    config: Config,
):
    inputs = x, block, index, score, config
    if config.grouped_gemm_func:
        if config.grouped_gemm_checkpoint:
            ops_to_save = []
            context_fn = functools.partial(checkpoint_contexts, ops_to_save)
            return checkpoint(
                grouped_gemm_func_original,
                *inputs,
                use_reentrant=False,
                context_fn=context_fn,
            )
        else:
            return grouped_gemm_func_original(*inputs)
    else:
        return grouped_gemm(*inputs)


class RotaryEmb(nn.Module):
    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.register_buffer("inv_freq", (1 / theta) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.freqs_cos = None
        self.freqs_sin = None

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[2]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)[None, None, :, :]
            self.seq_len_cached = seq_len
            self.freqs_cos = freqs.cos()
            self.freqs_sin = freqs.sin()
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * self.freqs_cos + x2 * self.freqs_sin
        y2 = -x1 * self.freqs_sin + x2 * self.freqs_cos
        x_output = torch.cat([y1, y2], -1).type_as(x)
        return x_output


class Attention(nn.Module):
    def __init__(self, rotary_emb: RotaryEmb, attn_num: int):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.attn_num = attn_num

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_mask: BlockMask,
        config: Config,
    ):
        xq = rearrange(xq, "b s (h a) -> b h s a", h=self.attn_num)
        xk = rearrange(xk, "b s (h a) -> b h s a", h=self.attn_num)
        xv = rearrange(xv, "b s (h a) -> b h s a", h=self.attn_num)
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)
        kernel_options = config.kernel_options if config.flex_attn_spec_kernel else None
        x_attn = flex_attention(
            xq,
            xk,
            xv,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )
        x_attn = rearrange(x_attn, "b h s a -> b s (h a)")
        return x_attn


class Layer(nn.Module):
    def __init__(self, config: Config, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn_h_dim = int(dim * config.ffn_factor)

        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.ffn_up = nn.Linear(dim, self.ffn_h_dim * 2, bias=False)
        self.ffn_down = nn.Linear(self.ffn_h_dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

    def forward(
        self,
        x_input: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), self.dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config)
        x_attn_o = self.attn_o(x_attn)

        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)
        x1, x2 = torch.split(self.ffn_up(x_ffn), self.ffn_h_dim, -1)
        x3 = F.silu(x1) * x2
        x_ffn_out = self.ffn_down(x3)
        x_output = x_ffn_out + x_ffn_input
        return x_output


class AuxRouter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.aux_dim
        attn_dim = config.aux_attn_dim
        an = config.aux_attn_num
        rs = config.router_size
        aux_ln = config.aux_layer_num

        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(aux_ln)])
        self.router_token = nn.Parameter(torch.randn(1, 1, rs, dim))
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        mask_mod_type: int,
    ):
        n = config.block_size
        rs = config.router_size
        m = n + rs

        x = rearrange(x, "b (l n) d -> b l n d", n=n)
        b, l = x.shape[:2]
        router_token = self.router_token.expand(b, l, -1, -1)
        x = torch.cat([x, router_token], 2).flatten(1, 2)

        def mask_mod_1(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // m
            kv_b_idx = kv_idx // m
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return causal_mask & doc_mask
        
        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // m
            kv_b_idx = kv_idx // m
            block_mask = q_b_idx == kv_b_idx
            return block_mask

        def mask_mod_3(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // m
            kv_b_idx = kv_idx // m
            kv_token = (kv_idx % m) < n
            causal_mask = q_b_idx > kv_b_idx
            block_mask = q_b_idx == kv_b_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return (causal_mask & kv_token | block_mask) & doc_mask

        b, s = x.shape[:2]
        if mask_mod_type == 1:
            mask_mod = mask_mod_1
        if mask_mod_type == 2:
            mask_mod = mask_mod_2
        if mask_mod_type == 3:
            mask_mod = mask_mod_3
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)

        x = rearrange(x, "b (l m) d -> b l m d", m=m)
        x = x[:, :, n:].flatten(0, 1)
        return x
    

class AuxRouterVar(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.aux_dim
        attn_dim = config.aux_attn_dim
        an = config.aux_attn_num
        rs = config.router_size
        aux_t_ln = config.aux_token_layer_num
        aux_b_ln = config.aux_block_layer_num

        self.token_layers = nn.ModuleList([Layer(config, dim) for _ in range(aux_t_ln)])
        self.block_layers = nn.ModuleList([Layer(config, dim) for _ in range(aux_b_ln)])
        self.router_token = nn.Parameter(torch.randn(1, rs, dim))

        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)
        self.attention_s = Attention(RotaryEmb(attn_dim, config.theta_s), an)

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        mask_mod_type: int,
    ):
        n = config.block_size

        def mask_mod_1(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // n
            kv_b_idx = kv_idx // n
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return causal_mask & doc_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // n
            kv_b_idx = kv_idx // n
            block_mask = q_b_idx == kv_b_idx
            return block_mask

        b, s = x.shape[:2]
        if mask_mod_type == 1:
            token_mask_mod = mask_mod_1
        if mask_mod_type == 2:
            token_mask_mod = mask_mod_2
        block_mask = create_block_mask(token_mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.token_layers):
            x = layer(x, block_mask, self.attention, config)

        x = rearrange(x, "b (l n) d -> (b l) n d", n=n)
        router_token = self.router_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, router_token], 1)

        def block_mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= 0

        b, s = x.shape[:2]
        block_mask = create_block_mask(block_mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.block_layers):
            x = layer(x, block_mask, self.attention_s, config)
        x = x[:, n:]
        return x


class AuxModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        rn = config.router_num
        dim = config.main_dim
        dim_a = config.aux_dim
        en = config.routed_expert_num
        ge = config.group_expert_num
        gn = config.router_group_num

        self.main_to_aux = nn.Linear(dim, dim_a, bias=False)
        if config.aux_router_variant:
            self.aux_router = AuxRouterVar(config)
        else:
            self.aux_router = AuxRouter(config)
        self.keys = nn.Parameter(torch.randn(rn * gn, dim_a, ge))
        self.query_norm = nn.RMSNorm(dim_a, eps=config.norm_eps)
        self.register_buffer("stat", torch.zeros(rn * en))
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("logits_max", torch.zeros(1))
        self.register_buffer("logits_mean", torch.zeros(1))

    def reset(self):
        self.stat -= self.stat
        self.count -= self.count
        self.logits_max -= self.logits_max
        self.logits_mean -= self.logits_mean

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        aux_mode: bool,
        balance_loss_mode: bool,
    ):
        rn = config.router_num
        en = config.routed_expert_num
        bf = config.balance_loss_factor
        bc = config.main_balance_loss_coeff
        gt = config.group_expert_num_per_token
        gn = config.router_group_num
        ge = config.group_expert_num
        device = x.device

        x = self.main_to_aux(x)
        x = self.aux_router(x, doc, config, aux_mode)
        x = self.query_norm(x)

        logits = torch.bmm(x, self.keys)
        logits_m = logits
        if config.balance_loss_free:
            logits_m = logits_m + self.bias.reshape(-1, ge)
        indices_o = logits_m.topk(gt, -1, sorted=False)[1]
        indices = rearrange(indices_o, "(rn gn) bl gt -> rn bl gn gt", rn=rn)
        offset = torch.arange(gn, device=device)[None, None, :, None] * ge
        indices = indices + offset
        indices = indices.flatten(1)
        indices_bin_o = F.one_hot(indices, en).type_as(x).mean(1) * en

        values_f = logits.softmax(-1)
        values = torch.gather(values_f, -1, indices_o)
        values = rearrange(values, "(rn gn) bl gt -> rn (bl gn gt)", rn=rn)
        if config.value_coeff:
            bin_coeff = (1 / (indices_bin_o + 1e-3)).clamp_max(config.value_max_coeff)
            vc = torch.gather(bin_coeff, -1, indices)
            values = values * vc + values.detach() * (1 - vc)

        balance_loss = torch.tensor(0, device=device)
        if balance_loss_mode:
            self.stat += indices_bin_o.flatten()
            self.count += 1
            ddp = int(os.environ.get("RANK", -1)) != -1
            if ddp:
                dist.all_reduce(self.stat, op=dist.ReduceOp.AVG)
            if not config.balance_loss_free:
                indices_bin = self.stat / self.count
                values_aux = rearrange(logits, "(rn gn) bl ge -> rn bl (gn ge)", rn=rn)
                values_bin = values_aux.softmax(-1).mean(1).flatten()
                bc = 1 if aux_mode else bc
                balance_loss = (indices_bin * values_bin).sum() * bf * bc
                self.logits_max += values_aux.amax(-1).mean().detach()
                self.logits_mean += values_aux.mean().detach()

        return indices, values, balance_loss

class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        dim_e = config.expert_dim
        dim_s = config.shared_expert_dim
        le = config.layer_expert_num
        self.dim = dim
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_experts = nn.Parameter(torch.randn(3, le, dim, dim_e))
        if config.token_keys:
            self.token_keys = nn.Parameter(torch.randn(dim, le))
            self.main_bias = nn.Parameter(torch.zeros(le))
        if config.shared_expert:
            self.ffn_up = nn.Linear(dim, dim_s * 2, bias=False)
            self.ffn_down = nn.Linear(dim_s, dim, bias=False)

    def forward(
        self,
        x_input: torch.Tensor,
        indices: torch.LongTensor,
        values: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), self.dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config)
        x_attn_o = self.attn_o(x_attn)
        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)

        scores = values
        if config.token_keys:
            token_values = (x_ffn @ self.token_keys).flatten(0, 1)
            token_values = torch.gather(token_values, -1, indices)
            token_values = (token_values + self.main_bias[indices]).sigmoid()
            scores = scores + token_values

        scores = scores / scores.sum(-1, keepdim=True)
        indices = indices.flatten()
        scores = scores.flatten().unsqueeze(-1) * config.routed_scaling_factor
        y = grouped_gemm_func(x_ffn, self.ffn_experts, indices, scores, config)

        if config.shared_expert:
            x1, x2 = torch.split(self.ffn_up(x_ffn), config.shared_expert_dim, -1)
            x3 = F.silu(x1) * x2
            y_shared = self.ffn_down(x3)
            y = y + y_shared

        y = y + x_ffn_input
        return y


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.main_dim
        attn_dim = config.main_attn_dim
        ln = config.moe_layer_num
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense
        an = config.main_attn_num

        if first_k > 0:
            self.first_dense_layers = nn.ModuleList(
                [Layer(config, dim) for _ in range(first_k)]
            )
        self.moe_layers = nn.ModuleList([MoELayer(config) for _ in range(ln)])
        if last_k > 0:
            self.last_dense_layers = nn.ModuleList(
                [Layer(config, dim) for _ in range(last_k)]
            )
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor,
        indices: torch.LongTensor,
        values: torch.Tensor,
        config: Config,
    ):
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return causal_mask & doc_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        if first_k > 0:
            for layer in self.first_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        for i, layer in enumerate(self.moe_layers):
            j = i if config.layer_wise_router else 0
            x = layer(x, indices[j], values[j], block_mask, self.attention, config)

        if last_k > 0:
            for layer in self.last_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        return x


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        rn = config.router_num
        self.vocab_emb = nn.Embedding(config.vocab_size, dim)
        self.aux_model = AuxModel(config)
        self.main_model = MainModel(config)
        self.final_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        self.default_bias = nn.Parameter(torch.zeros(rn, 1))
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        input: torch.LongTensor,
        target: torch.LongTensor | None,
        config: Config,
        aux_mode: bool = False,
        bl_mode: bool = False,
    ):
        b = input.shape[0]
        n = config.block_size
        l = config.seq_block_num
        rn = config.router_num
        rs = config.router_size
        en = config.routed_expert_num
        le = config.layer_expert_num
        et = config.expert_num_per_token

        doc_aux = None
        if not aux_mode or config.aux_block_attn:
            eot = input == config.eot_idx
            eot_aux = rearrange(eot, "b (l n) -> b l n", n=n)
            router_pad = torch.zeros(1, 1, 1, device=eot_aux.device).expand(b, l, rs)
            eot_aux = torch.cat([eot_aux, router_pad], -1).flatten(1)
            doc_aux = torch.zeros_like(eot_aux)
            doc_aux[:, 1:] = eot_aux.cumsum(-1)[:, :-1]

        x = self.vocab_emb(input)
        indices, values, balance_loss = self.aux_model(
            x, doc_aux, config, aux_mode, bl_mode
        )
        if aux_mode:
            return balance_loss

        eot = input == config.eot_idx
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        b_eot = rearrange(eot, "b (l n) -> b l n", n=n).any(-1)
        mask = b_eot | b_eot.roll(1, 1)
        mask[:, 0] = True
        mask = repeat(mask, "b l -> rn (b l et)", rn=rn, et=et)
        default_idx = torch.arange(en, le, device=indices.device)
        default_idx = repeat(default_idx, "et -> rn (bl et)", rn=rn, bl=b * l)
        indices = torch.where(mask, default_idx, indices)
        values = torch.where(mask, self.default_bias, values)
        indices = repeat(indices, "rn (bl et) -> rn (bl n) et", n=n, et=et)
        values = repeat(values, "rn (bl et) -> rn (bl n) et", n=n, et=et)

        y = self.main_model(x, doc, indices, values, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(logits.flatten(0, -2), target.flatten())
        loss = main_loss + balance_loss
        return loss, main_loss.detach(), balance_loss.detach()

    def init_weights(self, config: Config):
        std = config.init_std
        g = torch.Generator()
        g.manual_seed(42)

        for name, param in self.named_parameters():
            if "norm" in name or "bias" in name:
                continue
            else:
                nn.init.trunc_normal_(
                    param, mean=0, std=std, a=-3 * std, b=3 * std, generator=g
                )

    def params_count(self, config: Config):
        aux_model_params = sum(x.numel() for x in self.aux_model.parameters())
        main_model_params = sum(x.numel() for x in self.main_model.parameters())
        total_params = sum(x.numel() for x in self.parameters())
        vocab_emb_params = total_params - aux_model_params - main_model_params
        params_result = {
            "aux_model_params": aux_model_params,
            "main_model_params": main_model_params,
            "vocab_emb_params": vocab_emb_params,
            "total_params": total_params,
        }

        aux_router_params = sum(
            x.numel() for x in self.aux_model.aux_router.parameters()
        )
        aux_model_a_params = aux_model_params + int(
            aux_router_params * config.router_size / config.block_size
        )
        ffn_experts_not_a_params = (
            3
            * config.main_dim
            * config.expert_dim
            * config.routed_expert_num
            * config.moe_layer_num
        )
        main_model_a_params = main_model_params - ffn_experts_not_a_params
        vocab_emb_a_params = config.main_dim * config.vocab_size
        total_a_params = aux_model_a_params + main_model_a_params + vocab_emb_a_params
        activated_params_result = {
            "aux_model_a_params": aux_model_a_params,
            "main_model_a_params": main_model_a_params,
            "vocab_emb_a_params": vocab_emb_a_params,
            "total_a_params": total_a_params,
        }

        return params_result, activated_params_result

    def get_optimizer(self, config: Config):
        main_lr = config.max_lr
        aux_lr = config.aux_max_lr
        wd = config.weight_decay
        betas = config.adam_betas
        main_params = [
            param for name, param in self.named_parameters() if "aux_model" not in name
        ]
        aux_params = [
            param for name, param in self.named_parameters() if "aux_model" in name
        ]
        param_groups = [
            {"params": main_params, "lr": main_lr, "weight_decay": wd},
            {"params": aux_params, "lr": aux_lr, "weight_decay": wd},
        ]
        if config.zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                param_groups,
                optimizer_class=torch.optim.AdamW,
                betas=betas,
                foreach=True,
            )
        else:
            optimizer = torch.optim.AdamW(param_groups, betas=betas, foreach=True)
        return optimizer
