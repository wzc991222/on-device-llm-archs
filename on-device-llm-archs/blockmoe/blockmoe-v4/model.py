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


class AuxEncoder(nn.Module):
    def __init__(self, config: Config, layer_num: int):
        super().__init__()

        dim = config.aux_dim
        attn_dim = config.aux_attn_dim
        an = config.aux_attn_num
        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(layer_num)])
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        global_attn: bool,
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
        mask_mod = mask_mod_1 if global_attn else mask_mod_2
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        return x


class AuxRouter(nn.Module):
    def __init__(self, config: Config, layer_num: int):
        super().__init__()

        dim = config.aux_dim
        attn_dim = config.aux_attn_dim
        an = config.aux_attn_num
        ln = config.moe_layer_num
        self.router_token = nn.Parameter(torch.randn(1, ln, dim))
        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(layer_num)])
        self.ouput = nn.Linear(dim, config.aux_output_dim, bias=False)
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        config: Config,
        aux_mode: bool,
    ):
        n = config.block_size
        gn = config.router_group_num
        dr = config.aux_router_dim
        dg = config.aux_gate_dim
        l = config.batch_block_num

        x = rearrange(x, "b (l n) d -> (b l) n d", n=n)
        router_token = self.router_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, router_token], 1)

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= 0

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        x = self.ouput(x[:, n:])

        xr, xg = x.split([dr, dg], -1)
        xr = repeat(xr, "(b l) ln dr -> (ln gn) b l dr", l=l, gn=gn)
        xg = repeat(xg, "(b l) ln dg -> (ln gn) b l dg", l=l, gn=gn)
        if not aux_mode:
            xr = xr.roll(1, 2)
            xg = xg.roll(1, 2)
        xr = xr.flatten(1, 2)
        xg = xg.flatten(1, 2)

        return xr, xg


class AuxModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        encoder_ln = config.aux_encoder_layer_num
        router_ln = config.aux_router_layer_num
        ln = config.moe_layer_num
        en = config.routed_expert_num
        ge = config.group_expert_num
        gn = config.router_group_num
        dr = config.aux_router_dim
        dg = config.aux_gate_dim

        self.aux_encoder = AuxEncoder(config, encoder_ln)
        self.aux_router = AuxRouter(config, router_ln)
        self.keys_router = nn.Parameter(torch.randn(ln * gn, dr, ge))
        self.keys_gate = nn.Parameter(torch.randn(ln * gn, dg, ge))
        self.router_norm = nn.RMSNorm(dr, eps=config.norm_eps)
        self.gate_norm = nn.RMSNorm(dg, eps=config.norm_eps)
        self.register_buffer("stat", torch.zeros(ln * en))
        self.register_buffer("count", torch.zeros(1))

    def reset(self):
        self.stat -= self.stat
        self.count -= self.count

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        aux_mode: bool,
        print_info: bool,
    ):
        ln = config.moe_layer_num
        en = config.routed_expert_num
        gt = config.group_expert_num_per_token
        gn = config.router_group_num
        ge = config.group_expert_num
        device = x.device
        global_attn = True if config.aux_encoder_global_attn and not aux_mode else False

        x = self.aux_encoder(x, doc, config, global_attn)
        x_router, x_gate = self.aux_router(x, config, aux_mode)
        x_router = self.router_norm(x_router)
        x_gate = self.gate_norm(x_gate)
        router_logits = torch.bmm(x_router, self.keys_router)
        router_values, indices_o = router_logits.topk(gt, -1, sorted=False)
        indices = rearrange(indices_o, "(ln gn) bl gt -> ln bl gn gt", ln=ln)
        offset = torch.arange(gn, device=device)[None, None, :, None] * ge
        indices = indices + offset
        indices = indices.flatten(1)

        gate_logits = torch.bmm(x_gate, self.keys_gate)
        gate_values = torch.gather(gate_logits, -1, indices_o)
        values = router_values + gate_values
        values = rearrange(values, "(ln gn) bl gt -> ln (bl gn gt)", ln=ln)

        balance_loss = torch.tensor(0, device=device)
        if aux_mode or config.main_router_active:
            indices_bin_o = (
                F.one_hot(indices, en).type_as(x).mean(1).flatten()
            )
            self.stat += indices_bin_o
            self.count += 1
            ddp = int(os.environ.get("RANK", -1)) != -1
            if ddp:
                dist.all_reduce(self.stat, op=dist.ReduceOp.AVG)

            indices_bin = self.stat / self.count
            values_aux = rearrange(
                router_logits, "(ln gn) bl ge -> ln bl (gn ge)", ln=ln
            )
            values_bin = values_aux.softmax(-1).mean(1).flatten()
            balance_loss = (
                (indices_bin * values_bin).sum() * en * config.balance_loss_factor
            )

            if aux_mode and print_info:
                print(f"aux mode:")
                print(f"logits max: {values_aux.amax(-1).mean()}")
                print(f"logits mean: {values_aux.mean()}")
                indices_bin_m = rearrange(indices_bin, "(ln en) -> ln en", ln=ln)
                print(f"indices layer 0: {indices_bin_m[0]}")
                print(f"indices max: {indices_bin_m.amax(1).mean()}")

        return indices, values, balance_loss


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        dim_s = config.shared_expert_dim
        te = config.total_layer_expert_num
        self.dim = dim
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_experts = nn.Parameter(torch.randn(3, te, dim, config.expert_dim))
        self.main_keys = nn.Parameter(torch.randn(dim, te))
        self.main_bias = nn.Parameter(torch.zeros(te))
        if config.expert_post_norm:
            self.output_coeff = nn.Parameter(torch.ones(1, 1, dim))
        if config.shared_expert:
            self.ffn_up = nn.Linear(dim, dim_s * 2, bias=False)
            self.ffn_down = nn.Linear(dim_s, dim, bias=False)
            if config.expert_post_norm:
                self.shared_expert_norm = nn.RMSNorm(dim, eps=config.norm_eps)

    def forward(
        self,
        x_input: torch.Tensor,
        indices: torch.LongTensor,
        values: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
        aux_mode: bool,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), self.dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config)
        x_attn_o = self.attn_o(x_attn)
        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)

        token_values = (x_ffn @ self.main_keys).flatten(0, 1)
        token_values = torch.gather(token_values, -1, indices)
        values = values + token_values + self.main_bias[indices]
        if not aux_mode and not config.main_router_active:
            values = values.detach()

        if config.gate_func == "softmax":
            scores = values.softmax(-1)
        if config.gate_func == "sigmoid":
            scores = values.sigmoid()
        if config.gate_func == "sigmoid_norm":
            scores = values.sigmoid()
            scores = scores / scores.sum(-1, keepdim=True)
        indices = indices.flatten()
        scores = scores.flatten().unsqueeze(-1) * config.routed_scaling_factor
        y = grouped_gemm_func(x_ffn, self.ffn_experts, indices, scores, config)
        if config.expert_post_norm:
            y = y * self.output_coeff

        if config.shared_expert:
            x1, x2 = torch.split(self.ffn_up(x_ffn), config.shared_expert_dim, -1)
            x3 = F.silu(x1) * x2
            y_shared = self.ffn_down(x3)
            if config.expert_post_norm:
                y_shared = self.shared_expert_norm(y_shared)
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
        aux_mode: bool,
    ):
        n = config.block_size
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense

        def mask_mod_1(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return causal_mask & doc_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // n
            kv_b_idx = kv_idx // n
            causal_mask = q_idx >= kv_idx
            block_mask = q_b_idx == kv_b_idx
            return causal_mask & block_mask

        b, s = x.shape[:2]
        mask_mod = mask_mod_2 if aux_mode else mask_mod_1
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        if first_k > 0:
            for layer in self.first_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        for i, layer in enumerate(self.moe_layers):
            x = layer(
                x, indices[i], values[i], block_mask, self.attention, config, aux_mode
            )

        if last_k > 0:
            for layer in self.last_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        return x


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        self.vocab_emb = nn.Embedding(config.vocab_size, dim)
        self.main_to_aux = nn.Linear(dim, config.aux_dim, bias=False)
        self.aux_model = AuxModel(config)
        self.main_model = MainModel(config)
        self.final_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        self.default_bias = nn.Parameter(torch.zeros(config.moe_layer_num, 1))
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        aux_input: torch.LongTensor | None,
        input: torch.LongTensor,
        target: torch.LongTensor,
        config: Config,
        aux_mode: bool,
        print_info: bool = False
    ):
        b = input.shape[0]
        n = config.block_size
        ln = config.moe_layer_num
        en = config.routed_expert_num
        te = config.total_layer_expert_num
        et = config.expert_num_per_token
        l = config.batch_block_num

        if not aux_mode:
            eot = input == config.eot_idx
            doc = torch.zeros_like(eot)
            doc[:, 1:] = eot.cumsum(-1)[:, :-1]
            eot = rearrange(eot, "b (l n) -> b l n", n=n).any(-1)
            mask = eot | eot.roll(1, 1)
            mask[:, 0] = True
            x = self.vocab_emb(input)
            x_aux = self.main_to_aux(x)

        if aux_mode:
            doc = None
            aux_eot = aux_input == config.eot_idx
            input_eot = input == config.eot_idx
            aux_eot = rearrange(aux_eot, "b (l n) -> b l n", n=n).any(-1)
            input_eot = rearrange(input_eot, "b (l n) -> b l n", n=n).any(-1)
            mask = aux_eot | input_eot
            x = self.vocab_emb(input)
            x_aux = self.main_to_aux(self.vocab_emb(aux_input))

        indices, values, balance_loss = self.aux_model(x_aux, doc, config, aux_mode, print_info)
        mask = repeat(mask, "b l -> ln (b l et)", ln=ln, et=et)
        default_idx = torch.arange(en, te, device=indices.device)
        default_idx = repeat(default_idx, "et -> ln (bl et)", ln=ln, bl=b * l)
        indices = torch.where(mask, default_idx, indices)
        values = torch.where(mask, self.default_bias, values)
        indices = repeat(indices, "ln (bl et) -> ln (bl n) et", n=n, et=et)
        values = repeat(values, "ln (bl et) -> ln (bl n) et", n=n, et=et)

        y = self.main_model(x, doc, indices, values, config, aux_mode)
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
            if "norm" in name or "coeff" in name or "bias" in name:
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
            aux_router_params * config.moe_layer_num / config.block_size
        )
        ffn_experts_not_a_params = (
            3
            * config.main_dim
            * config.expert_dim
            * config.routed_expert_num
            * config.moe_layer_num
        )
        main_model_a_params = main_model_params - ffn_experts_not_a_params
        vocab_emb_a_params = config.main_dim * (config.vocab_size + config.aux_dim)
        total_a_params = aux_model_a_params + main_model_a_params + vocab_emb_a_params
        activated_params_result = {
            "aux_model_a_params": aux_model_a_params,
            "main_model_a_params": main_model_a_params,
            "vocab_emb_a_params": vocab_emb_a_params,
            "total_a_params": total_a_params,
        }

        return params_result, activated_params_result

    def get_optimizer(self, config: Config):
        aux_lr = config.max_aux_lr
        main_lr = config.max_lr
        wd = config.weight_decay
        betas = config.adam_betas
        aux_list = ["aux_model", "main_to_aux", "main_keys", "main_bias"]
        aux_params = [
            param
            for name, param in self.named_parameters()
            if any(i in name for i in aux_list)
        ]
        main_params = [
            param
            for name, param in self.named_parameters()
            if all(i not in name for i in aux_list)
        ]
        param_groups = [
            {"params": aux_params, "lr": aux_lr, "weight_decay": wd},
            {"params": main_params, "lr": main_lr, "weight_decay": wd},
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
