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
            kv_token = (kv_idx % m) < n
            causal_mask = q_b_idx > kv_b_idx
            block_mask = q_b_idx == kv_b_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return (causal_mask & kv_token | block_mask) & doc_mask

        def mask_mod_3(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // m
            kv_b_idx = kv_idx // m
            block_mask = q_b_idx == kv_b_idx
            return block_mask

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
        x = x[:, :, n:]
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
        t = config.router_sum_steps
        w = config.world_size
        b = config.router_batch_size

        self.main_to_aux = nn.Linear(dim, dim_a, bias=False)
        self.aux_router = AuxRouter(config)
        self.keys = nn.Parameter(torch.randn(rn * gn, dim_a, ge))
        self.query_norm = nn.RMSNorm(dim_a, eps=config.norm_eps)

        self.register_buffer("stat", torch.zeros(rn, en))
        self.register_buffer("logits_max", torch.zeros(1))
        self.register_buffer("logits_mean", torch.zeros(1))
        self.register_buffer("seq_stat", torch.zeros(t, w, b, rn, en))
        self.register_buffer("reweight", torch.zeros(t, w, b))
        self.count = 0
        self.step_count = 0

    def reset(self):
        self.stat -= self.stat
        self.count -= self.count
        self.logits_max -= self.logits_max
        self.logits_mean -= self.logits_mean

    def reweight_func(self, config: Config):
        r_max = config.reweight_max
        r_min = config.reweight_min
        wt = config.reweight_steps
        n = config.router_num * config.routed_expert_num
        m = config.router_sum_steps * config.world_size * config.router_batch_size

        w0 = torch.ones_like(self.reweight).flatten().unsqueeze(-1)
        seq_stat = self.seq_stat.flatten(0, 2).flatten(1)
        b0 = (seq_stat * w0).sum(0, keepdim=True)
        b0 = b0 / b0.sum() * n
        w_list = [w0]
        b_list = [b0]
        for _ in range(wt):
            u = (seq_stat * b_list[-1]).mean(1, keepdim=True)
            w = w_list[-1] / (u + 1e-6)
            w = (w / w.sum() * m).clamp(min=r_min, max=r_max)
            b = (seq_stat * w).mean(0, keepdim=True)
            b = b / b.sum() * n
            w_list.append(w)
            b_list.append(b)
        w = w_list[-1]
        w = w / w.sum() * m
        self.reweight = w.reshape_as(self.reweight)
        print(f"b-1: {b_list[-1]}")

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        aux_mode: bool,
        router_mode: bool,
        pre_reweight_mode: bool,
        reweight_mode: bool,
        stat_mode: bool,
    ):
        rn = config.router_num
        rr = config.router_repeated_num
        en = config.routed_expert_num
        bf = config.balance_loss_factor
        bc = config.main_balance_loss_coeff
        gt = config.group_expert_num_per_token
        gn = config.router_group_num
        ge = config.group_expert_num
        b = config.router_batch_size
        rt = config.router_sum_steps
        device = x.device
        rank = int(os.environ.get("RANK", -1))
        ddp = rank != -1
        c = 1 if aux_mode else bc
        shift = 0 if aux_mode else config.predict_type
        mask_mod_type = 2 if config.aux_spec_attn else 1
        if aux_mode:
            mask_mod_type = 3

        x = self.main_to_aux(x)
        x = self.aux_router(x, doc, config, mask_mod_type)
        x = repeat(x, "b l rs d -> (rs rr) b l d", rr=rr)
        x = x.roll(shift, 2).flatten(1, 2)
        x = self.query_norm(x)
        logits = torch.bmm(x, self.keys)

        indices_o = logits.topk(gt, -1, sorted=False)[1]
        indices = rearrange(indices_o, "(rn gn) bl gt -> rn bl gn gt", rn=rn)
        offset = torch.arange(gn, device=device)[None, None, :, None] * ge
        indices = indices + offset
        indices = indices.flatten(1)
        indices_bin_o = F.one_hot(indices, en).type_as(x).mean(1) * en
        values = torch.gather(logits, -1, indices_o)
        values = rearrange(values, "(rn gn) bl gt -> rn (bl gn gt)", rn=rn)

        if pre_reweight_mode:
            t = self.step_count
            seq_indices = rearrange(indices, "rn (b c) -> b rn c", b=b)
            seq_indices_bin = F.one_hot(seq_indices, en).type_as(x).mean(2) * en
            if ddp:
                dist.all_gather_into_tensor(self.seq_stat[t], seq_indices_bin)
            else:
                self.seq_stat[t, 0] = seq_indices_bin
            self.step_count += 1
            if self.step_count == rt:
                self.reweight_func(config)
                if ddp:
                    dist.all_reduce(self.reweight, op=dist.ReduceOp.AVG)
                self.step_count -= self.step_count

        if reweight_mode and config.grad_reweight:
            w = self.reweight[self.step_count, rank][None, :, None]
            values = rearrange(values, "rn (b c) -> rn b c", b=b)
            values = values * w + values.detach() * (1 - w)
            values = values.flatten(1)
            self.step_count += 1
            if self.step_count == rt:
                self.step_count -= self.step_count

        balance_loss = torch.tensor(0, device=device)
        if router_mode:
            values_aux = rearrange(logits, "(rn gn) bl ge -> rn bl (gn ge)", rn=rn)
            values_bin = values_aux.softmax(-1).mean(1).flatten()

            if stat_mode:
                self.logits_max += values_aux.amax(-1).mean().detach()
                self.logits_mean += values_aux.mean().detach()
                self.stat += indices_bin_o
                self.count += 1
                if ddp:
                    dist.all_reduce(self.stat, op=dist.ReduceOp.AVG)
                    dist.all_reduce(self.logits_max, op=dist.ReduceOp.AVG)
                    dist.all_reduce(self.logits_mean, op=dist.ReduceOp.AVG)

            indices_bin = self.stat.flatten() / self.count
            balance_loss = (indices_bin * values_bin).sum() * bf * c

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
        if config.token_router_bias:
            self.token_router_bias = nn.Parameter(torch.zeros(le))
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

        if config.token_keys:
            token_values = (x_ffn @ self.token_keys).flatten(0, 1)
            token_values = torch.gather(token_values, -1, indices)
            values = values + token_values
        if config.token_router_bias:
            values = values + self.token_router_bias[indices]

        scores = values.sigmoid()
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
    def __init__(self, config: Config, router: bool):
        super().__init__()

        dim = config.main_dim
        attn_dim = config.main_attn_dim
        ln = config.router_moe_layer_num if router else config.moe_layer_num
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
        self.main_model = MainModel(config, False)
        self.router_main_model = MainModel(config, True)
        self.final_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        self.default_bias = nn.Parameter(torch.zeros(rn, 1, 1))
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        input: torch.LongTensor,
        target: torch.LongTensor | None,
        config: Config,
        aux_mode: bool,
        router_mode: bool,
        peek_mode: bool,
        pre_reweight_mode: bool,
        reweight_mode: bool,
        stat_mode: bool,
    ):
        b, s = input.shape
        n = config.block_size
        l = config.seq_block_num
        rs = config.router_size
        en = config.routed_expert_num
        le = config.layer_expert_num
        et = config.expert_num_per_token
        p = config.predict_type
        device = input.device

        eot = input == config.eot_idx
        eot_m = rearrange(eot, "b (l n) -> b l n", n=n)
        eot_b = eot_m.any(-1, keepdim=True).expand(-1, -1, n)
        doc_aux = None
        if not aux_mode:
            router_pad = torch.zeros(1, 1, 1, device=device).expand(b, l, rs)
            eot_aux = torch.cat([eot_m, router_pad], -1).flatten(1)
            doc_aux = torch.zeros_like(eot_aux)
            doc_aux[:, 1:] = eot_aux.cumsum(-1)[:, :-1]

        x = self.vocab_emb(input)
        indices, values, balance_loss = self.aux_model(
            x,
            doc_aux,
            config,
            aux_mode,
            router_mode,
            pre_reweight_mode,
            reweight_mode,
            stat_mode,
        )
        if aux_mode:
            return balance_loss
        if peek_mode:
            return

        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        indices = repeat(indices, "rn (bl et) -> rn (bl n) et", n=n, et=et)
        values = repeat(values, "rn (bl et) -> rn (bl n) et", n=n, et=et)
        default_idx = torch.arange(en, le, device=device)[None, None, :]
        mask = eot_b.roll(1, 1) | eot_b.roll(p, 1)
        mask[:, :, 1:] = eot_m.cumsum(-1)[:, :, :-1] > 0
        mask[:, [0, p - 1], :] = True
        mask = mask.flatten()[None, :, None]
        indices = torch.where(mask, default_idx, indices)
        values = torch.where(mask, self.default_bias, values)
        if not router_mode:
            values = values.detach()
            if config.no_router_values:
                values -= values

        main_model = self.router_main_model if router_mode else self.main_model
        y = main_model(x, doc, indices, values, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(
            logits.flatten(0, -2), target.flatten(), reduction="none"
        )

        if reweight_mode and not config.grad_reweight:
            rank = int(os.environ.get("RANK", -1))
            w = self.aux_model.reweight[self.aux_model.step_count, rank]
            w = repeat(w, "b -> (b s)", s=s)
            main_loss = (main_loss * w).mean()
            self.aux_model.step_count += 1
            if self.aux_model.step_count == config.router_sum_steps:
                self.aux_model.step_count -= self.aux_model.step_count
        else:
            main_loss = main_loss.mean()

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
        router_main_model_params = sum(
            x.numel() for x in self.router_main_model.parameters()
        )
        total_params = sum(x.numel() for x in self.parameters())
        vocab_emb_params = (
            total_params
            - aux_model_params
            - main_model_params
            - router_main_model_params
        )
        main_total_params = total_params - router_main_model_params
        router_total_params = total_params - main_model_params
        params_result = {
            "aux_model_params": aux_model_params,
            "main_model_params": main_model_params,
            "router_main_model_params": router_main_model_params,
            "vocab_emb_params": vocab_emb_params,
            "main_total_params": main_total_params,
            "router_total_params": router_total_params,
        }

        aux_router_params = sum(
            x.numel() for x in self.aux_model.aux_router.parameters()
        )
        aux_model_a_params = aux_model_params + int(
            aux_router_params * config.router_size / config.block_size
        )
        c = 3 * config.main_dim * config.expert_dim * config.routed_expert_num
        experts_not_a_params = c * config.moe_layer_num
        router_experts_not_a_params = c * config.router_moe_layer_num
        main_model_a_params = main_model_params - experts_not_a_params
        router_model_a_params = router_main_model_params - router_experts_not_a_params
        vocab_emb_a_params = config.main_dim * config.vocab_size
        other_a_params = aux_model_a_params + vocab_emb_a_params
        main_total_a_params = other_a_params + main_model_a_params
        router_total_a_params = other_a_params + router_model_a_params
        activated_params_result = {
            "aux_model_a_params": aux_model_a_params,
            "main_model_a_params": main_model_a_params,
            "router_model_a_params": router_model_a_params,
            "vocab_emb_a_params": vocab_emb_a_params,
            "main_total_a_params": main_total_a_params,
            "router_total_a_params": router_total_a_params,
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
