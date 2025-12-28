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


class AuxTokenModel(nn.Module):
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
        mask_mod_type: int,
    ):
        n = config.block_size
        rs = config.router_size
        m = n + rs

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
        x = x[:, :, n:].flatten(1, 2)
        return x


class AuxBlockModel(nn.Module):
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
        doc: torch.Tensor,
        config: Config,
        mask_mod_type: int,
    ):
        rs = config.router_size

        def mask_mod_1(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return causal_mask & doc_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // rs
            kv_b_idx = kv_idx // rs
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & doc_mask

        def mask_mod_3(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // rs
            kv_b_idx = kv_idx // rs
            q_r_idx = q_idx % rs
            kv_r_idx = kv_idx % rs
            causal_mask = q_b_idx > kv_b_idx
            seq_mask = q_r_idx == kv_r_idx
            block_mask = q_b_idx == kv_b_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & seq_mask & doc_mask | block_mask

        def mask_mod_4(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // rs
            kv_b_idx = kv_idx // rs
            block_mask = q_b_idx == kv_b_idx
            return block_mask

        b, s = x.shape[:2]
        if mask_mod_type == 1:
            mask_mod = mask_mod_1
        if mask_mod_type == 2:
            mask_mod = mask_mod_2
        if mask_mod_type == 3:
            mask_mod = mask_mod_3
        if mask_mod_type == 4:
            mask_mod = mask_mod_4
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        return x


class AuxModel(nn.Module):
    def __init__(
        self, config: Config, keys: torch.Tensor | None, aux_predict_model: bool
    ):
        super().__init__()

        if aux_predict_model:
            self.aux_t = config.aux_p_token_layer_num
            self.aux_b = config.aux_p_block_layer_num
        else:
            self.aux_t = config.aux_token_layer_num
            self.aux_b = config.aux_block_layer_num
        rn = config.router_num
        rs = config.router_size
        dim = config.main_dim
        dim_a = config.aux_dim
        en = config.routed_expert_num
        ge = config.group_expert_num
        gn = config.router_group_num

        self.main_to_aux = nn.Linear(dim, dim_a, bias=False)
        if self.aux_t > 0:
            self.aux_token_model = AuxTokenModel(config, self.aux_t)
        if self.aux_b > 0:
            self.aux_block_model = AuxBlockModel(config, self.aux_b)
        if keys is None:
            self.keys = nn.Parameter(torch.randn(rn * gn, dim_a, ge))
            self.router_token = nn.Parameter(torch.randn(1, 1, rs, dim_a))
        else:
            self.keys = keys
        self.query_norm = nn.RMSNorm(dim_a, eps=config.norm_eps)
        self.register_buffer("stat", torch.zeros(rn * en))
        self.register_buffer("count", torch.zeros(1))
        if config.balance_loss_free:
            self.register_buffer("bias", torch.zeros(rn * en))
        self.register_buffer("logits_max", torch.zeros(1))
        self.register_buffer("logits_mean", torch.zeros(1))

    def update(self, config: Config):
        error = self.stat / self.count - 1
        if config.balance_loss_sign:
            error = torch.sign(error)
        self.bias -= error * config.balance_update_rate

    def reset(self):
        self.stat -= self.stat
        self.count -= self.count
        self.logits_max -= self.logits_max
        self.logits_mean -= self.logits_mean

    def forward(
        self,
        x: torch.Tensor,
        router_token: torch.Tensor | None,
        token_doc: torch.Tensor | None,
        block_doc: torch.Tensor | None,
        config: Config,
        aux_mode: bool,
        balance_loss_mode: bool,
        predict_mode: bool,
        token_mask_type: int,
        block_mask_type: int,
    ):
        n = config.block_size
        rs = config.router_size
        rn = config.router_num
        rr = config.router_repeated_num
        en = config.routed_expert_num
        bf = config.balance_loss_factor
        gt = config.group_expert_num_per_token
        gn = config.router_group_num
        ge = config.group_expert_num
        device = x.device
        bc = (
            config.p_main_balance_loss_coeff
            if predict_mode
            else config.main_balance_loss_coeff
        )
        if aux_mode:
            token_mask_type = 3
            block_mask_type = 4

        x = self.main_to_aux(x)
        x = rearrange(x, "b (l n) d -> b l n d", n=n)
        b, l = x.shape[:2]
        if router_token is None:
            router_token = self.router_token.expand(b, l, -1, -1)
        if self.aux_t > 0:
            x = torch.cat([x, router_token], 2).flatten(1, 2)
            x = self.aux_token_model(x, token_doc, config, token_mask_type)
        else:
            x = router_token.flatten(1, 2)
        if self.aux_b > 0:
            x = self.aux_block_model(x, block_doc, config, block_mask_type)
        x = repeat(x, "b (l rs) d -> (rs rr) b l d", rs=rs, rr=rr)
        if predict_mode:
            x = x.roll(1, 2)
        x = x.flatten(1, 2)
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

        return indices, values_f, values, x, balance_loss


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
        dim_a = config.aux_dim
        rn = config.router_num
        rs = config.router_size
        self.vocab_emb = nn.Embedding(config.vocab_size, dim)
        self.aux_model = AuxModel(config, None, False)
        if config.aux_predict_model:
            self.aux_predict_model = AuxModel(config, self.aux_model.keys, True)
        self.main_model = MainModel(config)
        self.final_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        self.default_bias = nn.Parameter(torch.zeros(rn, 1, 1))
        self.default_router_token = nn.Parameter(torch.randn(1, 1, rs, dim_a))
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        input: torch.LongTensor,
        target: torch.LongTensor | None,
        config: Config,
        aux_mode: bool,
        predict_mode: bool,
        bl_mode: bool,
    ):
        b = input.shape[0]
        n = config.block_size
        l = config.seq_block_num
        rn = config.router_num
        gn = config.router_group_num
        rs = config.router_size
        en = config.routed_expert_num
        le = config.layer_expert_num
        et = config.expert_num_per_token
        tm_type = config.aux_token_mask_type
        bm_type = config.aux_block_mask_type
        ptm_type = config.aux_p_token_mask_type
        pbm_type = config.aux_p_block_mask_type
        device = input.device
        m = n + rs

        t_doc = pt_doc = b_doc = pb_doc = None
        eot_o = input == config.eot_idx
        if not aux_mode:
            eot = rearrange(eot_o, "b (l n) -> b l n", n=n)
            router_pad = torch.zeros(1, 1, 1, device=device).expand(b, l, rs)
            if tm_type != 3:
                t_eot = torch.cat([router_pad, eot], -1).flatten(1)
                t_doc = torch.zeros_like(t_eot)
                t_doc[:, 1:] = t_eot.cumsum(-1)[:, :-1]
                t_doc = rearrange(t_doc, "b (l m) -> b l m", m=m)
                t_doc = torch.cat([t_doc[:, :, rs:], t_doc[:, :, :rs]], -1).flatten(1)
            if ptm_type != 3 and config.aux_predict_model:
                pt_eot = torch.cat([eot, router_pad], -1).flatten(1)
                pt_doc = torch.zeros_like(pt_eot)
                pt_doc[:, 1:] = pt_eot.cumsum(-1)[:, :-1]
            b_eot = eot.any(-1)
            bc_eot = b_eot.cumsum(-1)
            if bm_type != 4:
                b_doc = torch.zeros_like(b_eot)
                b_doc[:, 1:] = bc_eot[:, :-1]
                b_doc = repeat(b_doc, "b l -> b (l rs)", rs=rs)
            if pbm_type != 4 and config.aux_predict_model:
                pb_doc = bc_eot
                pb_doc = repeat(pb_doc, "b l -> b (l rs)", rs=rs)

        x = self.vocab_emb(input)
        indices, values_f, values, router_x, balance_loss = self.aux_model(
            x, None, t_doc, b_doc, config, aux_mode, bl_mode, False, tm_type, bm_type
        )
        if aux_mode:
            return balance_loss

        p_balance_loss = torch.tensor(0, device=device)
        router_predict_loss = torch.tensor(0, device=device)
        if config.aux_predict_model:
            router_token = rearrange(
                router_x, "(rs rr) (b l) d -> rr b l rs d", rs=rs, l=l
            )[0]
            b_eot_e = b_eot[:, :, None, None]
            router_token = torch.where(b_eot_e, self.default_router_token, router_token)

            p_indices, p_values_f, p_values, _, p_balance_loss = self.aux_predict_model(
                x,
                router_token,
                pt_doc,
                pb_doc,
                config,
                False,
                bl_mode,
                True,
                ptm_type,
                pbm_type,
            )

            router_predict_loss = F.cross_entropy(
                p_values_f.flatten(0, -2),
                values_f.flatten(0, -2).detach(),
                reduction="none",
            )
            router_mask = torch.ones(l, device=device)
            router_mask[0] = 0
            c = rn * gn * b
            router_mask = repeat(router_mask, "l -> (c l)", c=c)
            router_predict_loss = (
                (router_predict_loss * router_mask).sum()
                / router_mask.sum()
                * config.router_predict_loss_factor
            )

        doc = torch.zeros_like(eot_o)
        doc[:, 1:] = eot_o.cumsum(-1)[:, :-1]
        if predict_mode and config.aux_predict_model:
            f_indices = p_indices
            f_values = p_values
        else:
            f_indices = indices
            f_values = values

        f_indices = repeat(f_indices, "rn (bl et) -> rn (bl n) et", n=n, et=et)
        f_values = repeat(f_values, "rn (bl et) -> rn (bl n) et", n=n, et=et)
        default_idx = torch.arange(en, le, device=device)[None, None, :]
        mask = torch.zeros_like(eot)
        mask[:, :, 1:] = eot.cumsum(-1)[:, :, :-1] > 0
        mask[:, 0, :] = True
        mask = mask.flatten()[None, :, None]
        f_indices = torch.where(mask, default_idx, f_indices)
        f_values = torch.where(mask, self.default_bias.sigmoid(), f_values)

        y = self.main_model(x, doc, f_indices, f_values, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(logits.flatten(0, -2), target.flatten())
        loss = main_loss + balance_loss + p_balance_loss + router_predict_loss
        return (
            loss,
            main_loss.detach(),
            balance_loss.detach(),
            p_balance_loss.detach(),
            router_predict_loss.detach(),
        )

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
        aux_p_model_params = 0
        if config.aux_predict_model:
            aux_p_model_params = sum(
                x.numel() for x in self.aux_predict_model.parameters()
            )
        main_model_params = sum(x.numel() for x in self.main_model.parameters())
        total_params = sum(x.numel() for x in self.parameters())
        vocab_emb_params = (
            total_params - aux_model_params - aux_p_model_params - main_model_params
        )
        params_result = {
            "aux_model_params": aux_model_params,
            "aux_p_model_params": aux_p_model_params,
            "main_model_params": main_model_params,
            "vocab_emb_params": vocab_emb_params,
            "total_params": total_params,
        }

        aux_token_params = sum(
            x.numel() for x in self.aux_model.aux_token_model.parameters()
        )
        aux_model_a_params = int(
            aux_token_params * (config.router_size / config.block_size + 1)
            + (aux_model_params - aux_token_params)
            * config.router_size
            / config.block_size
        )
        aux_p_model_a_params = 0
        if config.aux_predict_model:
            aux_p_token_params = sum(
                x.numel() for x in self.aux_predict_model.aux_token_model.parameters()
            )
            aux_p_model_a_params = int(
                aux_p_token_params * (config.router_size / config.block_size + 1)
                + (aux_p_model_params - aux_p_token_params)
                * config.router_size
                / config.block_size
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
        total_a_params = (
            aux_model_a_params
            + aux_p_model_a_params
            + main_model_a_params
            + vocab_emb_a_params
        )
        activated_params_result = {
            "aux_model_a_params": aux_model_a_params,
            "aux_p_model_a_params": aux_p_model_a_params,
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
            param
            for name, param in self.named_parameters()
            if ("aux_model" not in name and "aux_predict_model" not in name)
        ]
        aux_params = [
            param
            for name, param in self.named_parameters()
            if ("aux_model" in name or "aux_predict_model" in name)
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
