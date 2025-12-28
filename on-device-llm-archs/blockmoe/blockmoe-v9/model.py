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
from utils import GroupedGEMM, grouped_gemm_func_original # type: ignore
from config import Config # type: ignore

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
        dual: bool = False,
    ):
        xq = rearrange(xq, "b s (h a) -> b h s a", h=self.attn_num)
        xk = rearrange(xk, "b s (h a) -> b h s a", h=self.attn_num)
        xv = rearrange(xv, "b s (h a) -> b h s a", h=self.attn_num)
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)
        kernel_options = config.kernel_options if config.flex_attn_spec_kernel else None
        if dual:
            xq = rearrange(xq, "(c b) h s a -> b h (s c) a", c=2)
            xk = rearrange(xk, "(c b) h s a -> b h (s c) a", c=2)
            xv = rearrange(xv, "(c b) h s a -> b h (s c) a", c=2)
        x_attn = flex_attention(
            xq,
            xk,
            xv,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )
        if dual:
            x_attn = rearrange(x_attn, "b h (s c) a -> (c b) s (h a)", c=2)
        else:
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


class AuxRouterP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.aux_dim
        attn_dim = config.aux_attn_dim
        an = config.aux_attn_num
        p_aux_ln = config.p_aux_layer_num

        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(p_aux_ln)])
        self.router_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.attention = Attention(RotaryEmb(attn_dim, config.theta_s), an)

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        mask_mod_type: int,
    ):
        n = config.p_block_size
        m = n + 1

        x = rearrange(x, "b (l n) d -> b l n d", n=n)
        b, l = x.shape[:2]
        router_token = self.router_token.expand(b, l, -1, -1)
        x = torch.cat([x, router_token], 2).flatten(1, 2)

        def mask_mod_1(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // m
            kv_b_idx = kv_idx // m
            if config.p_global_mask:
                block_mask = q_b_idx >= kv_b_idx
            else:
                block_mask = q_b_idx == kv_b_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return block_mask & doc_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // m
            kv_b_idx = kv_idx // m
            block_mask = q_b_idx == kv_b_idx
            return block_mask

        b, s = x.shape[:2]
        if mask_mod_type == 1:
            mask_mod = mask_mod_1
        if mask_mod_type == 2:
            mask_mod = mask_mod_2

        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)

        x = rearrange(x, "b (l m) d -> b l m d", m=m)
        x = x[:, :, -1]
        return x


class AuxRouterF(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.aux_dim
        attn_dim = config.aux_attn_dim
        an = config.aux_attn_num
        f_aux_ln = config.f_aux_layer_num

        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(f_aux_ln)])
        self.router_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)
        self.attention_p = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor | None,
        config: Config,
        mask_mod_type: int,
    ):
        if mask_mod_type == 1:
            n = config.f_block_size
            attention = self.attention
        if mask_mod_type == 2:
            n = config.p_block_size
            attention = self.attention_p
        m = n + 1

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

        b, s = x.shape[:2]
        if mask_mod_type == 1:
            mask_mod = mask_mod_1
        if mask_mod_type == 2:
            mask_mod = mask_mod_2

        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, attention, config)

        x = rearrange(x, "b (l m) d -> b l m d", m=m)
        x = x[:, :, -1]
        return x


class AuxModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.main_dim
        dim_a = config.aux_dim
        pen = config.p_routed_expert_num
        pge = config.p_group_expert_num
        pgn = config.p_router_group_num
        fen = config.f_routed_expert_num
        fge = config.f_group_expert_num
        fgn = config.f_router_group_num

        self.main_to_aux = nn.Linear(dim, dim_a, bias=False)
        self.p_aux_router = AuxRouterP(config)
        self.p_keys = nn.Parameter(torch.randn(pgn, dim_a, pge))
        self.p_query_norm = nn.RMSNorm(dim_a, eps=config.norm_eps)

        self.f_aux_router = AuxRouterF(config)
        self.f_keys = nn.Parameter(torch.randn(fgn, dim_a, fge))
        self.f_query_norm = nn.RMSNorm(dim_a, eps=config.norm_eps)

        self.register_buffer("p_stat", torch.zeros(pen))
        self.register_buffer("p_logits_max", torch.zeros(1))
        self.register_buffer("p_logits_mean", torch.zeros(1))
        self.p_count = 0

        self.register_buffer("f_stat", torch.zeros(fen))
        self.register_buffer("f_logits_max", torch.zeros(1))
        self.register_buffer("f_logits_mean", torch.zeros(1))
        self.f_count = 0

    def p_reset(self):
        self.p_stat -= self.p_stat
        self.p_count -= self.p_count
        self.p_logits_max -= self.p_logits_max
        self.p_logits_mean -= self.p_logits_mean

    def f_reset(self):
        self.f_stat -= self.f_stat
        self.f_count -= self.f_count
        self.f_logits_max -= self.f_logits_max
        self.f_logits_mean -= self.f_logits_mean

    def forward(
        self,
        x: torch.Tensor,
        p_doc: torch.Tensor | None,
        f_doc: torch.Tensor | None,
        config: Config,
        aux_mode: bool,
        stat_mode: bool,
    ):
        pen = config.p_routed_expert_num
        pbf = config.p_balance_loss_factor
        pbc = config.p_main_balance_loss_coeff
        pgt = config.p_group_expert_num_per_token
        pgn = config.p_router_group_num
        pge = config.p_group_expert_num
        pc = 1 if aux_mode else pbc

        fen = config.f_routed_expert_num
        fbf = config.f_balance_loss_factor
        fbc = config.f_main_balance_loss_coeff
        fgt = config.f_group_expert_num_per_token
        fgn = config.f_router_group_num
        fge = config.f_group_expert_num
        fc = 1 if aux_mode else fbc

        device = x.device
        rank = int(os.environ.get("RANK", -1))
        ddp = rank != -1
        mask_mod_type = 2 if aux_mode else 1
        x = self.main_to_aux(x)

        px = self.p_aux_router(x, p_doc, config, mask_mod_type)
        px = repeat(px, "b pl d -> r b pl d", r=pgn).flatten(1, 2)
        px = self.p_query_norm(px)
        p_logits = torch.bmm(px, self.p_keys)
        p_logits = rearrange(p_logits, "pgn bpl pge -> bpl pgn pge")
        p_values, p_indices_o = p_logits.topk(pgt, -1, sorted=False)
        p_offset = torch.arange(pgn, device=device)[None, :, None] * pge
        p_indices = (p_indices_o + p_offset).flatten()
        p_values = p_values.flatten()

        p_balance_loss = torch.tensor(0, device=device)
        p_values_aux = p_logits.flatten(1)
        p_values_bin = p_values_aux.softmax(-1).mean(0)
        if stat_mode:
            p_indices_bin_o = F.one_hot(p_indices, pen).type_as(x).mean(0) * pen
            self.p_stat += p_indices_bin_o
            self.p_count += 1
            self.p_logits_max += p_values_aux.amax(-1).mean().detach()
            self.p_logits_mean += p_values_aux.mean().detach()
            if ddp:
                dist.all_reduce(self.p_stat, op=dist.ReduceOp.AVG)
                dist.all_reduce(self.p_logits_max, op=dist.ReduceOp.AVG)
                dist.all_reduce(self.p_logits_mean, op=dist.ReduceOp.AVG)
        p_indices_bin = self.p_stat / self.p_count if self.p_count > 0 else 0
        p_balance_loss = (p_indices_bin * p_values_bin).sum() * pbf * pc

        fx = self.f_aux_router(x, f_doc, config, mask_mod_type)
        fx = repeat(fx, "b l d -> r b l d", r=fgn)
        fx = fx.roll(1, 2).flatten(1, 2)
        fx = self.f_query_norm(fx)
        f_logits = torch.bmm(fx, self.f_keys)
        f_logits = rearrange(f_logits, "fgn bl fge -> bl fgn fge")
        f_values, f_indices_o = f_logits.topk(fgt, -1, sorted=False)
        f_offset = torch.arange(fgn, device=device)[None, :, None] * fge
        f_indices = (f_indices_o + f_offset).flatten()
        f_values = f_values.flatten()

        f_balance_loss = torch.tensor(0, device=device)
        f_values_aux = f_logits.flatten(1)
        f_values_bin = f_values_aux.softmax(-1).mean(0)
        if stat_mode:
            f_indices_bin_o = F.one_hot(f_indices, fen).type_as(x).mean(0) * fen
            self.f_stat += f_indices_bin_o
            self.f_count += 1
            self.f_logits_max += f_values_aux.amax(-1).mean().detach()
            self.f_logits_mean += f_values_aux.mean().detach()
            if ddp:
                dist.all_reduce(self.f_stat, op=dist.ReduceOp.AVG)
                dist.all_reduce(self.f_logits_max, op=dist.ReduceOp.AVG)
                dist.all_reduce(self.f_logits_mean, op=dist.ReduceOp.AVG)
        f_indices_bin = self.f_stat / self.f_count if self.f_count > 0 else 0
        f_balance_loss = (f_indices_bin * f_values_bin).sum() * fbf * fc

        p_results = p_indices, p_values, p_balance_loss
        f_results = f_indices, f_values, f_balance_loss

        return p_results, f_results


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        p_dim = config.p_expert_dim
        ple = config.p_layer_expert_num
        f_dim = config.f_expert_dim
        fle = config.f_layer_expert_num
        dim = config.main_dim
        dim_s = config.shared_expert_dim
        self.dim = dim
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        if config.shared_expert:
            self.ffn_up = nn.Linear(dim, dim_s * 2, bias=False)
            self.ffn_down = nn.Linear(dim_s, dim, bias=False)

        if not config.pf_shared_expert:
            self.p_ffn_experts = nn.Parameter(torch.randn(3, ple, dim, p_dim))
        self.f_ffn_experts = nn.Parameter(torch.randn(3, fle, dim, f_dim))

        if config.token_keys:
            self.p_token_keys = nn.Parameter(torch.randn(dim, ple))
            self.f_token_keys = nn.Parameter(torch.randn(dim, fle))
        if config.token_router_bias:
            self.p_token_router_bias = nn.Parameter(torch.zeros(ple))
            self.f_token_router_bias = nn.Parameter(torch.zeros(fle))

    def forward(
        self,
        x_input: torch.Tensor,
        p_indices: torch.LongTensor,
        p_values: torch.Tensor,
        f_indices: torch.LongTensor,
        f_values: torch.Tensor,
        block_mask_d: BlockMask,
        attention: Attention,
        config: Config,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), self.dim, -1)
        x_attn = attention(xq, xk, xv, block_mask_d, config, True)
        x_attn_o = self.attn_o(x_attn)
        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)
        b = x_ffn.shape[0] // 2
        p_x_ffn, f_x_ffn = x_ffn[:b], x_ffn[b:]

        if config.token_keys:
            p_token_values = (p_x_ffn @ self.p_token_keys).flatten(0, 1)
            p_token_values = torch.gather(p_token_values, -1, p_indices)
            p_values = p_values + p_token_values

            f_token_values = (f_x_ffn @ self.f_token_keys).flatten(0, 1)
            f_token_values = torch.gather(f_token_values, -1, f_indices)
            f_values = f_values + f_token_values

        if config.token_router_bias:
            p_values = p_values + self.p_token_router_bias[p_indices]
            f_values = f_values + self.f_token_router_bias[f_indices]

        p_scores = p_values.sigmoid()
        p_scores = p_scores / p_scores.sum(-1, keepdim=True)
        p_indices = p_indices.flatten()
        p_scores = p_scores.flatten().unsqueeze(-1) * config.p_routed_scaling_factor
        p_ffn_experts = (
            self.f_ffn_experts if config.pf_shared_expert else self.p_ffn_experts
        )
        py = grouped_gemm_func(
                p_x_ffn, p_ffn_experts, p_indices, p_scores, config
            )
        
        f_scores = f_values.sigmoid()
        f_scores = f_scores / f_scores.sum(-1, keepdim=True)
        f_indices = f_indices.flatten()
        f_scores = f_scores.flatten().unsqueeze(-1) * config.f_routed_scaling_factor
        fy = grouped_gemm_func(
            f_x_ffn, self.f_ffn_experts, f_indices, f_scores, config
        )

        y = torch.cat([py, fy], dim=0) + x_ffn_input

        if config.shared_expert:
            x1, x2 = torch.split(self.ffn_up(x_ffn), config.shared_expert_dim, -1)
            x3 = F.silu(x1) * x2
            y_shared = self.ffn_down(x3)
            y = y + y_shared

        return y


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        dim = config.main_dim
        attn_dim = config.main_attn_dim
        an = config.main_attn_num
        ln = config.moe_layer_num
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense

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
        doc_d: torch.Tensor,
        p_indices: torch.LongTensor,
        p_values: torch.Tensor,
        f_indices: torch.LongTensor,
        f_values: torch.Tensor,
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

        n = config.p_block_size * 2

        def mask_mod_d(b, h, q_idx, kv_idx):
            pq = q_idx % 2 == 0
            pkv = kv_idx % 2 == 0
            fq = q_idx % 2 == 1
            fkv = kv_idx % 2 == 1
            causal_mask = q_idx >= kv_idx
            q_b_idx = q_idx // n
            kv_b_idx = kv_idx // n
            p_mask = q_b_idx >= kv_b_idx + 2
            f_mask = q_b_idx < kv_b_idx + 2
            doc_mask = doc_d[b, q_idx] == doc_d[b, kv_idx]
            mask = (
                pq & pkv & causal_mask
                | fq & (pkv & p_mask | fkv & causal_mask & f_mask)
            ) & doc_mask
            return mask

        b, s = x.shape[:2]
        block_mask_d = create_block_mask(
            mask_mod_d, b, None, 2 * s, 2 * s, _compile=True
        )
        x = repeat(x, "b s d -> (2 b) s d")

        for _, layer in enumerate(self.moe_layers):
            x = layer(
                x,
                p_indices,
                p_values,
                f_indices,
                f_values,
                block_mask_d,
                self.attention,
                config,
            )

        x = x[b:]
        if last_k > 0:
            for layer in self.last_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        return x


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        self.vocab_emb = nn.Embedding(config.vocab_size, dim)
        self.aux_model = AuxModel(config)
        self.main_model = MainModel(config)
        self.final_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        self.f_default_bias = nn.Parameter(torch.zeros(1, 1, 1))
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        input: torch.LongTensor,
        target: torch.LongTensor | None,
        config: Config,
        aux_mode: bool,
        stat_mode: bool,
    ):
        b = input.shape[0]
        device = input.device

        pl = config.p_seq_block_num
        pn = config.p_block_size
        pet = config.p_expert_num_per_token

        fl = config.f_seq_block_num
        fn = config.f_block_size
        fen = config.f_routed_expert_num
        fle = config.f_layer_expert_num
        fet = config.f_expert_num_per_token

        eot = input == config.eot_idx
        p_eot_m = rearrange(eot, "b (pl pn) -> b pl pn", pn=pn)
        p_doc_aux = None

        f_eot_m = rearrange(eot, "b (fl fn) -> b fl fn", fn=fn)
        f_eot_b = f_eot_m.any(-1, keepdim=True).expand(-1, -1, fn)
        f_doc_aux = None

        if not aux_mode:
            p_router_pad = torch.zeros(1, 1, 1, device=device).expand(b, pl, 1)
            p_eot_aux = torch.cat([p_eot_m, p_router_pad], -1).flatten(1)
            p_doc_aux = torch.zeros_like(p_eot_aux)
            p_doc_aux[:, 1:] = p_eot_aux.cumsum(-1)[:, :-1]

            f_router_pad = torch.zeros(1, 1, 1, device=device).expand(b, fl, 1)
            f_eot_aux = torch.cat([f_eot_m, f_router_pad], -1).flatten(1)
            f_doc_aux = torch.zeros_like(f_eot_aux)
            f_doc_aux[:, 1:] = f_eot_aux.cumsum(-1)[:, :-1]

        x = self.vocab_emb(input)
        p_results, f_results = self.aux_model(
            x, p_doc_aux, f_doc_aux, config, aux_mode, stat_mode
        )
        p_indices, p_values, p_balance_loss = p_results
        f_indices, f_values, f_balance_loss = f_results

        if aux_mode:
            balance_loss = p_balance_loss + f_balance_loss
            return balance_loss, p_balance_loss.detach(), f_balance_loss.detach()
        elif stat_mode:
            return

        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        doc_d = repeat(doc, "b s -> b (s 2)")

        p_indices = repeat(p_indices, "(bpl pet) -> (bpl pn) pet", pn=pn, pet=pet)
        p_values = repeat(p_values, "(bpl pet) -> (bpl pn) pet", pn=pn, pet=pet)

        f_indices = repeat(f_indices, "(bfl fet) -> (bfl fn) fet", fn=fn, fet=fet)
        f_values = repeat(f_values, "(bfl fet) -> (bfl fn) fet", fn=fn, fet=fet)
        f_default_idx = torch.arange(fen, fle, device=device)[None, :]
        f_mask = f_eot_b.roll(1, 1)
        f_mask[:, :, 1:] = f_eot_m.cumsum(-1)[:, :, :-1] > 0
        f_mask[:, 0, :] = True
        f_mask = f_mask.flatten()[:, None]
        f_indices = torch.where(f_mask, f_default_idx, f_indices)
        f_values = torch.where(f_mask, self.f_default_bias, f_values)

        y = self.main_model(
            x, doc, doc_d, p_indices, p_values, f_indices, f_values, config
        )
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(logits.flatten(0, -2), target.flatten())
        loss = main_loss + p_balance_loss + f_balance_loss
        return (
            loss,
            main_loss.detach(),
            p_balance_loss.detach(),
            f_balance_loss.detach(),
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
        main_model_params = sum(x.numel() for x in self.main_model.parameters())
        total_params = sum(x.numel() for x in self.parameters())
        vocab_emb_params = total_params - aux_model_params - main_model_params
        params_result = {
            "aux_model_params": aux_model_params,
            "main_model_params": main_model_params,
            "vocab_emb_params": vocab_emb_params,
            "total_params": total_params,
        }

        aux_model_a_params = aux_model_params
        p_ffn_experts_params = (
            3
            * config.main_dim
            * config.p_expert_dim
            * config.p_layer_expert_num
            * config.moe_layer_num
        )
        if config.p_shared_expert:
            p_ffn_experts_params += (
                3 * config.main_dim * config.p_shared_expert_dim * config.moe_layer_num
            )
        f_ffn_experts_not_a_params = 0
        if not config.f_dense:
            f_ffn_experts_not_a_params = (
                3
                * config.main_dim
                * config.f_expert_dim
                * config.f_routed_expert_num
                * config.moe_layer_num
            )
        main_model_a_params = (
            main_model_params - p_ffn_experts_params - f_ffn_experts_not_a_params
        )
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
