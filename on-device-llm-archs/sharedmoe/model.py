import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.autograd import Function
import numpy as np
from einops import rearrange
from dataclasses import dataclass
from typing import List, Tuple, Optional
from grouped_gemm import GroupedGEMM

flex_attention = torch.compile(flex_attention, dynamic=False)


@dataclass
class Config:
    layer_num: int = 8
    block_layer_num: int = 8
    seq_len: int = 8 * 1024
    vocab_size: int = 50304
    main_dim: int = 768
    attn_head_dim: int = 128
    q_head_num: int = 6
    kv_head_num: int = 3
    ffn_hidden_dim: int = 1536
    attn_expert_dim: int = 128
    attn_o_expert_dim: int = 128
    ffn_expert_dim: int = 256
    main_router_part_dim: int = 128
    router_mid_dim: int = 64
    attn_router_dim: int = 48
    attn_o_router_dim: int = 48
    ffn_router_dim: int = 80
    gate_dim: int = 16
    attn_top_k_num: int = 1
    attn_o_top_k_num: int = 1
    ffn_top_k_num: int = 4
    dense_layer_list: List[bool] = [True] * layer_num
    moe_layer_list: List[bool] = [True] * layer_num
    dense_layer_id_list = list(np.cumsum(dense_layer_list) - 1)  # can be customized

    assert max(dense_layer_id_list) < block_layer_num
    moe_layer_id_list = list(np.cumsum(moe_layer_list) - 1)
    dense_layer_num = sum(dense_layer_list)
    moe_layer_num = sum(moe_layer_list)
    layer_attn_expert_num = (
        (q_head_num + 2 * kv_head_num) * attn_head_dim // attn_expert_dim
    )
    layer_attn_o_expert_num = q_head_num * attn_head_dim // attn_o_expert_dim
    layer_ffn_expert_num = ffn_hidden_dim // ffn_expert_dim
    attn_expert_num = block_layer_num * layer_attn_expert_num
    attn_o_expert_num = block_layer_num * layer_attn_o_expert_num
    ffn_expert_num = block_layer_num * layer_ffn_expert_num
    linear_expert_num = attn_expert_num + attn_o_expert_num
    total_expert_num = linear_expert_num + ffn_expert_num
    attn_query_dim = attn_router_dim + gate_dim
    attn_o_query_dim = attn_o_router_dim + gate_dim
    ffn_query_dim = ffn_router_dim + gate_dim
    attn_dim_list = [
        q_head_num * attn_head_dim,
        kv_head_num * attn_head_dim,
        kv_head_num * attn_head_dim,
    ]

    init_std: float = 0.02
    gain_init_std: float = 0.002
    learning_rate: float = 6e-4
    aux_loss_factor: float = 0.01
    theta: float = 10000
    weight_decay: float = 0.1
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    norm_eps: float = 1e-6
    adam_eps: float = 1e-8
    window_len: int = 4096
    dtype: torch.dtype = torch.bfloat16
    router_dtype: torch.dtype = torch.float32
    grouped_gemm_option: str = "gmm"  # options: "pad_bmm", "gmm", "cutlass_gmm"

    main_router_proj: bool = False
    expert_grad_detach: bool = True
    router_grad_detach: bool = False
    attn_moe: bool = True
    attn_o_moe: bool = True
    ffn_moe: bool = True
    expert_bias: bool = False
    query_bias: bool = True
    router_bias: bool = True
    gate_gain: bool = True
    gate_inner_bias: bool = True
    gate_outer_bias: bool = True
    gate_norm: bool = False
    tied_vocab_emb: bool = True
    qk_norm: bool = False
    flex_attn_spec_kernel: bool = True
    unified_aux_loss: bool = True
    permute: bool = False  # TODO
    gain_normal_init: bool = False
    gain_ones_init: bool = False


class BlockPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn_pool = nn.Parameter(
            torch.zeros(
                config.attn_expert_num,
                config.main_dim,
                config.attn_expert_dim,
            )
        )

        self.attn_o_pool = nn.Parameter(
            torch.zeros(
                config.attn_o_expert_num,
                config.attn_o_expert_dim,
                config.main_dim,
            )
        )

        self.ffn_pool = nn.Parameter(
            torch.zeros(
                config.ffn_expert_num,
                3,
                config.main_dim,
                config.ffn_expert_dim,
            )
        )

        if config.expert_bias:
            self.attn_bias = nn.Parameter(
                torch.zeros(
                    config.attn_expert_num,
                    config.attn_expert_dim,
                )
            )

            self.attn_o_bias = nn.Parameter(
                torch.zeros(
                    config.attn_o_expert_num,
                    config.main_dim,
                )
            )

            self.ffn_bias = nn.Parameter(
                torch.zeros(
                    config.ffn_expert_num,
                    config.main_dim,
                )
            )


class VecPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        if config.attn_moe:
            self.attn_key_pool = nn.Parameter(
                torch.zeros(
                    config.moe_layer_num,
                    config.attn_query_dim + 4,
                    config.attn_expert_num,
                )
            )

        if config.attn_o_moe:
            self.attn_o_key_pool = nn.Parameter(
                torch.zeros(
                    config.moe_layer_num,
                    config.attn_o_query_dim + 4,
                    config.attn_o_expert_num,
                )
            )

        if config.ffn_moe:
            self.ffn_key_pool = nn.Parameter(
                torch.zeros(
                    config.moe_layer_num,
                    config.ffn_query_dim + 4,
                    config.ffn_expert_num,
                )
            )


class RotaryEmb(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        device = f"cuda:{torch.cuda.current_device()}"
        inv_freq = (1 / config.theta) ** (
            torch.arange(0, config.attn_head_dim, 2) / config.attn_head_dim
        )
        freqs = torch.outer(torch.arange(config.seq_len), inv_freq)[None, None, :, :]
        self.freqs_cos = freqs.cos().to(device)
        self.freqs_sin = freqs.sin().to(device)

    def forward(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * self.freqs_cos + x2 * self.freqs_sin
        y2 = -x1 * self.freqs_sin + x2 * self.freqs_cos
        x_output = torch.cat((y1, y2), -1).type_as(x)
        return x_output


class Attention(nn.Module):
    def __init__(self, config: Config, rotary_emb: RotaryEmb):
        super().__init__()

        self.config = config
        self.rotary_emb = rotary_emb
        self.kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        }

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_mask: BlockMask,
    ):
        config = self.config
        dim = config.attn_head_dim
        eps = config.norm_eps
        dtype = config.dtype
        xq = rearrange(xq, "s (h d) -> 1 h s d", d=dim)
        xk = rearrange(xk, "s (h d) -> 1 h s d", d=dim)
        xv = rearrange(xv, "s (h d) -> 1 h s d", d=dim)
        if config.qk_norm:
            xq = F.normalize(xq, dim=-1, eps=eps).to(dtype)
            xk = F.normalize(xk, dim=-1, eps=eps).to(dtype)
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)
        kernel_options = self.kernel_options if config.flex_attn_spec_kernel else None
        x_attn = flex_attention(
            xq,
            xk,
            xv,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=kernel_options,
        )
        x_attn = rearrange(x_attn, "c h s d -> (c s) (h d)")
        return x_attn


class Wbincount(Function):
    @staticmethod
    def forward(ctx, indices: torch.Tensor, weights: torch.Tensor, num_bins: int):
        w_bincount = torch.bincount(indices, weights=weights, minlength=num_bins)
        ctx.save_for_backward(indices)
        return w_bincount

    @staticmethod
    def backward(ctx, grad_w_bincount):
        indices = ctx.saved_tensors
        grad_weights = grad_w_bincount[indices]
        return (None, grad_weights, None)


class Router(nn.Module):
    def __init__(
        self,
        config: Config,
        moe_layer_id: int,
    ):
        super().__init__()
        self.config = config
        self.moe_layer_id = moe_layer_id
        if config.gate_norm:
            self.gate_factor = nn.Parameter(torch.zeros(3))

    def forward(self, query: torch.Tensor, key_pool: torch.Tensor, type: str):
        config = self.config
        moe_layer_id = self.moe_layer_id
        if type == "attn":
            type_id = 0
            router_dim = config.attn_router_dim
            query_dim = config.attn_query_dim
            top_k_num = config.attn_top_k_num
        if type == "attn_o":
            type_id = 1
            router_dim = config.attn_o_router_dim
            query_dim = config.attn_o_query_dim
            top_k_num = config.attn_o_top_k_num
        if type == "ffn":
            type_id = 2
            router_dim = config.ffn_router_dim
            query_dim = config.ffn_query_dim
            top_k_num = config.ffn_top_k_num

        router = torch.matmul(
            query[:, :router_dim],
            key_pool[moe_layer_id, :router_dim, :],
        )
        if config.router_bias:
            router += key_pool[moe_layer_id, -4, :].unsqueeze(0)
        if top_k_num == 1:
            router_score, router_index = torch.max(router, -1)
        else:
            router_score, router_index = torch.topk(router, top_k_num, -1, sorted=False)
        gate_score = router_score.clone()
        router_score = torch.sigmoid(router_score)

        if config.gate_dim != 0:
            gate = torch.matmul(
                query[:, router_dim:],
                key_pool[moe_layer_id, router_dim:query_dim, :],
            )
            grid = torch.arange(router_index.shape[0], device=router_index.device)
            if top_k_num != 1:
                grid = grid.unsqueeze(-1)
            gate_score += gate[grid, router_index]
        if config.gate_inner_bias:
            gate_score += key_pool[moe_layer_id, -3, router_index]
        gate_score = torch.sigmoid(gate_score)
        if config.gate_gain:
            gate_score *= key_pool[moe_layer_id, -2, router_index]
        if config.gate_outer_bias:
            gate_score += key_pool[moe_layer_id, -1, router_index]

        router_dtype = config.router_dtype
        assert query.dtype == router_dtype
        assert gate_score.dtype == router_dtype
        assert router_score.dtype == router_dtype

        if config.gate_norm:
            if gate_score.ndim == 2:
                gate_score = (
                    gate_score
                    / torch.sum(gate_score, -1, keepdim=True)
                    * self.gate_factor[type_id]
                )

        return (
            gate_score.flatten().unsqueeze(-1),
            router_score.flatten(),
            router_index.flatten(),
        )


class Layer(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_id: int,
        block_pool: BlockPool,
        vec_pool: VecPool,
        attention: Attention,
        grouped_gemm: Function,
        w_bincount: Function,
    ):
        self.config = config
        self.vec_pool = vec_pool
        self.block_pool = block_pool
        self.attention = attention
        self.w_bincount = w_bincount
        self.dense = config.dense_layer_list[layer_id]
        self.moe = config.moe_layer_list[layer_id]
        self.dense_layer_id = config.dense_layer_id_list[layer_id]
        moe_layer_id = config.moe_layer_id_list[layer_id]
        self.router = Router(config, moe_layer_id)
        self.grouped_gemm = grouped_gemm

        self.attn_norm = nn.RMSNorm(
            config.main_dim, eps=config.norm_eps, dtype=config.dtype
        )
        self.ffn_norm = nn.RMSNorm(
            config.main_dim, eps=config.norm_eps, dtype=config.dtype
        )

        if self.moe:
            if config.main_router_proj:
                if config.attn_moe:
                    self.attn_main_router_proj = nn.Linear(
                        config.main_dim, config.router_mid_dim, bias=False
                    )
                if config.ffn_moe:
                    self.ffn_main_router_proj = nn.Linear(
                        config.main_dim,
                        config.ffn_query_dim,
                        bias=config.query_bias,
                    )

            attn_router_input_dim = (
                config.router_mid_dim
                if config.main_router_proj
                else config.main_router_part_dim
            )
            if config.attn_moe:
                self.attn_multihead_router_proj = nn.Linear(
                    attn_router_input_dim,
                    config.layer_attn_expert_num * config.attn_query_dim,
                    bias=config.query_bias,
                )

            if config.attn_o_moe:
                self.attn_o_router_w = nn.Parameter(
                    torch.zeros(
                        config.layer_attn_o_expert_num,
                        config.attn_o_expert_dim,
                        config.attn_o_query_dim,
                    )
                )
                if config.query_bias:
                    self.attn_o_router_bias = nn.Parameter(
                        torch.zeros(
                            config.layer_attn_o_expert_num,
                            1,
                            config.attn_o_query_dim,
                        )
                    )

            if config.ffn_moe and not config.main_router_proj:
                self.ffn_router_proj = nn.Linear(
                    config.main_router_part_dim,
                    config.ffn_query_dim,
                    bias=config.query_bias,
                )

    def forward(
        self,
        x_input: torch.Tensor,
        block_mask: BlockMask,
        block_pool: Tuple,
        block_bias: Optional[Tuple],
    ):
        config = self.config
        dense_layer_id = self.dense_layer_id
        attn_num, attn_o_num, ffn_num = (
            config.layer_attn_expert_num,
            config.layer_attn_o_expert_num,
            config.layer_ffn_expert_num,
        )
        attn_pos, attn_o_pos, ffn_pos = (
            dense_layer_id * attn_num,
            dense_layer_id * attn_o_num,
            dense_layer_id * ffn_num,
        )
        router_dtype = config.router_dtype
        part_dim = config.main_router_part_dim
        attn_pool, attn_o_pool, ffn_pool = block_pool
        attn_bias, attn_o_bias, ffn_bias = (
            block_bias
            if config.expert_bias
            else (torch.tensor([0], requires_grad=False) for _ in range(3))
        )

        x = self.attn_norm(x_input)
        if self.dense:
            attn_w = self.block_pool.attn_pool[attn_pos : attn_pos + attn_num]
            attn_w = rearrange(attn_w, "h d e -> d (h e)")
            x_attn_in = torch.matmul(x, attn_w)
        if self.moe and config.attn_moe:
            x_r = x.detach() if config.router_grad_detach else x
            if config.main_router_proj:
                attn_router_input = self.attn_main_router_proj(x_r)
            else:
                attn_router_input = x_r[:, :part_dim]
            attn_router_input = attn_router_input.to(router_dtype)
            attn_query = self.attn_multihead_router_proj(attn_router_input)
            attn_query = rearrange(
                attn_query, "s (h d) -> (s h) d", d=config.attn_query_dim
            )
            attn_gate, attn_router, attn_index = self.router(
                attn_query, self.vec_pool.attn_key_pool, "attn"
            )
            x_attn_moe = self.grouped_gemm(
                x, attn_pool, attn_bias, attn_gate, attn_index, config, "attn"
            )
            x_attn_in = x_attn_in + x_attn_moe if self.dense else x_attn_moe
        xq, xk, xv = torch.split(x_attn_in, config.attn_dim_list, -1)
        x_attn = self.attention(xq, xk, xv, block_mask)

        if self.dense:
            attn_o_w = self.block_pool.attn_o_pool[attn_o_pos : attn_o_pos + attn_o_num]
            attn_o_w = rearrange(attn_w, "h e d -> (h e) d")
            x_attn_o = torch.matmul(x_attn, attn_o_w)
        if self.moe and config.attn_o_moe:
            x_attn = rearrange(x_attn, "s (h d) -> h s d", d=config.attn_o_expert_dim)
            x_attn_r = x_attn.detach() if config.router_grad_detach else x_attn
            attn_o_query = torch.bmm(x_attn_r.to(router_dtype), self.attn_o_router_w)
            if config.query_bias:
                attn_o_query += self.attn_o_router_bias
            attn_o_query = rearrange(attn_o_query, "h s d -> (s h) d")
            attn_o_gate, attn_o_router, attn_o_index = self.router(
                attn_o_query, self.vec_pool.attn_o_key_pool, "attn_o"
            )
            x_attn_o_moe = self.grouped_gemm(
                x_attn,
                attn_o_pool,
                attn_o_bias,
                attn_o_gate,
                attn_o_index,
                config,
                "attn_o",
            )
            x_attn_o = x_attn_o + x_attn_o_moe if self.dense else x_attn_o_moe

        x_ffn_input = x_attn_o + x_input
        x = self.ffn_norm(x_ffn_input)
        if self.dense:
            ffn_w = self.block_pool.ffn_pool[ffn_pos : ffn_pos + ffn_num]
            ffn_w = rearrange(ffn_w, "h c d e -> c d (h e)")
            x1 = torch.matmul(x, ffn_w[0])
            x2 = torch.matmul(x, ffn_w[1])
            x3 = F.silu(x1) * x2
            x_ffn = torch.matmul(x3, ffn_w[2].transpose(0, 1))
        if self.moe and config.ffn_moe:
            x_r = x.detach() if config.router_grad_detach else x
            if config.main_router_proj:
                ffn_query = self.ffn_main_router_proj(x_r.to(router_dtype))
            else:
                ffn_router_input = x_r[:, :part_dim].to(router_dtype)
                ffn_query = self.ffn_router_proj(ffn_router_input)
            ffn_gate, ffn_router, ffn_index = self.router(
                ffn_query, self.vec_pool.ffn_key_pool, "ffn"
            )
            x_ffn_moe = self.grouped_gemm(
                x, ffn_pool, ffn_bias, ffn_gate, ffn_index, config, "ffn"
            )
            x_ffn = x_ffn + x_ffn_moe if self.dense else x_ffn_moe
        x_output = x_ffn + x_ffn_input

        aux_loss = 0
        if not self.moe:
            return x_output, aux_loss
        score_list = []
        index_list = []
        num_list = []
        if config.attn_moe:
            score_list.append(attn_router)
            index_list.append(attn_index)
            num_list.append(config.attn_expert_num)
        if config.attn_o_moe:
            score_list.append(attn_o_router)
            if config.unified_aux_loss:
                attn_o_index += config.attn_expert_num
            index_list.append(attn_o_index)
            num_list.append(config.attn_o_expert_num)
        if config.ffn_moe:
            score_list.append(ffn_router)
            if config.unified_aux_loss:
                ffn_index += config.linear_expert_num
            index_list.append(ffn_index)
            num_list.append(config.ffn_expert_num)
        if score_list:
            if config.unified_aux_loss:
                score = torch.cat(score_list, 0)
                index = torch.cat(index_list, 0)
                bin_len = config.total_expert_num
                score_bin = self.w_bincount(index, score, bin_len)
                index_bin = torch.bincount(index, minlength=bin_len)
                aux_loss = (
                    torch.dot(score_bin, index_bin.type_as(score_bin))
                    / (torch.sum(score_bin) * index.numel())
                    * sum(num_list)
                )
            else:
                for score, index, num in zip(score_list, index_list, num_list):
                    score_bin = self.w_bincount(index, score, num)
                    index_bin = torch.bincount(index, minlength=num)
                    aux_loss_item = (
                        torch.dot(score_bin, index_bin.type_as(score_bin))
                        / (torch.sum(score_bin) * index.numel())
                        * num
                    )
                    aux_loss += aux_loss_item

        return x_output, aux_loss


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        rotary_emb = RotaryEmb(config)
        attention = Attention(config, rotary_emb)
        grouped_gemm = GroupedGEMM.apply
        w_bincount = Wbincount.apply
        self.block_pool = BlockPool(config)
        self.vec_pool = VecPool(config)
        self.model = nn.ModuleDict(
            {
                "vocab_emb": nn.Embedding(config.vocab_size, config.main_dim),
                "layers": nn.ModuleList(
                    [
                        Layer(
                            config,
                            layer_id,
                            self.block_pool,
                            self.vec_pool,
                            attention,
                            grouped_gemm,
                            w_bincount,
                        )
                        for layer_id in range(config.layer_num)
                    ]
                ),
                "lm_head": nn.Linear(config.main_dim, config.vocab_size, bias=False),
            }
        )
        self.config = config
        if config.tied_vocab_emb:
            self.model.vocab_emb.weight = self.model.lm_head.weight
        self.init_weights()

    def forward(self, idx: torch.LongTensor, targets: torch.Tensor):
        config = self.config
        device = idx.device
        docs = (idx == 50256).cumsum(0)
        seq_len = config.seq_len

        def doc_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < config.window_len
            return causal_mask & doc_mask & window_mask

        block_mask = create_block_mask(
            doc_causal_mask, None, None, seq_len, seq_len, device=device, _compile=True
        )
        x = self.model.vocab_emb(idx)

        dtype = torch.bfloat16 if config.grouped_gemm_option == "gmm" else config.dtype
        block_pool = [
            self.attn_pool.to(dtype),
            self.attn_o_pool.to(dtype),
            self.ffn_pool.to(dtype),
        ]
        if config.expert_grad_detach:
            block_pool = [pool.detach() for pool in block_pool]
        block_bias = None
        if config.expert_bias:
            block_bias = [
                self.attn_bias.to(config.dtype),
                self.attn_o_bias.to(config.dtype),
                self.ffn_bias.to(config.dtype),
            ]
            if config.grouped_gemm_option == "pad_bmm":
                block_bias = [bias.unsqueeze(1) for bias in block_bias]

        aux_loss = 0
        for _, layer in enumerate(self.model.layers):
            x, layer_aux_loss = layer(x, block_mask, block_pool, block_bias)
            aux_loss += layer_aux_loss

        x = F.normalize(x, dim=-1, eps=config.norm_eps)
        logits = self.model.lm_head(x)
        train_loss = F.cross_entropy(logits, targets)

        aux_loss *= config.aux_loss_factor
        loss = train_loss + aux_loss

        return loss, train_loss.detach(), aux_loss.detach()

    def init_weights(self):
        std = self.config.init_std
        gain_std = self.config.gain_init_std
        g = torch.Generator()
        g.manual_seed(42)

        for name, param in self.block_pool.named_parameters():
            if "pool" in name:
                nn.init.trunc_normal_(
                    param, mean=0, std=std, a=-3 * std, b=3 * std, generator=g
                )
            if "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.vec_pool.named_parameters():
            nn.init.trunc_normal_(
                param, mean=0, std=std, a=-3 * std, b=3 * std, generator=g
            )
            with torch.no_grad():
                nn.init.zeros_(param.data[:, -4:, :])
                if self.config.gain_normal_init:
                    nn.init.trunc_normal_(
                        param.data[:, -2, :],
                        mean=0,
                        std=gain_std,
                        a=-3 * gain_std,
                        b=3 * gain_std,
                        generator=g,
                    )
                if self.config.gain_ones_init:
                    nn.init.ones_(param.data[:, -2, :])

        for name, param in self.model.named_parameters():
            if "pool" in name:
                continue
            if "bias" in name:
                nn.init.zeros_(param)
            elif "gate_factor" in name:
                nn.init.zeros_(param)
                if self.config.gain_normal_init:
                    nn.init.trunc_normal_(
                        param,
                        mean=0,
                        std=gain_std,
                        a=-3 * gain_std,
                        b=3 * gain_std,
                        generator=g,
                    )
                if self.config.gain_ones_init:
                    nn.init.ones_(param)
            elif "norm" in name:
                nn.init.ones_(param)
            else:
                nn.init.trunc_normal_(
                    param, mean=0, std=std, a=-3 * std, b=3 * std, generator=g
                )

    def params_count(self):
        block_pool_params = sum(x.numel() for x in self.block_pool.parameters())
        vec_pool_params = sum(x.numel() for x in self.vec_pool.parameters())
        model_params = sum(
            param.numel()
            for name, param in self.model.named_parameters()
            if "pool" not in name
        )
        emb_num = int(2 - self.config.tied_vocab_emb)
        emb_params = emb_num * self.model.vocab_emb.weight.numel()
        total_params = block_pool_params + vec_pool_params + model_params
        total_params_no_emb = total_params - emb_params
        model_params_no_emb = model_params - emb_params

        result = {
            "total_params": total_params,
            "total_params_wo_emb": total_params_no_emb,
            "block_pool_params": block_pool_params,
            "vec_pool_params": vec_pool_params,
            "model_params": model_params,
            "model_params_wo_emb": model_params_no_emb,
            "emb_params": emb_params,
        }
        return result

    def activated_params_count(self):
        vec_pool_params = sum(x.numel() for x in self.vec_pool.parameters())
        model_params = sum(
            param.numel()
            for name, param in self.model.named_parameters()
            if not "pool" in name
        )
        emb_num = int(2 - self.config.tied_vocab_emb)
        emb_params = emb_num * self.model.vocab_emb.weight.numel()
        layer_attn_params = int(
            self.block_pool.attn_pool.numel() / self.config.block_layer_num
        )
        layer_attn_o_params = int(
            self.block_pool.attn_o_pool.numel() / self.config.block_layer_num
        )
        layer_ffn_params = int(
            self.block_pool.ffn_pool.numel() / self.config.block_layer_num
        )
        block_ffn_params = int(layer_ffn_params / self.config.layer_ffn_expert_num)
        layer_block_params = layer_attn_params + layer_attn_o_params + layer_ffn_params
        dense_params = self.config.dense_layer_num * layer_block_params
        moe_params = self.config.moe_layer_num * (
            self.config.attn_moe * layer_attn_params * self.config.attn_top_k_num
            + self.config.attn_o_moe
            * layer_attn_o_params
            * self.config.attn_o_top_k_num
            + self.config.ffn_moe * block_ffn_params * self.config.ffn_top_k_num
        )
        total_params = vec_pool_params + model_params + dense_params + moe_params
        total_params_no_emb = total_params - emb_params
        model_params_no_emb = model_params - emb_params

        result = {
            "total_params": total_params,
            "total_params_no_emb": total_params_no_emb,
            "router_params": vec_pool_params,
            "dense_params": dense_params,
            "moe_params": moe_params,
            "model_params": model_params,
            "model_params_no_emb": model_params_no_emb,
            "emb_params": emb_params,
        }
        return result

    def optimizer(self):
        param_dict = {name: param for name, param in self.named_parameters()}
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        betas = self.config.betas

        no_decay_str_list = ["bias", "key", "vocab_emb"]
        no_decay_params = [
            param
            for name, param in param_dict.items()
            if (any(str in name for str in no_decay_str_list) or param.shape[0] == 1)
        ]
        decay_params = [
            param
            for name, param in param_dict.items()
            if (
                all(str not in name for str in no_decay_str_list)
                and param.shape[0] != 1
            )
        ]

        optim_groups = [
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
            {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        ]

        if self.config.zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                optim_groups,
                optimizer_class=torch.optim.AdamW,
                betas=betas,
                foreach=True,
            )
        else:
            optimizer = torch.optim.AdamW(optim_groups, betas=betas, foreach=True)

        return optimizer
