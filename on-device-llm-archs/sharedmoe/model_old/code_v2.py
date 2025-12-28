import os
import math
import sys
import pandas as pd
import glob
import time
from contextlib import nullcontext
import numpy as np
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)
from torch.autograd import Function
from torch.profiler import profile, ProfilerActivity
from einops import rearrange, repeat
from fast_pytorch_kmeans import KMeans  # type: ignore


@dataclass
class Config:
    layer_num: int = 8
    seq_len: int = 8 * 1024
    vocab_size: int = 50304
    c_vocab_dim: int = 768

    main_dim: int = 1024  # TODO
    attn_q_dim: int = 1024
    attn_kv_dim: int = 512
    ffn_dim: int = 1024
    c_q_dim: int = 256
    c_kv_dim: int = 256
    c_ffn_dim: int = 512
    c_attn_o_dim: int = 512
    c_ffn_o_dim: int = 512
    c_mid_dim: int = 128
    fusion_mid_dim: int = 32
    linear_mid_dim: int = 64
    ffn_hidden_dim: int = 3 * 1024  # TODO

    b_in_dim: int = 64
    b_out_dim: int = 64
    b_attn_dim: int = 64
    b_ffn_dim: int = 64
    b_ffn_hidden_dim: int = 64
    b_query_dim: int = 32
    b_query_router_dim: int = 24
    b_query_gate_dim: int = 8

    unified_b_num: int = 1800
    linear_b_num: int = 1200
    in_linear_b_num: int = 800
    out_linear_b_num: int = 400
    ffn_b_num: int = 600
    q_key_c_num: int = 64
    k_key_c_num: int = 64
    v_key_c_num: int = 64
    ffn_in_key_c_num: int = 96
    attn_o_key_c_num: int = 96
    ffn_o_key_c_num: int = 96
    ffn_key_c_num: int = 128
    batch_key_num: int = 64
    batch_key_valid_num: int = 60  # TODO
    ffn_top_k_num: int = 4

    vanilla_first_layer: bool = True
    shared_c_proj: bool = True
    separate_c_qkv_proj: bool = True  # TODO
    routed_linear: bool = True
    two_layer_router: bool = True  # TODO
    c_lora: bool = True
    linear_lora: bool = True
    unified_b_pool: bool = True
    unified_linear_b_pool: bool = False
    paired_block_router: bool = True
    sampled_query: bool = True
    partitioned_query: bool = True
    in_query_fusion: bool = True
    out_query_fusion: bool = True
    query_bias: bool = True
    sole_gated_query: bool = True
    key_router_bias: bool = True
    key_c_router_bias: bool = True  # TODO
    key_gate_bias: bool = True
    gate_gain: bool = True
    gate_norm: bool = False
    block_inner_bias: bool = False
    block_outer_bias: bool = False
    out_linear_norm: bool = True
    out_linear_mul: bool = False
    out_proj_mul: bool = True
    bias: bool = False
    tied_vocab_emb: bool = True
    vocab_compressed: bool = True
    qk_norm: bool = False
    zero_optimizer: bool = False
    no_weight_decay_1d: bool = True
    model_compile: bool = False
    scaler: bool = True
    flex_attention_compile: bool = True
    residual_norm: bool = True
    residual_mul: bool = True
    flex_attn_kernel: bool = True
    no_main_vec: bool = True
    shared_linear: bool = False
    chain_of_query: bool = True

    init_std: float = 0.02
    theta: float = 10000.0
    block_lr: float = 4e-4
    vec_lr: float = 4e-4
    model_lr: float = 8e-4
    min_lr_ratio: float = 0.01
    scheduler: str = "wsd"
    dtype: torch.dtype = torch.bfloat16
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    norm_eps: float = 1e-6
    adam_eps: float = 1e-8
    window_len: int = 4096
    cluster_token_balance_factor: float = 0.01
    block_token_balance_factor: float = 0.01  # TODO
    cluster_key_balance_factor: float = 0.01
    ffn_balance_ratio: float = 1.5
    first_ratios: Tuple[float, float, float] = (0.75, 0.25, 0.0)  # TODO
    second_ratios: Tuple[float, float, float] = (0.0, 0.25, 0.75)

    b_per_router = 2 if paired_block_router else 1
    b_routed_dim = b_per_router * b_in_dim  # TODO
    in_proj_dim = c_q_dim + c_kv_dim + c_ffn_dim
    mid_in_dim = attn_q_dim + 2 * attn_kv_dim + ffn_dim
    mid_out_dim = attn_q_dim + ffn_dim
    out_proj_dim = c_attn_o_dim + c_ffn_o_dim
    proj_in_dim_list = [c_q_dim, c_kv_dim, c_ffn_dim, attn_q_dim, ffn_dim]
    proj_out_dim_list = [
        attn_q_dim,
        2 * attn_kv_dim,
        ffn_dim,
        c_attn_o_dim,
        c_ffn_o_dim,
    ]
    ffn_routed_num = ffn_dim // b_ffn_dim

    init_dim = main_dim if not no_main_vec and vanilla_first_layer else in_proj_dim
    output_dim = in_proj_dim if no_main_vec else main_dim
    vocab_in_dim = init_dim if not vocab_compressed else c_vocab_dim
    vocab_out_dim = output_dim if not vocab_compressed else c_vocab_dim
    routed_layer_num = layer_num - 1 if vanilla_first_layer else layer_num
    key_c_num_list = [
        q_key_c_num,
        k_key_c_num,
        v_key_c_num,
        ffn_in_key_c_num,
        attn_o_key_c_num,
        ffn_o_key_c_num,
        ffn_key_c_num,
    ]
    key_c_num_cumsum_list = np.cumsum(key_c_num_list)
    key_c_num_sum = key_c_num_cumsum_list[-1]
    total_linear_b_num = (
        linear_b_num if unified_linear_b_pool else in_linear_b_num + out_linear_b_num
    )
    key_factor_num = key_router_bias + b_per_router * (key_gate_bias + gate_gain)

    def __post_init__(self):
        self.routed_num_list = [
            dim // self.b_routed_dim for dim in self.proj_in_dim_list
        ]
        self.head_num_list = [dim // self.b_out_dim for dim in self.proj_out_dim_list]
        self.query_num_list = [
            routed_num * head_num
            for routed_num, head_num in zip(self.routed_num_list, self.head_num_list)
        ]
        q_n_list = self.query_num_list
        self.query_num_list_adjusted = (
            [q_n_list[0]] + [q_n_list[1] // 2] * 2 + q_n_list[2:]
        )


class BlockPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        if config.unified_b_pool:
            self.weight_pool = nn.Parameter(
                torch.zeros(
                    config.unified_b_num,
                    config.b_in_dim,
                    config.b_out_dim,
                )
            )
        else:
            self.weight_pool = nn.ParameterDict()
            self.weight_pool["linear"] = nn.Parameter(
                torch.zeros(
                    config.total_linear_b_num,
                    config.b_in_dim,
                    config.b_out_dim,
                )
            )
            self.weight_pool["ffn"] = nn.Parameter(
                torch.zeros(
                    config.ffn_b_num,
                    3,
                    config.b_ffn_dim,
                    config.b_ffn_hidden_dim,
                )
            )

        if config.bias:
            if config.unified_b_pool:
                self.bias_pool = nn.Parameter(
                    torch.zeros(config.unified_b_num, 1, config.b_out_dim)
                )
            else:
                self.bias_pool = nn.ParameterDict()
                self.bias_pool["linear"] = nn.Parameter(
                    torch.zeros(config.total_linear_b_num, 1, config.b_out_dim)
                )
                self.bias_pool["ffn"] = nn.Parameter(
                    torch.zeros(config.ffn_b_num, 1, config.b_ffn_dim)
                )


class VecPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.device = f"cuda:{torch.cuda.current_device()}"
        self.config = config

        self.key_pool = nn.Parameter(
            torch.zeros(
                config.routed_layer_num,
                config.key_c_num_sum,
                config.b_query_dim,
                config.batch_key_num,
            )
        )

        self.key_cluster_pool = nn.Parameter(
            torch.zeros(
                config.routed_layer_num, config.b_query_router_dim, config.key_c_num_sum
            )
        )

        self.key_factor_pool = nn.Parameter(
            torch.zeros(
                config.routed_layer_num,
                config.key_c_num_sum,
                config.batch_key_num,
                config.key_factor_num,
            )
        )

        if config.key_c_router_bias:
            self.key_cluster_bias = nn.Parameter(
                torch.zeros(config.routed_layer_num, 1, config.key_c_num_sum)
            )

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        self.block_index_pool = self.block_index_gen()  # TODO

        self.key_mask = None
        self.key_cand = None
        self.key_split_list = None

    def block_index_gen(self):
        pass

    def vec_cat(self, init: bool):
        pass

    def vec_init(self):
        pass

    def vec_update(self):
        pass

    def training_update(self):
        pass


class RotaryEmb(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        device = f"cuda:{torch.cuda.current_device()}"
        inv_freq = (1 / config.theta) ** (
            torch.arange(0, config.b_attn_dim, 2) / config.b_attn_dim
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
        dim = self.config.b_attn_dim
        eps = self.config.norm_eps
        dtype = self.config.dtype
        xq = rearrange(xq, "s (h d) -> 1 h s d", d=dim)
        xk = rearrange(xk, "s (h d) -> 1 h s d", d=dim)
        xv = rearrange(xv, "s (h d) -> 1 h s d", d=dim)
        if self.config.qk_norm:
            xq = F.normalize(xq, dim=-1, eps=eps).to(dtype)
            xk = F.normalize(xk, dim=-1, eps=eps).to(dtype)
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)
        kernel_options = self.kernel_options if self.config.flex_attn_kernel else None
        x_attn = flex_attention(
            xq,
            xk,
            xv,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=kernel_options,
        )
        x_attn = rearrange(x_attn, "c h s d -> s c (h d)")
        return x_attn
    

class GroupedGEMM(Function):
    @staticmethod
    def forward(self):
        pass
    @staticmethod
    def backward(self):
        pass


class Query(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        in_query_num = sum(config.query_num_list[:3])
        out_query_num = sum(config.query_num_list[3:])
        query_dim = config.b_query_dim

        self.blockwise_w_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        routed_num,
                        config.b_routed_dim,
                        head_num * query_dim,
                    )
                )
                for routed_num, head_num in zip(
                    self.routed_num_list, self.head_num_list
                )
            ]
        )

        self.ffn_blockwise_w = nn.Parameter(
            torch.zeros(config.ffn_routed_num, config.b_ffn_dim, query_dim)
        )

        if self.config.in_query_fusion:
            self.in_fusion_w1 = nn.Linear(
                config.in_proj_dim, config.fusion_mid_dim, bias=False
            )
            self.in_fusion_w2 = nn.Linear(
                config.fusion_mid_dim, in_query_num * query_dim, bias=False
            )

        if self.config.out_query_fusion:
            self.out_fusion_w1 = nn.Linear(
                config.mid_out_dim, config.fusion_mid_dim, bias=False
            )
            self.out_fusion_w2 = nn.Linear(
                config.fusion_mid_dim, out_query_num * query_dim, bias=False
            )

        if self.config.query_bias:
            self.in_bias = nn.Parameter(torch.zeros(1, in_query_num * query_dim))
            self.out_bias = nn.Parameter(torch.zeros(1, out_query_num * query_dim))
            self.ffn_bias = nn.Parameter(
                torch.zeros(1, config.ffn_routed_num * query_dim)
            )

    def forward(self, x: torch.Tensor, pre_query: Optional[torch.Tensor], type: str):
        if type != "ffn":
            x_batch = rearrange(x, "s (c d) -> c s d", d=self.config.b_routed_dim)
            routed_num_list = (
                self.routed_num_list[:3] if type == "in" else self.routed_num_list[3:]
            )
            x_list = torch.split(x_batch, routed_num_list, 0)
            x_output_list = []
            for i, x in enumerate(x_list):
                i = i + 3 if type == "out" else i
                y = torch.bmm(x, self.blockwise_w_list[i]).transpose(0, 1).flatten(1)
                x_output_list.append(y)
            x_ouput = torch.cat(x_list, -1)
            if type == "in" and self.config.in_query_fusion:
                x_ouput += self.in_fusion_w2(self.in_fusion_w1(x))
            if type == "out" and self.config.out_query_fusion:
                x_ouput += self.out_fusion_w2(self.out_fusion_w1(x))
            if self.config.query_bias:
                bias = self.in_bias if type == "in" else self.out_bias
                x_ouput += bias
        else:
            x_batch = rearrange(x, "s (c d) -> c s d", d=self.config.b_ffn_dim)
            x_ouput = (
                torch.bmm(x_batch, self.ffn_blockwise_w).transpose(0, 1).flatten(1)
            )
            if self.config.query_bias:
                x_ouput += self.ffn_bias
        if self.config.chain_of_query and pre_query is not None:
            x_ouput += pre_query
        return x_ouput


class Router(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_id: int,
        vec_pool: VecPool,
        grouped_gemm: Function
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.vec_pool = vec_pool
        self.grouped_gemm = grouped_gemm

    def forward(self, query: torch.Tensor, type: str):
        query_num_list = self.config.query_num_list_adjusted
        if type == "in":
            query_num_list = query_num_list[:4]
        if type == "out":
            query_num_list = query_num_list[4:-1]
        if type == "ffn":
            query_num_list = query_num_list[-1]
        query_batch = rearrange(query, "s (n v) -> n s v", v=self.config.b_query_dim)
        query_list = torch.split(query_batch, query_num_list, 0)
        query_c_score_list = []
        query_c_index_list = []
        key_c_list = self.config.key_c_num_cumsum_list
        for i, x in enumerate(query_list):
            if type == "out":
                i = i + 4
            if type == "ffn":
                i = i + 6
            pos1 = key_c_list[i - 1] if i > 0 else 0
            pos2 = key_c_list[i]
            query = x[:, :, : self.config.b_query_router_dim]
            key_c = self.vec_pool.key_cluster_pool[self.layer_id, :, pos1:pos2]
            c_result = torch.matmul(query, key_c)
            if self.config.key_c_router_bias:
                key_c_bias = self.vec_pool.key_cluster_bias[self.layer_id, :, pos1:pos2]
                c_result += key_c_bias.unsqueeze(0)
            score, index = torch.max(c_result, -1)
            offset = pos1
            index += offset
            query_c_score_list.append(score)
            query_c_index_list.append(index)
        query_c_score = torch.cat(query_c_score_list, 0)
        query_c_index = torch.cat(query_c_index_list, 0)
        if type == "in":
            pos1 = 0
            pos2 = key_c_list[3]
        if type == "out":
            pos1 = key_c_list[3]
            pos2 = key_c_list[5]
        if type == "ffn":
            pos1 = key_c_list[5]
            pos2 = key_c_list[-1]
        weight = self.vec_pool.key_pool[self.layer_id, pos1:pos2, :, :]
        info = 0
        result = self.grouped_gemm(query_batch, query_c_index, weight, info)


class Layer(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_id: int,
        block_pool: BlockPool,
        vec_pool: VecPool,
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        attention: Attention,
        grouped_gemm: Function,
    ):
        self.config = config
        self.layer_id = layer_id
        self.vec_pool = vec_pool
        self.block_pool = block_pool
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.attention = attention
        self.grouped_gemm = grouped_gemm

        self.vanilla_condition = config.vanilla_first_layer and layer_id == 0
        main_dim = config.main_dim
        ffn_h_dim = config.ffn_hidden_dim
        if self.vanilla_condition:
            self.ffn_w1 = nn.Linear(main_dim, ffn_h_dim, bias=False)
            self.ffn_w2 = nn.Linear(main_dim, ffn_h_dim, bias=False)
            self.ffn_w3 = nn.Linear(ffn_h_dim, main_dim, bias=False)
            self.qkv_w = nn.Linear(main_dim, 3 * main_dim, bias=False)
            self.o_w = nn.Linear(main_dim, main_dim, bias=False)
            self.norm = nn.ModuleList(
                [
                    nn.RMSNorm(
                        [config.seq_len, config.main_dim],
                        eps=config.norm_eps,
                        dtype=config.dtype,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.query = Query(config)
            self.router = Router(config, layer_id, vec_pool, grouped_gemm)
            self.in_coeff = nn.Parameter(torch.ones(1, config.output_dim))
            if config.out_linear_mul:
                self.out_coeff = nn.Parameter(torch.ones(1, config.out_proj_dim))
            if config.out_proj_mul:
                self.out_coeff = nn.Parameter(torch.ones(1, config.output_dim))
            if config.residual_factor:
                self.residual_coeff = nn.Parameter(torch.ones(1, config.output_dim))
            if config.chain_of_query:
                self.query_coeff = nn.Parameter(torch.ones(4))
            if config.c_lora:
                if config.no_main_vec:
                    self.c_w1 = nn.Linear(config.out_proj_dim, config.c_mid_dim, bias=False)
                    self.c_w2 = nn.Linear(config.c_mid_dim, config.in_proj_dim, bias=False)
                else:
                    self.in_c_w1 = nn.Linear(main_dim, config.c_mid_dim, bias=False)
                    self.in_c_w2 = nn.Linear(
                        config.c_mid_dim, config.in_proj_dim, bias=False
                    )
                    self.out_c_w1 = nn.Linear(
                        config.out_proj_dim, config.c_mid_dim, bias=False
                    )
                    self.out_c_w2 = nn.Linear(config.c_mid_dim, main_dim, bias=False)
            if config.linear_lora:
                self.in_linear_w1 = nn.Linear(
                    config.in_proj_dim, config.linear_mid_dim, bias=False
                )
                self.in_linear_w2 = nn.Linear(
                    config.linear_mid_dim, config.mid_in_dim, bias=False
                )
                self.out_linear_w1 = nn.Linear(
                    config.mid_out_dim, config.linear_mid_dim, bias=False
                )
                self.out_linear_w2 = nn.Linear(
                    config.linear_mid_dim, config.out_proj_dim, bias=False
                )

    def forward(self, x_input: torch.Tensor, block_mask: BlockMask):
        if self.vanilla_condition:
            x = self.norm[0](x_input)
            x = x_input + self.ffn_w3(F.silu(self.ffn_w1(x)) * self.ffn_w2(x))
            xq, xk, xv = self.qkv_w(self.norm[1](x)).chunk(3, dim=-1)
            x += self.o_w(self.attention(xq, xk, xv, block_mask))
        else:
            norm = (
                torch.linalg.vector_norm(
                    x_input, dim=-1, keepdim=True, dtype=self.config.dtype
                )
                + self.config.norm_eps
            )
            x = x_input / norm * self.in_coeff.to(self.config.dtype)
            if not self.config.no_main_vec:
                if self.config.c_lora:
                    x = self.in_proj(x) + self.in_c_w2(self.in_c_w1(x))
                else:
                    x = self.in_proj(x)
            in_query = self.query(x, "in")
            in_router_result = self.router(in_query, "in")


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.block_pool = BlockPool(config)
        self.vec_pool = VecPool(config)
        if config.no_main_vec:
            self.proj = nn.Linear(config.out_proj_dim, config.in_proj_dim, bias=False)
        else:
            self.in_proj = nn.Linear(config.main_dim, config.in_proj_dim, bias=False)
            self.out_proj = nn.Linear(config.out_proj_dim, config.main_dim, bias=False)
        self.rotary_emb = RotaryEmb(config)
        self.attention = Attention(config, self.rotary_emb)
        self.grouped_gemm = GroupedGEMM.apply
        self.model = nn.ModuleDict(
            {
                "vocab_emb": nn.Embedding(config.vocab_size, config.vocab_in_dim),
                "vocab_in_proj": nn.Linear(
                    config.c_vocab_dim, config.init_dim, bias=False
                )
                if config.vocab_compressed
                else nn.Identity(),
                "layers": nn.ModuleList(
                    [
                        Layer(
                            config,
                            layer_id,
                            self.block_pool,
                            self.vec_pool,
                            self.in_proj,
                            self.out_proj,
                            self.attention,
                            self.grouped_gemm,
                        )
                        for layer_id in range(config.layer_num)
                    ]
                ),
                "vocab_out_proj": nn.Linear(
                    config.output_dim, config.c_vocab_dim, bias=False
                )
                if config.vocab_compressed
                else nn.Identity(),
                "lm_head": nn.Linear(
                    config.vocab_out_dim, config.vocab_size, bias=False
                ),
            }
        )
        if config.tied_vocab_emb:
            self.model.vocab_emb.weight = self.model.lm_head.weight
        self.generator = torch.Generator()
        self.generator.manual_seed(42)
        self.init_weights()

    def forward(
        self,
        idx: torch.LongTensor,
        targets: torch.Tensor,
    ):
        device = f"cuda:{torch.cuda.current_device()}"
        docs = (idx == 50256).cumsum(0)
        seq_len = self.config.seq_len

        def doc_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < self.config.window_len
            return causal_mask & doc_mask & window_mask

        block_mask = create_block_mask(
            doc_causal_mask, None, None, seq_len, seq_len, device=device, _compile=True
        )
        x = self.model.vocab_emb(idx)
        x = self.model.vocab_in_proj(x)

        c_aux_loss = 0
        b_aux_loss = 0
        for _, layer in enumerate(self.model.layers):
            x, layer_c_aux_loss, layer_b_aux_loss = layer(x, block_mask)
            c_aux_loss += layer_c_aux_loss
            b_aux_loss += layer_b_aux_loss

        x = F.normalize(x, dim=-1, eps=self.config.norm_eps)
        x = self.model.vocab_out_proj(x)
        logits = self.model.lm_head(x)
        loss = F.cross_entropy(logits, targets)

        c_aux_loss *= self.config.cluster_token_balance_factor
        b_aux_loss *= self.config.block_token_balance_factor
        total_loss = loss + c_aux_loss + b_aux_loss

        return loss.detach(), total_loss, c_aux_loss.detach(), b_aux_loss.detach()
