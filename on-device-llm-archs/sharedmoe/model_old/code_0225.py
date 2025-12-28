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
from typing import List, Tuple, Dict, Optional, Callable

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


class Config:
    layer_num: int = 8
    seq_len: int = 8 * 1024
    main_dim: int = 1024
    vec_dim: int = 32
    compressed_dim: int = 512
    ffn_block_dim: int = 128
    linear_block_in_dim: int = 64
    linear_block_out_dim: int = 256
    attn_block_dim = 128
    vocab_size: int = 50304
    vocab_c_dim: int = 768
    linear_fusion_c_dim: int = 48
    ffn_fusion_c_dim: int = 64
    external_c_dim: int = 64
    linear_fixed_in_dim: int = 48
    linear_fixed_out_dim: int = 32
    ffn_fixed_dim: int = 64
    ffn_vanilla_dim: int = 128
    ffn_hidden_dim: int = 128
    ffn_top_k: int = 3
    x_len: int = 32

    init_std: float = 0.02
    theta: float = 10000.0
    pool_lr: float = 3e-4
    model_lr: float = 6e-4
    min_lr_ratio: float = 0.01
    scheduler: str = "wsd"
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    norm_eps: float = 1e-6
    adam_eps: float = 1e-8
    window_len: int = 4096
    capacity_factor: float = 1.2
    c_token_balance_factor: float = 1e-3
    vec_token_balance_factor: float = 1e-3
    c_vec_balance_factor: float = 0.05
    c_ffn_balance_ratio: float = 2.0
    vec_ffn_balance_ratio: float = 1.0
    ffn_balance_ratio: float = 1.0
    dtype: torch.dtype = torch.bfloat16

    linear_block_num: Dict[str, int] = {
        "linear_1": 200,
        "linear_2": 600,
        "linear_3": 200,
        "linear_4": 150,
    }
    global_linear_block_num: int = 800
    ffn_block_num: int = 600
    linear_cluster_num: int = 64
    ffn_cluster_num: int = 128
    c_vec_num: int = 64
    valid_c_vec_num: int = 48

    fixed: bool = True
    global_linear_pool: bool = False
    paired: bool = True
    pre_gated: bool = True  # TODO
    bias: bool = False
    tied_vocab_emb: bool = True
    vocab_compressed: bool = True
    last_layer_vanilla: bool = False
    external: bool = True
    vanilla_router: bool = False  # TODO
    residual_norm: bool = True
    residual_factor: bool = True
    qk_norm: bool = True
    zero_optimizer: bool = False
    no_weight_decay_1d: bool = True
    model_compile: bool = False
    scaler: bool = True
    flex_attention_compile: bool = True
    linear_router_prob_norm: bool = False
    ffn_router_prob_norm: bool = False
    linear_fused: bool = True
    ffn_fused: bool = True

    linear_module_list: List[str] = [
        "qkv_c",
        "ffn_c",
        "q",
        "k",
        "v",
        "ffn_g",
        "ffn_l",
        "v_oc",
        "ffn_oc",
        "output",
    ]
    linear_module_parent_list: List[str] = (
        ["linear_1"] * 2 + ["linear_3"] * 2 + ["linear_2"] * 5 + ["linear_4"]
    )

    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
    }


def layer_info(layer_id: int, config: Config) -> Tuple[bool, bool]:
    first_layer = layer_id == 0
    last_layer = layer_id == (config.layer_num - 1)
    if config.last_layer_vanilla:
        last_standard_layer = layer_id == (config.layer_num - 2)
        last_vanilla_layer = last_layer
    else:
        last_standard_layer = last_layer
        last_vanilla_layer = False
    have_router = not last_standard_layer and not last_vanilla_layer
    vanilla_matmul = first_layer or last_vanilla_layer

    return have_router, vanilla_matmul


class ParamBlockPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.weight_pool = nn.ParameterDict()

        if config.global_linear_pool:
            self.weight_pool["linear"] = nn.Parameter(
                torch.zeros(
                    config.global_linear_block_num,
                    config.linear_block_in_dim,
                    config.linear_block_out_dim,
                )
            )
            # [n_glb, d_li, d_lo], n_glb = global_linear_block_num,
            # d_li = linear_block_in_dim, d_lo = linear_block_out_dim
        else:
            for sublayer, block_num in config.linear_block_num.items():
                self.weight_pool[sublayer] = nn.Parameter(
                    torch.zeros(
                        block_num,
                        config.linear_block_in_dim,
                        config.linear_block_out_dim,
                    )
                )
                # [n_slb, d_li, d_lo], n_slb = subayer_linear_block_num
        self.weight_pool["ffn"] = nn.Parameter(
            torch.zeros(
                config.ffn_block_num, 3, config.ffn_block_dim, config.ffn_hidden_dim
            )
        )
        # [n_fb, 3, d_f, d_fh], n_fb = ffn_block_num, d_f = ffn_block_dim, d_fh = ffn_hidden_dim

        if config.bias:
            self.bias_pool = nn.ParameterDict()
            if config.global_linear_pool:
                self.bias_pool["linear"] = nn.Parameter(
                    torch.zeros(
                        config.global_linear_block_num, 1, config.linear_block_out_dim
                    )
                )
                # [n_glb, 1, d_lo]
            else:
                for sublayer, block_num in config.linear_block_num.items():
                    self.bias_pool[sublayer] = nn.Parameter(
                        torch.zeros(block_num, 1, config.linear_block_out_dim)
                    )
                    # [n_slb, 1, d_lo]
            self.bias_pool["ffn"] = nn.Parameter(
                torch.zeros(config.ffn_block_num, 1, config.ffn_block_dim)
            )
            # [n_fb, 1, d_f], d = block_dim


class ParamVectorPool(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        layer_num = (
            config.layer_num - 2 if config.last_layer_vanilla else config.layer_num - 1
        )
        index_num_per_vec = 2 if config.paired else 1
        linear_module_num = len(config.linear_module_list)

        self.device = f"cuda:{torch.cuda.current_device()}"
        self.config = config
        self.layer_num = layer_num
        self.n_iv = index_num_per_vec
        self.n_lm = linear_module_num
        self.linear_c_num = layer_num * linear_module_num * config.linear_cluster_num
        self.ffn_c_num = layer_num * config.ffn_cluster_num

        self.linear_vec = nn.ModuleDict()
        self.ffn_vec = nn.ModuleDict()
        self.linear_vec_info = {}
        self.ffn_vec_info = {}

        self.vec_pool = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        linear_module_num * config.linear_cluster_num
                        + config.ffn_cluster_num,
                        config.vec_dim,
                        config.c_vec_num,
                    )
                )
                for _ in range(layer_num)
            ]
        )
        # [n_lm * n_lc + n_fc, v, n_cv] * n_layer, n_layer = layer_num
        # n_lm = linear_module_num, n_lc = linear_cluster_num, n_fc = ffn_cluster_num
        # v = vec_dim, n_cv = c_vec_num

        self.linear_vec["vec_clusters"] = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        linear_module_num,
                        config.vec_dim,
                        config.linear_cluster_num,
                    )
                )
                for _ in range(layer_num)
            ]
        )
        # [n_lm, v, n_lc] * n_layer,

        self.linear_vec["gain_factors"] = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        linear_module_num,
                        config.linear_cluster_num,
                        config.c_vec_num,
                        index_num_per_vec,
                    )
                )
                for _ in range(layer_num)
            ]
        )
        # [n_lm, n_lc, n_cv, n_iv] * n_layer,
        # n_iv = index_num_per_vec

        self.ffn_vec["vec_clusters"] = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(config.vec_dim, config.ffn_cluster_num))
                for _ in range(layer_num)
            ]
        )
        # [v, n_fc] * n_layer

        self.ffn_vec["gain_factors"] = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        config.ffn_cluster_num,
                        config.c_vec_num,
                        index_num_per_vec,
                    )
                )
                for _ in range(layer_num)
            ]
        )
        # [n_fc, n_cv, n_iv] * n_layer

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        self.linear_vec_info["block_indices"] = [
            torch.cat(
                [
                    torch.randint(
                        1,
                        config.linear_block_num[config.linear_module_parent_list[i]],
                        (
                            1,
                            config.linear_cluster_num,
                            config.c_vec_num,
                            index_num_per_vec,
                        ),
                        dtype=torch.int32,
                        generator=generator,
                        device=self.device,
                    )
                    for i in range(linear_module_num)
                ],
                0,
            )
            for _ in range(layer_num)
        ]
        # cat([1, n_lc, n_cv, n_iv] * n_lm, dim = 0) * n_layer = [n_lm, n_lc, n_cv, n_iv] * n_layer

        self.ffn_vec_info["block_indices"] = [
            torch.randint(
                1,
                config.ffn_block_num,
                (
                    config.ffn_cluster_num,
                    config.c_vec_num,
                    index_num_per_vec,
                ),
                dtype=torch.int32,
                generator=generator,
                device=self.device,
            )
            for _ in range(layer_num)
        ]
        # [n_fc, n_cv, n_iv] * n_layer

        self.vec_mask = None
        self.candidates = None
        self.vec_split_list = None

    def vec_cat(self, init: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        num_linear = self.n_lm * self.config.linear_cluster_num
        linear_vec = torch.cat(
            ([self.vec_pool[i][:num_linear] for i in range(self.layer_num)]), 0
        )
        # [n_layer * n_lm * n_lc, v, n_cv]
        linear_vec = rearrange(
            linear_vec,
            "(n n_lc) v n_cv -> n n_lc n_cv v",
            n_lc=self.config.linear_cluster_num,
        )
        # [n_layer * n_lm, n_lc, n_cv, v]
        linear_gain_factors = torch.cat(
            ([self.linear_vec["gain_factors"][i] for i in range(self.layer_num)]), 0
        )
        # [n_layer * n_lm, n_lc, n_cv, n_iv]
        linear_block_indices = torch.cat(
            ([self.linear_vec_info["block_indices"][i] for i in range(self.layer_num)]),
            0,
        )
        # [n_layer * n_lm, n_lc, n_cv, n_iv]
        linear_vec_cat = torch.cat(
            (
                linear_vec,
                linear_gain_factors,
                linear_block_indices.to(linear_vec.dtype),
            ),
            -1,
        )
        # [n_layer * n_lm, n_lc, n_cv, v + 2 * n_iv]
        if init:
            linear_vec_cat = linear_vec_cat[:, :, : self.config.valid_c_vec_num, :]
            # [n_layer * n_lm, n_lc, n_v, v + 2 * n_iv], n_v = valid_c_vec_num

        linear_vec_cat = linear_vec_cat.flatten(1, 2)
        # [n_layer * n_lm, n_lc * n_cv (n_v), v + 2 * n_iv]

        ffn_vec = torch.cat(
            ([self.vec_pool[i][num_linear:] for i in range(self.layer_num)]), 0
        ).transpose(1, 2)
        # [n_layer * n_fc, n_cv, v]
        ffn_gain_factors = torch.cat(
            ([self.ffn_vec["gain_factors"][i] for i in range(self.layer_num)]), 0
        )
        # [n_layer * n_fc, n_cv, n_iv]
        ffn_block_indices = torch.cat(
            ([self.ffn_vec_info["block_indices"][i] for i in range(self.layer_num)]), 0
        )
        # [n_layer * n_fc, n_cv, n_iv]
        ffn_vec_cat = torch.cat(
            (
                ffn_vec,
                ffn_gain_factors,
                ffn_block_indices.to(ffn_vec.dtype),
            ),
            -1,
        )
        # [n_layer * n_fc, n_cv, v + 2 * n_iv]
        ffn_vec_cat = rearrange(
            ffn_vec_cat,
            "(n_layer n_fc) n_cv v -> n_layer n_fc n_cv v",
            n_fc=self.config.ffn_cluster_num,
        )
        # [n_layer, n_fc, n_cv, v + 2 * n_iv]
        if init:
            ffn_vec_cat = ffn_vec_cat[:, :, : self.config.valid_c_vec_num, :]
            # [n_layer, n_fc, n_v, v + 2 * n_iv]

        ffn_vec_cat = ffn_vec_cat.flatten(1, 2)
        # [n_layer, n_fc * n_cv (n_v), v + 2 * n_iv]

        return linear_vec_cat, ffn_vec_cat

    def vec_init(self):
        init = True
        linear_vec_cat, ffn_vec_cat = self.vec_cat(init)
        # linear_vec_cat: [n_layer * n_lm, n_lc * n_v, v + 2 * n_iv]
        # ffn_vec_cat: [n_layer, n_fc * n_v, v + 2 * n_iv]
        linear_vec = linear_vec_cat[:, :, : self.config.vec_dim]
        # [n_layer * n_lm, n_lc * n_v, v]
        ffn_vec = ffn_vec_cat[:, :, : self.config.vec_dim]
        # [n_layer, n_fc * n_v, v]
        n_lc = self.config.linear_cluster_num
        n_fc = self.config.ffn_cluster_num

        device = linear_vec_cat.device
        linear_vec_labels = torch.zeros(
            linear_vec_cat.shape[0], linear_vec_cat.shape[1], device=device
        )
        # [n_layer * n_lm, n_lc * n_v]
        ffn_vec_labels = torch.zeros(
            ffn_vec_cat.shape[0], ffn_vec_cat.shape[1], device=device
        )
        # [n_layer, n_fc * n_v]
        linear_vec_centroids = torch.zeros(
            linear_vec_cat.shape[0],
            n_lc,
            self.config.vec_dim,
            device=device,
        )
        # [n_layer * n_lm, n_lc, v]
        ffn_vec_centroids = torch.zeros(
            ffn_vec_cat.shape[0],
            n_fc,
            self.config.vec_dim,
            device=device,
        )
        # [n_layer, n_fc, v]

        linear_k_means = KMeans(n_clusters=n_lc, verbose=1)
        ffn_k_means = KMeans(n_clusters=n_fc, verbose=1)

        for i in range(linear_vec_cat.shape[0]):
            linear_vec_labels[i] = linear_k_means.fit_predict(linear_vec[i])

        for i in range(n_lc):
            label_mask = linear_vec_labels == i
            # [n_layer * n_lm, n_lc * n_v]
            linear_masked_vec = torch.sum(linear_vec * label_mask.unsqueeze(-1), 1)
            # [n_layer * n_lm, v]
            label_mask_num = torch.sum(label_mask, 1, keepdim=True)
            # [n_layer * n_lm, 1]
            linear_vec_centroids[:, i, :] = linear_masked_vec / label_mask_num

        for i in range(ffn_vec_cat.shape[0]):
            ffn_vec_labels[i] = ffn_k_means.fit_predict(ffn_vec[i])

        for i in range(n_lc):
            label_mask = ffn_vec_labels == i
            # [n_layer, n_fc * n_v]
            ffn_masked_vec = torch.sum(ffn_vec * label_mask.unsqueeze(-1), 1)
            # [n_layer, v]
            label_mask_num = torch.sum(label_mask, 1, keepdim=True)
            # [n_layer, 1]
            ffn_vec_centroids[:, i, :] = ffn_masked_vec / label_mask_num

        linear_vec_centroids = rearrange(
            linear_vec_centroids,
            "(n_layer n_lm) n_lc v -> n_layer n_lm v n_lc",
            n_layer=self.layer_num,
        )
        # [n_layer, n_lm, v, n_lc]

        with torch.no_grad():
            for i in range(self.layer_num):
                self.linear_vec["vec_clusters"][i].data = linear_vec_centroids[i]
                self.ffn_vec["vec_clusters"][i].data = ffn_vec_centroids[i].transpose(
                    0, 1
                )

        self.update(
            linear_vec_cat, ffn_vec_cat, linear_vec_labels, ffn_vec_labels, init
        )

    def training_update(self):
        init = False
        linear_vec_cat, ffn_vec_cat = self.vec_cat(init)
        linear_vec_labels, ffn_vec_labels = None, None
        return self.update(
            linear_vec_cat, ffn_vec_cat, linear_vec_labels, ffn_vec_labels, init
        )

    def update(
        self,
        linear_vec,  # [n_layer * n_lm, n_lc * n_cv (n_v), v + 2 * n_iv]
        ffn_vec,  # [n_layer, n_fc * n_cv (n_v), v + 2 * n_iv]
        linear_vec_labels,  # [n_layer * n_lm, n_lc * n_v] or None
        ffn_vec_labels,  # [n_layer, n_fc * n_v] or None
        init,
    ):
        device = linear_vec.device
        v_cat_dim = linear_vec.shape[-1]

        linear_vec_num_valid = self.linear_c_num * self.config.valid_c_vec_num
        ffn_vec_num_valid = self.ffn_c_num * self.config.valid_c_vec_num
        vec_num_valid = linear_vec_num_valid + ffn_vec_num_valid

        linear_vec_num = self.linear_c_num * self.config.c_vec_num
        ffn_vec_num = self.ffn_c_num * self.config.c_vec_num
        vec_num = linear_vec_num + ffn_vec_num

        vec_c_num = self.linear_c_num + self.ffn_c_num

        c_vec_aux_loss = None
        c_mean_prob = None

        if not init:
            vec_main = torch.cat(
                (
                    linear_vec.flatten(0, 1),
                    ffn_vec.flatten(0, 1),
                ),
                0,
            )[self.vec_mask]
            vec_main_list = torch.split(vec_main, self.vec_split_list, 0)
            # module_num

            vec_list = [None] * len(vec_main_list) * 2
            vec_list[::2] = vec_main_list
            vec_list[1::2] = self.candidates
            vec = torch.cat(vec_list, 0)
            # [vec_num_valid, v + 2 * n_iv], vec_num_valid = vec_c_num * n_v
            # = n_layer * n_lm * n_lc * n_v + n_layer * n_fc * n_v
            assert vec.shape[0] == vec_num_valid

            linear_vec = rearrange(
                vec[:linear_vec_num_valid],
                "(n1 n2) v -> n1 n2 v",
                n1=self.layer_num * self.n_lm,
            )
            # [n_layer * n_lm, n_lc * n_v, v + 2 * n_iv]
            ffn_vec = rearrange(
                vec[linear_vec_num_valid:], "(n1 n2) v -> n1 n2 v", n1=self.layer_num
            )
            # [n_layer, n_fc * n_v, v + 2 * n_iv]

            linear_vec_c = torch.cat(
                ([self.linear_vec["vec_clusters"][i] for i in range(self.layer_num)]), 0
            )
            # [n_layer * n_lm, v, n_lc]
            linear_vec_scores = torch.bmm(
                linear_vec[:, :, : self.config.vec_dim], linear_vec_c
            )
            # [n_layer * n_lm, n_lc * n_v, v] bmm [n_layer * n_lm, v, n_lc]
            # = [n_layer * n_lm, n_lc * n_v, n_lc]
            linear_vec_values, linear_vec_labels = torch.max(linear_vec_scores, -1)
            # [n_layer * n_lm, n_lc * n_v] * 2

            ffn_vec_c = torch.cat(
                ([self.ffn_vec["vec_clusters"][i] for i in range(self.layer_num)]), 0
            )
            # [n_layer * v, n_fc]
            ffn_vec_c = rearrange(
                ffn_vec_c,
                "(n_layer v) n_fc -> n_layer v n_fc",
                n_layer=self.layer_num,
            )
            # [n_layer, v, n_fc]
            ffn_vec_scores = torch.bmm(ffn_vec[:, :, : self.config.vec_dim], ffn_vec_c)
            # [n_layer, n_fc * n_v, v] bmm [n_layer, v, n_fc] = [n_layer, n_fc * n_v, n_fc]
            ffn_vec_values, ffn_vec_labels = torch.max(ffn_vec_scores, -1)
            # [n_layer, n_fc * n_v] * 2
        else:
            vec = torch.cat((linear_vec.flatten(0, 1), ffn_vec.flatten(0, 1)), 0)
            # [vec_num_valid, v + 2 * n_iv]

        n_lc = self.config.linear_cluster_num
        linear_label_offset = (
            torch.arange(linear_vec_labels.shape[0], device=device) * n_lc
        ).unsqueeze(-1)
        # [n_layer * n_lm, 1]
        linear_vec_new_labels = linear_vec_labels + linear_label_offset
        # [n_layer * n_lm, n_lc * n_v]

        n_fc = self.config.ffn_cluster_num
        ffn_label_offset = (
            torch.arange(ffn_vec_labels.shape[0], device=device) * n_fc
        ).unsqueeze(-1)
        # [n_layer, 1]
        ffn_vec_new_labels = ffn_vec_labels + ffn_label_offset + self.linear_c_num
        # [n_layer, n_fc * n_v]

        vec_labels = torch.cat(
            (linear_vec_new_labels.flatten(), ffn_vec_new_labels.flatten()), 0
        ).to(torch.int64)
        # [vec_num_valid]

        with torch.no_grad():
            vec_indices_bin = torch.bincount(vec_labels, minlength=vec_c_num)
            # [vec_c_num]

        if not init:
            vec_probs = torch.cat(
                (
                    torch.sigmoid(linear_vec_values).flatten(),
                    torch.sigmoid(ffn_vec_values).flatten()
                    * self.config.ffn_balance_ratio,
                ),
                0,
            )
            vec_probs_bin = torch.zeros(vec_c_num, device=device, dtype=vec_probs.dtype)
            vec_probs_bin.scatter_add_(0, vec_labels, vec_probs)

            c_vec_aux_loss = (
                torch.dot(vec_indices_bin.type_as(vec_probs_bin), vec_probs_bin)
                / (vec_labels.numel() * torch.sum(vec_probs_bin))
                * vec_c_num
                * self.config.c_vec_balance_factor
            )
            c_mean_prob = torch.mean(vec_probs_bin / vec_indices_bin).item() # TODO: non_zero

        vec_indices_bin_main = vec_indices_bin.clamp_max(self.config.c_vec_num)
        # [vec_c_num]
        vec_indices_bin_cand = vec_indices_bin - vec_indices_bin_main
        # [vec_c_num]
        vec_indices_bin_pad = self.config.c_vec_num - vec_indices_bin_main
        # [vec_c_num]
        vec_cand_num = torch.sum(vec_indices_bin_cand).item()

        linear_vec_bin_main = (
            vec_indices_bin_main[: self.linear_c_num]
            .reshape(-1, self.config.linear_cluster_num)
            .sum(-1)
        )
        # [n_layer * n_lm]
        ffn_vec_bin_main = (
            vec_indices_bin_main[self.linear_c_num :]
            .reshape(-1, self.config.ffn_cluster_num)
            .sum(-1)
        )
        # [n_layer]
        module_bin_main_list = torch.cat(
            (linear_vec_bin_main, ffn_vec_bin_main), 0
        ).tolist()
        # module_num, module_num = n_layer * n_lm + n_layer

        linear_vec_bin_cand = (
            vec_indices_bin_cand[: self.linear_c_num]
            .reshape(-1, self.config.linear_cluster_num)
            .sum(-1)
        )
        # [n_layer * n_lm]
        ffn_vec_bin_cand = (
            vec_indices_bin_cand[self.linear_c_num :]
            .reshape(-1, self.config.ffn_cluster_num)
            .sum(-1)
        )
        # [n_layer]
        module_bin_cand_list = torch.cat(
            (linear_vec_bin_cand, ffn_vec_bin_cand), 0
        ).tolist()
        # module_num

        vec_bool = torch.cat(
            (
                torch.full((vec_c_num, 1), True, device=device),
                torch.full((vec_c_num, 1), False, device=device),
            ),
            -1,
        ).flatten()
        # [vec_c_num * 2]
        vec_repeat = torch.cat(
            (
                torch.full((vec_c_num, 1), self.config.c_vec_num, device=device),
                vec_indices_bin_cand.unsqueeze(-1),
            ),
            -1,
        ).flatten()
        # [vec_c_num * 2]
        vec_mask = torch.repeat_interleave(vec_bool, vec_repeat, 0)
        # [vec_num + vec_cand_num], vec_num = vec_c_num * n_cv

        vec_labels_pad = torch.repeat_interleave(
            torch.arange(vec_c_num, device=device), vec_indices_bin_pad, 0
        )
        # [vec_pad_num]
        vec_labels_w_pad = torch.cat((vec_labels, vec_labels_pad), 0)
        # [vec_num_valid + vec_pad_num], vec_num_valid + vec_pad_num = vec_num + vec_cand_num
        vec_order = torch.argsort(vec_labels_w_pad, stable=True)
        # [vec_num_valid + vec_pad_num]
        vec_rank = torch.empty_like(vec_order).scatter_(
            0, vec_order, torch.arange(vec_order.shape[0], device=device)
        )
        # [vec_num_valid + vec_pad_num]
        vec_rank = vec_rank[:vec_num_valid]
        # [vec_num_valid]

        vec_zone = torch.zeros(
            vec_labels_w_pad.shape[0], v_cat_dim, device=device, dtype=vec.dtype
        )
        # [vec_num_valid + vec_pad_num, v + 2 * n_iv]
        vec_zone[vec_rank] = vec
        new_vec = vec_zone[vec_mask]
        # [vec_num, v + 2 * n_iv]
        vec_cand = vec_zone[~vec_mask]
        # [vec_cand_num, v + 2 * n_iv]
        new_vec_mask = new_vec[:, -1] != 0
        # [vec_num]
        vec_cand_list = torch.split(vec_cand, module_bin_cand_list, 0)
        # module_num

        linear_new_vec = rearrange(
            new_vec[:linear_vec_num],
            "(n_layer n_lm n_lc n_cv) v -> n_layer n_lm n_lc n_cv v",
            n_layer=self.layer_num,
            n_lm=self.n_lm,
            n_lc=self.config.linear_cluster_num,
            n_cv=self.config.c_vec_num,
        )
        # [n_layer, n_lm, n_lc, n_cv, v + 2 * n_iv]

        ffn_new_vec = rearrange(
            new_vec[linear_vec_num:],
            "(n_layer n_fc n_cv) v -> n_layer n_fc n_cv v",
            n_layer=self.layer_num,
            n_fc=self.config.ffn_cluster_num,
            n_cv=self.config.c_vec_num,
        )
        # [n_layer, n_fc, n_cv, v + 2 * n_iv]

        v = self.config.vec_dim
        n_iv = self.n_iv
        linear_new_vec_reshape = linear_new_vec.flatten(1, 2).transpose(2, 3)
        # [n_layer, n_lm, n_lc, n_cv, v + 2 * n_iv] ->
        # [n_layer, n_lm * n_lc, v + 2 * n_iv, n_cv]
        layer_linear_c_num = linear_new_vec_reshape.shape[1]
        ffn_new_vec_reshape = ffn_new_vec.transpose(2, 3)
        # [n_layer, n_fc, v + 2 * n_iv, n_cv]

        for i in range(self.layer_num):
            with torch.no_grad():
                self.vec_pool[i].data[:layer_linear_c_num] = linear_new_vec_reshape[
                    i, :, :v, :
                ]
                self.vec_pool[i].data[layer_linear_c_num:] = ffn_new_vec_reshape[
                    i, :, :v, :
                ]
                self.linear_vec["gain_factors"][i].data = linear_new_vec[
                    i, :, :, :, v : v + n_iv
                ]
                self.ffn_vec["gain_factors"][i].data = ffn_new_vec[
                    i, :, :, v : v + n_iv
                ]
            self.linear_vec_info["block_indices"][i] = linear_new_vec[
                i, :, :, :, v + n_iv :
            ].to(torch.int32)
            self.ffn_vec_info["block_indices"][i] = ffn_new_vec[i, :, :, v + n_iv :].to(
                torch.int32
            )

        self.vec_mask = new_vec_mask
        self.candidates = vec_cand_list
        self.vec_split_list = module_bin_main_list
        cand_p = vec_cand_num / vec_num_valid

        return c_vec_aux_loss, c_mean_prob, cand_p


class Query(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()

        self.config = config
        self.dtype = self.config.dtype
        self.v = config.vec_dim
        linear_type1_module_num = 4
        # n_1m
        linear_type2_module_num = 6
        # n_2m
        self.n_1m = linear_type1_module_num
        self.n_2m = linear_type2_module_num

        linear_type1_multihead_num = int(
            config.compressed_dim / config.linear_block_out_dim
        )
        # n_1mh
        linear_type2_multihead_num = int(config.main_dim / config.linear_block_out_dim)
        # n_2mh

        linear_router_block_dim = (
            2 * config.linear_block_in_dim
            if config.paired
            else config.linear_block_in_dim
        )
        # d_r
        self.d_r = linear_router_block_dim

        main_router_block_num = int(config.main_dim / linear_router_block_dim)
        # n_mr

        compressed_router_block_num = int(
            config.compressed_dim / linear_router_block_dim
        )
        # n_cr

        layer_linear_type1_router_block_num = int(
            linear_type1_module_num * main_router_block_num
        )
        # n_1r
        layer_linear_type2_router_block_num = int(
            linear_type2_module_num * compressed_router_block_num
        )
        # n_2r

        linear_type1_fusion_dim = int(config.vec_dim * main_router_block_num)
        # d_1f
        linear_type2_fusion_dim = int(config.vec_dim * compressed_router_block_num)
        # d_2f

        layer_ffn_router_block_num = int(config.main_dim / config.ffn_block_dim)
        # n_fr

        self.linear_type1_blockwise_w = nn.Parameter(
            torch.zeros(
                layer_linear_type1_router_block_num,
                linear_router_block_dim,
                linear_type1_multihead_num * config.vec_dim,
            )
        )
        # [n_1r, d_r, n_1mh * v], n_1r = layer_linear_type1_router_block_num,
        # d_r = linear_router_block_dim, n_1mh = linear_type1_multihead_num

        self.linear_type2_blockwise_w = nn.Parameter(
            torch.zeros(
                layer_linear_type2_router_block_num,
                linear_router_block_dim,
                linear_type2_multihead_num * config.vec_dim,
            )
        )
        # [n_2r, d_r, n_2mh * v]

        self.linear_type1_fusion_w_in = nn.Parameter(
            torch.zeros(
                linear_type1_module_num * linear_type1_multihead_num,
                linear_type1_fusion_dim,
                config.linear_fusion_c_dim,
            )
        )
        # [n_1m * n_1mh, d_1f, d_lc], n_1m = linear_type1_module_num,
        # d_1f = linear_type1_fusion_dim, d_lfc = linear_fusion_c_dim

        self.linear_type2_fusion_w_in = nn.Parameter(
            torch.zeros(
                linear_type2_module_num * linear_type2_multihead_num,
                linear_type2_fusion_dim,
                config.linear_fusion_c_dim,
            )
        )
        # [n_2m * n_2mh, d_2f, d_lc]

        self.linear_type1_fusion_w_out = nn.Parameter(
            torch.zeros(
                linear_type1_module_num * linear_type1_multihead_num,
                config.linear_fusion_c_dim,
                linear_type1_fusion_dim,
            )
        )
        # [n_1m * n_1mh, d_lc, d_1f]

        self.linear_type2_fusion_w_out = nn.Parameter(
            torch.zeros(
                linear_type2_module_num * linear_type2_multihead_num,
                config.linear_fusion_c_dim,
                linear_type2_fusion_dim,
            )
        )
        # [n_2m * n_2mh, d_lc, d_2f]

        self.linear_type1_b = nn.Parameter(
            torch.zeros(
                1,
                linear_type1_module_num,
                main_router_block_num,
                linear_type1_multihead_num,
                config.vec_dim,
            )
        )
        # [1, n_1m, n_mr, n_1mh, v]

        self.linear_type2_b = nn.Parameter(
            torch.zeros(
                1,
                linear_type2_module_num,
                compressed_router_block_num,
                linear_type2_multihead_num,
                config.vec_dim,
            )
        )
        # [1, n_2m, n_cr, n_2mh, v]

        self.ffn_blockwise_w = nn.Parameter(
            torch.zeros(
                layer_ffn_router_block_num,
                2 * config.ffn_block_dim,
                config.vec_dim,
            )
        )
        # [n_fr, 2 * d_f, v], n_fr = layer_ffn_router_block_num

        self.ffn_fusion_w_in = nn.Parameter(
            torch.zeros(
                layer_ffn_router_block_num * config.vec_dim,
                config.ffn_fusion_c_dim,
            )
        )
        # [n_fr * v, d_fc], d_fc = ffn_fusion_c_dim

        self.ffn_fusion_w_out = nn.Parameter(
            torch.zeros(
                config.ffn_fusion_c_dim,
                layer_ffn_router_block_num * config.vec_dim,
            )
        )
        # [d_fc, n_fr * v]

        self.ffn_b = nn.Parameter(
            torch.zeros(1, layer_ffn_router_block_num, config.vec_dim)
        )
        # [1, n_fr, v]

    def forward(
        self,
        x_linear_type1: torch.Tensor,  # [s, n_1m, d_m], d_m = main_dim
        x_linear_type2: torch.Tensor,  # [s, n_2m, d_c], d_c = compressed_dim
        x_ffn: torch.Tensor,  # [s, 2, d_m]
        external: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_linear_type1 = rearrange(
            x_linear_type1,
            "s n_1m (n_mr d_r) -> (n_1m n_mr) s d_r",
            d_r=self.d_r,
        )
        # [n_1r, s, d_r]

        x_linear_type1_blockwise = torch.bmm(
            x_linear_type1, self.linear_type1_blockwise_w
        )
        # [n_1r, s, d_r] bmm [n_1r, d_r, n_1mh * v] = [n_1r, s, n_1mh * v]

        x_linear_type2 = rearrange(
            x_linear_type2,
            "s n_2m (n_cr d_r) -> (n_2m n_cr) s d_r",
            d_r=self.d_r,
        )
        # [n_2r, s, d_r]

        x_linear_type2_blockwise = torch.bmm(
            x_linear_type2, self.linear_type2_blockwise_w
        )
        # [n_2r, s, d_r] bmm [n_2r, d_r, n_2mh * v] = [n_2r, s, n_2mh * v]

        x_linear_type1_fusion = rearrange(
            x_linear_type1_blockwise,
            "(n_1m n_mr) s (n_1mh v) -> (n_1m n_1mh) s (n_mr v)",
            n_1m=self.n_1m,
            v=self.v,
        )
        # [n_1m * n_1mh, s, d_1f]

        x_linear_type1_fusion = torch.bmm(
            x_linear_type1_fusion, self.linear_type1_fusion_w_in
        )
        # [n_1m * n_1mh, s, d_1f] bmm [n_1m * n_1mh, d_1f, d_lc] = [n_1m * n_1mh, s, d_lc]

        x_linear_type1_fusion = torch.bmm(
            x_linear_type1_fusion, self.linear_type1_fusion_w_out
        )
        # [n_1m * n_1mh, s, d_lc] bmm [n_1m * n_1mh, d_lc, d_1f] = [n_1m * n_1mh, s, d_1f]

        x_linear_type2_fusion = rearrange(
            x_linear_type2_blockwise,
            "(n_2m n_cr) s (n_2mh v) -> (n_2m n_2mh) s (n_cr v)",
            n_2m=self.n_2m,
            v=self.v,
        )
        # [n_2m * n_2mh, s, d_2f]

        x_linear_type2_fusion = torch.bmm(
            x_linear_type2_fusion, self.linear_type2_fusion_w_in
        )
        # [n_2m * n_2mh, s, d_2f] bmm [n_2m * n_2mh, d_2f, d_lc] = [n_2m * n_2mh, s, d_lc]

        x_linear_type2_fusion = torch.bmm(
            x_linear_type2_fusion, self.linear_type2_fusion_w_out
        )
        # [n_2m * n_2mh, s, d_lc] bmm [n_2m * n_2mh, d_lc, d_2f] = [n_2m * n_2mh, s, d_2f]

        x_linear_type1_blockwise = rearrange(
            x_linear_type1_blockwise,
            "(n_1m n_mr) s (n_1mh v) -> s n_1m n_mr n_1mh v",
            n_1m=self.n_1m,
            v=self.v,
        )
        # [s, n_1m, n_mr, n_1mh, v]

        x_linear_type1_fusion = rearrange(
            x_linear_type1_fusion,
            "(n_1m n_1mh) s (n_mr v) -> s n_1m n_mr n_1mh v",
            n_1m=self.n_1m,
            v=self.v,
        )
        # [s, n_1m, n_mr, n_1mh, v]

        x_linear_type2_blockwise = rearrange(
            x_linear_type2_blockwise,
            "(n_2m n_cr) s (n_2mh v) -> s n_2m n_cr n_2mh v",
            n_2m=self.n_2m,
            v=self.v,
        )
        # [s, n_2m, n_cr, n_2mh, v]

        x_linear_type2_fusion = rearrange(
            x_linear_type2_fusion,
            "(n_2m n_2mh) s (n_cr v) -> s n_2m n_cr n_2mh v",
            n_2m=self.n_2m,
            v=self.v,
        )
        # [s, n_2m, n_cr, n_2mh, v]

        query_linear_type1 = (
            x_linear_type1_blockwise
            + x_linear_type1_fusion
            + self.linear_type1_b.to(self.dtype)
        )
        # [s, n_1m, n_mr, n_1mh, v]
        if external is not None:
            query_linear_type1 += external["linear_type1"]

        query_linear_type2 = (
            x_linear_type2_blockwise
            + x_linear_type2_fusion
            + self.linear_type2_b.to(self.dtype)
        )
        # [s, n_2m, n_cr, n_2mh, v]
        if external is not None:
            query_linear_type2 += external["linear_type2"]

        x_ffn = rearrange(
            x_ffn, "s c (n_fr d_f) -> n_fr s (c d_f)", d_f=self.config.ffn_block_dim
        )
        # [n_fr, s, 2 * d_f]

        x_ffn_blockwise = torch.bmm(x_ffn, self.ffn_blockwise_w).transpose(0, 1)
        # [n_fr, s, 2 * d_f] bmm [n_fr, 2 * d_f, v] = [n_fr, s, v] -> [s, n_fr, v]
        x_ffn_fusion = torch.matmul(x_ffn_blockwise.flatten(1), self.ffn_fusion_w_in)
        # [s, n_fr * v] mm [n_fr * v, d_fc] = [s, d_fc]
        x_ffn_fusion = torch.matmul(x_ffn_fusion, self.ffn_fusion_w_out).reshape(
            self.config.seq_len, -1, self.config.vec_dim
        )
        # [s, d_fc] mm [d_fc, n_fr * v] = [s, n_fr * v] -> [s, n_fr, v]
        query_ffn = x_ffn_blockwise + x_ffn_fusion + self.ffn_b.to(self.dtype)
        # [s, n_fr, v]
        if external is not None:
            query_ffn += external["ffn"]

        query_linear_type1 = rearrange(
            query_linear_type1, "s n_1m n_mr n_1mh v -> n_1m (s n_mr n_1mh) v"
        )
        # [n_1m, s * n_lr, v], n_lr = n_mr * n_1mh, linear_module_router_block_num
        query_linear_type2 = rearrange(
            query_linear_type2, "s n_2m n_cr n_2mh v -> n_2m (s n_cr n_2mh) v"
        )
        # [n_2m, s * n_lr, v], n_lr = n_mr * n_1mh = n_cr * n_2mh
        query_linear = torch.cat((query_linear_type1, query_linear_type2), 0)
        # [n_lm, s * n_lr, v], n_lm = n_1m + n_2m

        return query_linear, query_ffn


def indices_pre_func(freqs, block_valid_num, x_len):
    device = freqs.device
    freq_sorted, freq_sorted_indices = torch.sort(freqs, descending=True, stable=False)
    # [m] * 2
    x_seq = torch.arange(x_len, device=device)
    # [x_len], s_len = x_len
    y_seq = torch.arange(block_valid_num, device=device)
    # [m_v], m_v = block_valid_num
    grid_x, grid_y = torch.meshgrid(x_seq, y_seq, indexing="ij")
    # [x_len, m_v] * 2
    mask = (grid_x > 0) & (grid_y > grid_x) & (grid_y < block_valid_num)

    freq_0 = freq_sorted[0]
    freq_1 = freq_sorted[grid_x]
    freq_2 = freq_sorted[grid_y]

    area_0 = freq_0 * grid_x
    area_1 = freq_1 * (grid_y - grid_x)
    area_2 = freq_2 * (block_valid_num - grid_y)
    area_original = area_0 + area_1 + area_2
    # [x_len, m_v]
    area_max = torch.amax(area_original)

    area = torch.where(
        mask,
        area_original,
        area_max,
    )
    spilt_point = torch.argmin(area)
    spilt_x = (spilt_point // block_valid_num).item()
    spilt_y = (spilt_point % block_valid_num).item()
    freq_max = freq_0.item()
    freq_x = freq_sorted[spilt_x].item()
    freq_y = freq_sorted[spilt_y].item()

    return freq_sorted, freq_sorted_indices, spilt_x, spilt_y, freq_max, freq_x, freq_y


def indices_func(
    indices,
    block_num,
    block_valid_num,
    num_output,
    indices_pre_results,
):
    freq_sorted, freq_sorted_indices, spilt_x, spilt_y, freq_max, freq_x, freq_y = (
        indices_pre_results
    )
    device = indices.device
    n = indices.shape[0]
    pos_seq = torch.arange(block_valid_num, device=device)
    pad_freqs = torch.cat(
        (
            freq_max - freq_sorted[:spilt_x],
            freq_x - freq_sorted[spilt_x:spilt_y],
            freq_y - freq_sorted[spilt_y:block_valid_num],
        ),
        0,
    )
    # [m_v]
    spilt_num_1 = freq_max * spilt_x
    spilt_num_2 = freq_x * (spilt_y - spilt_x)
    spilt_num_3 = freq_y * (block_valid_num - spilt_y)
    pad_num = spilt_num_1 + spilt_num_2 + spilt_num_3 - n
    indices_pad = torch.repeat_interleave(
        freq_sorted_indices[:block_valid_num], pad_freqs, 0, output_size=pad_num
    )
    # [n_p], n_p = pad_num
    indices_w_pad = torch.cat((indices, indices_pad), 0)
    # [n + n_p]
    indices_trans = torch.full(
        (block_num,), -1, device=device, dtype=pos_seq.dtype
    ).scatter_(0, freq_sorted_indices[:block_valid_num], pos_seq)
    # [m], m = block_num
    new_indices = indices_trans[indices_w_pad]
    # [n + n_p]
    # assert new_indices[order_indices[0]] == 0
    return (
        new_indices,
        freq_sorted_indices,
        block_valid_num,
        freq_max,
        freq_x,
        freq_y,
        spilt_x,
        spilt_y,
        spilt_num_1,
        spilt_num_2,
        spilt_num_3,
        n,
        num_output,
    )


class WBincount(Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,  # [n]
        weights: torch.Tensor,  # [n]
        num_bins: int,  # [m]
    ):
        w_bincount = torch.bincount(indices, weights=weights, minlength=num_bins)
        # [m]
        ctx.save_for_backward(indices)
        return w_bincount

    @staticmethod
    def backward(ctx, grad_w_bincount):
        indices = ctx.saved_tensors
        grad_weights = grad_w_bincount[indices]
        return (None, grad_weights, None)


weighted_bincount = WBincount.apply


class GroupedGEMM(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        probs: Optional[torch.Tensor],
        ffn: bool,
        fused: bool,
        router: bool,
        indices_results: Tuple,
    ) -> torch.Tensor:
        # TODO: no padding
        # TODO: verify
        # input: [s, n_m, n_i, d_i], weight: [m, d_i, d_o], bias: [m, 1, d_o] or None, probs: [s, n_m, n_i, n_o, 1]
        # for ffn:
        # input: [s, 1, n_fr, 2 * d_f], weight: [m, 3, d_f, d_fh], bias: [m, 1, d_f] or None (m = n_fb)
        # if router: input: [n, d_i]

        # new_indices, spilt_pos, freq_max, freq_sub, block_valid_num
        (
            new_indices,
            freq_sorted_indices,
            block_valid_num,
            freq_max,
            freq_x,
            freq_y,
            spilt_x,
            spilt_y,
            spilt_num_1,
            spilt_num_2,
            spilt_num_3,
            n,
            num_output,
        ) = indices_results

        order_indices = torch.argsort(new_indices, stable=False)
        # [n + n_p]
        if not router:
            seq_len, num_module, num_input = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
            )
        input_dim = input.shape[-1]
        device = input.device
        dtype = input.dtype
        n_total = new_indices.shape[0]

        order_indices_1 = order_indices[:spilt_num_1]
        order_indices_2 = order_indices[spilt_num_1 : spilt_num_1 + spilt_num_2]
        order_indices_3 = order_indices[spilt_num_1 + spilt_num_2 :]

        freq_sorted_indices_1 = freq_sorted_indices[:spilt_x]
        freq_sorted_indices_2 = freq_sorted_indices[spilt_x:spilt_y]
        freq_sorted_indices_3 = freq_sorted_indices[spilt_y:block_valid_num]

        ranks = torch.empty_like(order_indices).scatter_(
            0, order_indices, torch.arange(n_total, device=device)
        )
        # [n + n_p]
        ranks_n = ranks[:n]
        output_dim = weight.shape[-2] if ffn else weight.shape[-1]

        input_zone = torch.empty(n_total, input_dim, device=device, dtype=dtype)
        # [n + n_p, d_i (2 * d_f)]
        if router:
            input_repeat = input
        else:
            input_repeat = repeat(
                input, "s n_m n_i d_i -> (s n_m n_i n_o) d_i", n_o=num_output
            )
            # [n, d_i]
        input_zone[ranks_n] = input_repeat

        input_1 = input_zone[:spilt_num_1].reshape(-1, freq_max, input_dim)
        # [s_1, f_max, d_i (2 * d_f)], s_1 = spilt_num_1
        input_2 = input_zone[spilt_num_1 : spilt_num_1 + spilt_num_2].reshape(
            -1, freq_x, input_dim
        )
        # [s_2, f_x, d_i (2 * d_f)]
        input_3 = input_zone[spilt_num_1 + spilt_num_2 :].reshape(-1, freq_y, input_dim)
        # [s_3, f_y, d_i (2 * d_f)]

        weight_1 = weight[freq_sorted_indices_1]
        # [s_1, d_i, d_o] or [s_1, 3, d_f, d_fh]
        weight_2 = weight[freq_sorted_indices_2]
        # [s_2, d_i, d_o] or [s_2, 3, d_f, d_fh]
        weight_3 = weight[freq_sorted_indices_3]
        # [s_3, d_i, d_o] or [s_3, 3, d_f, d_fh]

        if ffn:
            ffn_input_dim = input_1.shape[-1] // 2
            input_1_g = input_1[:, :, :ffn_input_dim]
            # [s_1, f_max, d_f]
            input_1_l = input_1[:, :, ffn_input_dim:]
            # [s_1, f_max, d_f]
            input_2_g = input_2[:, :, :ffn_input_dim]
            # [s_2, f_x, d_f]
            input_2_l = input_2[:, :, ffn_input_dim:]
            # [s_2, f_x, d_f]
            input_3_g = input_3[:, :, :ffn_input_dim]
            # [s_3, f_y, d_f]
            input_3_l = input_3[:, :, ffn_input_dim:]
            # [s_3, f_y, d_f]

            output_1_g = torch.bmm(input_1_g, weight_1[:, 0, ...])
            # [s_1, f_max, d_f] bmm [s_1, d_f, d_fh] = [s_1, f_max, d_fh]
            output_1_l = torch.bmm(input_1_l, weight_1[:, 1, ...])
            output_2_g = torch.bmm(input_2_g, weight_2[:, 0, ...])
            # [s_2, f_x, d_f] bmm [s_2, d_f, d_fh] = [s_2, f_x, d_fh]
            output_2_l = torch.bmm(input_2_l, weight_2[:, 1, ...])
            output_3_g = torch.bmm(input_3_g, weight_3[:, 0, ...])
            # [s_3, f_x, d_f] bmm [s_3, d_f, d_fh] = [s_3, f_x, d_fh]
            output_3_l = torch.bmm(input_3_l, weight_3[:, 1, ...])

            output_1_o = F.silu(output_1_g) * output_1_l
            # [s_1, f_max, d_fh]
            output_2_o = F.silu(output_2_g) * output_2_l
            # [s_2, f_x, d_fh]
            output_3_o = F.silu(output_3_g) * output_3_l
            # [s_3, f_y, d_fh]

            output_1 = torch.bmm(output_1_o, weight_1[:, 2, ...].transpose(1, 2))
            # [s_1, f_max, d_fh] bmm [s_1, d_fh, d_f] = [s_1, f_max, d_f]
            output_2 = torch.bmm(output_2_o, weight_2[:, 2, ...].transpose(1, 2))
            # [s_2, f_x, d_fh] bmm [s_2, d_fh, d_f] = [s_2, f_x, d_f]
            output_3 = torch.bmm(output_3_o, weight_3[:, 2, ...].transpose(1, 2))
            # [s_3, f_y, d_fh] bmm [s_3, d_fh, d_f] = [s_3, f_y, d_f]
        else:
            output_1 = torch.bmm(input_1, weight_1)
            # [s_1, f_max, d_i] bmm [s_1, d_i, d_o] = [s_1, f_max, d_o]
            output_2 = torch.bmm(input_2, weight_2)
            # [s_2, f_x, d_i] bmm [s_2, d_i, d_o] = [s_2, f_x, d_o]
            output_3 = torch.bmm(input_3, weight_3)
            # [s_3, f_y, d_i] bmm [s_3, d_i, d_o] = [s_3, f_y, d_o]

        if bias is not None:
            bias = bias.to(dtype)
            bias_1 = bias[freq_sorted_indices_1]
            # [s_1, 1, d_o (d_f)]
            bias_2 = bias[freq_sorted_indices_2]
            # [s_2, 1, d_o (d_f)]
            bias_3 = bias[freq_sorted_indices_3]
            # [s_3, 1, d_o (d_f)]

            output_1 += bias_1
            output_2 += bias_2
            output_3 += bias_3

        output_1 = output_1.flatten(0, 1)
        output_2 = output_2.flatten(0, 1)
        output_3 = output_3.flatten(0, 1)

        output = torch.empty(
            new_indices.shape[0], output_dim, dtype=dtype, device=device
        )
        # [n + n_p, d_o (d_f)]

        output[order_indices_1] = output_1
        output[order_indices_2] = output_2
        output[order_indices_3] = output_3

        # TODO: Triton
        output = output[:n]
        # [n, d_o (d_f)]
        if not router:
            output = rearrange(
                output,
                "(s n_m n_i n_o) d_i -> s n_m n_i n_o d_i",
                s=seq_len,
                n_m=num_module,
                n_i=num_input,
                n_o=num_output,
            )
            # [s, n_m, n_i, n_o, d_o] or [s, 1, n_fr, k * n_iv, d_f]
            if fused:
                output = output * probs
                reduce_pos = 3 if ffn else 2
                output = torch.sum(output, reduce_pos, dtype=dtype)
                # [s, n_m, n_o, d_o] or [s, 1, n_fr, d_f]

        ctx.save_for_backward(
            input,
            probs,
            weight,
            bias,
            order_indices,
            freq_sorted_indices,
            ranks_n,
        )
        ctx.info = (
            ffn,
            fused,
            router,
            freq_max,
            freq_x,
            freq_y,
            spilt_x,
            spilt_y,
            spilt_num_1,
            spilt_num_2,
            spilt_num_3,
            n,
            num_output,
            block_valid_num,
        )
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        # return grad_input, None, grad_weight, grad_bias, None, None
        # grad_output:
        # not fused: [s, n_m, n_i, n_o, d_o] or [s, 1, n_fr, k * n_iv, d_f]
        # fused: [s, n_m, n_o, d_o] or [s, 1, n_fr, d_f]
        # if router: [n, d]
        device = grad_output.device
        dtype = grad_output.dtype
        (
            input,
            probs,
            weight,
            bias,
            order_indices,
            freq_sorted_indices,
            ranks_n,
        ) = ctx.saved_tensors
        (
            ffn,
            fused,
            router,
            freq_max,
            freq_x,
            freq_y,
            spilt_x,
            spilt_y,
            spilt_num_1,
            spilt_num_2,
            spilt_num_3,
            n,
            num_output,
            block_valid_num,
        ) = ctx.info
        # input: [s, n_m, n_i, d_i], order_indices: [n + n_p], freq_sorted_indices: [m]
        # weight: [m, d_i, d_o], bias: [m, 1, d_o] or None
        # for ffn:
        # input: [s, 1, n_fr, 2 * d_f]
        # weight: [m, 3, d_f, d_fh], bias: [m, 1, d_f] or None
        n_total, m, d_i, d_o = (
            order_indices.shape[0],
            freq_sorted_indices.shape[0],
            weight.shape[-2],
            weight.shape[-1],
        )
        if not router:
            seq_len, num_module, num_input = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
            )
        input_dim = input.shape[-1]
        grad_probs = None
        # TODO: smooth

        order_indices_1 = order_indices[:spilt_num_1]
        order_indices_2 = order_indices[spilt_num_1 : spilt_num_1 + spilt_num_2]
        order_indices_3 = order_indices[spilt_num_1 + spilt_num_2 :]

        freq_sorted_indices_1 = freq_sorted_indices[:spilt_x]
        freq_sorted_indices_2 = freq_sorted_indices[spilt_x:spilt_y]
        freq_sorted_indices_3 = freq_sorted_indices[spilt_y:block_valid_num]

        input_zone = torch.zeros(n_total, input_dim, device=device, dtype=dtype)
        # [n + n_p, d_i (2 * d_f)]
        # TODO: empty?
        if router:
            input_repeat = input
        else:
            input_repeat = repeat(
                input, "s n_m n_i d_i -> (s n_m n_i n_o) d_i", n_o=num_output
            )
            # [n, d_i] or [n, 2 * d_f]
        n = input_repeat.shape[0]
        input_zone[ranks_n] = input_repeat

        i_sorted_1 = input_zone[:spilt_num_1].reshape(-1, freq_max, input_dim)
        # [s_1, f_max, d_i (2 * d_f)], s_1 = spilt_num_1
        i_sorted_2 = input_zone[spilt_num_1 : spilt_num_1 + spilt_num_2].reshape(
            -1, freq_x, input_dim
        )
        # [s_2, f_x, d_i (2 * d_f)]
        i_sorted_3 = input_zone[spilt_num_1 + spilt_num_2 :].reshape(
            -1, freq_y, input_dim
        )
        # [s_3, f_y, d_i (2 * d_f)]

        weight_1 = weight[freq_sorted_indices_1]
        # [s_1, d_i, d_o] or [s_1, 3, d_f, d_fh]
        weight_2 = weight[freq_sorted_indices_2]
        # [s_2, d_i, d_o] or [s_2, 3, d_f, d_fh]
        weight_3 = weight[freq_sorted_indices_3]
        # [s_3, d_i, d_o] or [s_3, 3, d_f, d_fh]

        grad_o_dim = grad_output.shape[-1]
        grad_o_zone = torch.zeros(n_total, grad_o_dim, device=device, dtype=dtype)
        # [n + n_p, d_o (d_f)]
        if not router:
            if fused:
                if ffn:
                    grad_output = repeat(
                        grad_output,
                        "s n_m n_i d_i -> (s n_m n_i n_o) d_i",
                        n_o=num_output,
                    )
                    # [n, d_f]
                else:
                    grad_output = repeat(
                        grad_output,
                        "s n_m n_o d_i -> (s n_m n_i n_o) d_i",
                        n_i=num_input,
                    )
                    # [n, d_o]
            else:
                grad_output = grad_output.flatten(0, -2)
        grad_o_zone[ranks_n] = grad_output

        grad_o_sorted_1_pre = grad_o_zone[:spilt_num_1].reshape(
            -1, freq_max, grad_o_dim
        )
        # [s_1, f_max, d_o (d_f)]
        grad_o_sorted_2_pre = grad_o_zone[
            spilt_num_1 : spilt_num_1 + spilt_num_2
        ].reshape(-1, freq_x, grad_o_dim)
        # [s_2, f_x, d_o (d_f)]
        grad_o_sorted_3_pre = grad_o_zone[spilt_num_1 + spilt_num_2 :].reshape(
            -1, freq_y, grad_o_dim
        )
        # [s_3, f_y, d_o (d_f)]

        if fused:
            probs_zone = torch.zeros(n_total, device=device, dtype=dtype)
            # [n + n_p]
            probs_zone[ranks_n] = probs.flatten()

            probs_1 = probs_zone[:spilt_num_1].reshape(-1, freq_max, 1)
            # [s_1, f_max, 1]
            probs_2 = probs_zone[spilt_num_1 : spilt_num_1 + spilt_num_2].reshape(
                -1, freq_x, 1
            )
            # [s_2, f_x, 1]
            probs_3 = probs_zone[spilt_num_1 + spilt_num_2 :].reshape(-1, freq_y, 1)
            # [s_3, f_y, 1]

            grad_o_sorted_1 = grad_o_sorted_1_pre * probs_1
            grad_o_sorted_2 = grad_o_sorted_2_pre * probs_2
            grad_o_sorted_3 = grad_o_sorted_3_pre * probs_3
        else:
            grad_o_sorted_1 = grad_o_sorted_1_pre
            grad_o_sorted_2 = grad_o_sorted_2_pre
            grad_o_sorted_3 = grad_o_sorted_3_pre

        if fused:
            if ffn:
                dim = input_dim // 2
                i_sorted_1_g = i_sorted_1[:, :, :dim]
                # [s_1, f_max, d_f]
                i_sorted_1_l = i_sorted_1[:, :, dim:]
                i_sorted_2_g = i_sorted_2[:, :, :dim]
                # [s_2, f_x, d_f]
                i_sorted_2_l = i_sorted_2[:, :, dim:]
                i_sorted_3_g = i_sorted_3[:, :, :dim]
                # [s_3, f_y, d_f]
                i_sorted_3_l = i_sorted_3[:, :, dim:]

                output_1_g = torch.bmm(i_sorted_1_g, weight_1[:, 0, ...])
                # [s_1, f_max, d_f] bmm [s_1, d_f, d_fh] = [s_1, f_max, d_fh]
                output_1_l = torch.bmm(i_sorted_1_l, weight_1[:, 1, ...])
                output_2_g = torch.bmm(i_sorted_2_g, weight_2[:, 0, ...])
                # [s_2, f_x, d_f] bmm [s_2, d_f, d_fh] = [s_2, f_x, d_fh]
                output_2_l = torch.bmm(i_sorted_2_l, weight_2[:, 1, ...])
                output_3_g = torch.bmm(i_sorted_3_g, weight_3[:, 0, ...])
                # [s_3, f_y, d_f] bmm [s_2, d_f, d_fh] = [s_2, f_y, d_fh]
                output_3_l = torch.bmm(i_sorted_3_l, weight_3[:, 1, ...])

                output_1_o = F.silu(output_1_g) * output_1_l
                # [s_1, f_max, d_fh]
                output_2_o = F.silu(output_2_g) * output_2_l
                # [s_2, f_x, d_fh]
                output_3_o = F.silu(output_3_g) * output_3_l
                # [s_3, f_y, d_fh]

                output_1 = torch.bmm(output_1_o, weight_1[:, 2, ...].transpose(1, 2))
                # [s_1, f_max, d_f]
                output_2 = torch.bmm(output_2_o, weight_2[:, 2, ...].transpose(1, 2))
                # [s_2, f_x, d_f]
                output_3 = torch.bmm(output_3_o, weight_3[:, 2, ...].transpose(1, 2))
                # [s_3, f_y, d_f]
            else:
                output_1 = torch.bmm(i_sorted_1, weight_1)
                # [s_1, f_max, d_o]
                output_2 = torch.bmm(i_sorted_2, weight_2)
                # [s_2, f_x, d_o]
                output_3 = torch.bmm(i_sorted_3, weight_3)
                # [s_3, f_y, d_o]

            grad_probs_1 = torch.sum(
                grad_o_sorted_1_pre * output_1, -1, dtype=dtype
            ).flatten()
            # [s_1 * f_max]
            grad_probs_2 = torch.sum(
                grad_o_sorted_2_pre * output_2, -1, dtype=dtype
            ).flatten()
            # [s_2 * f_x]
            grad_probs_3 = torch.sum(
                grad_o_sorted_3_pre * output_3, -1, dtype=dtype
            ).flatten()
            # [s_3 * f_y]

            grad_probs = torch.empty(n_total, device=device, dtype=dtype)
            # [n + n_p]
            grad_probs[order_indices_1] = grad_probs_1
            grad_probs[order_indices_2] = grad_probs_2
            grad_probs[order_indices_3] = grad_probs_3
            grad_probs = grad_probs[:n]
            # [n]
            grad_probs = rearrange(
                grad_probs,
                "(s n_m n_i n_o) -> s n_m n_i n_o 1",
                s=seq_len,
                n_m=num_module,
                n_i=num_input,
                n_o=num_output,
            )
            # [s, n_m, n_i, n_o, 1] or [s, 1, n_fr, k * n_iv, 1]

        grad_bias = None
        if bias is not None:
            grad_bias_1 = torch.sum(grad_o_sorted_1, 1, keepdim=True, dtype=dtype)
            # [s_1, 1, d_o (d_f)]
            grad_bias_2 = torch.sum(grad_o_sorted_2, 1, keepdim=True, dtype=dtype)
            # [s_2, 1, d_o (d_f)]
            grad_bias_3 = torch.sum(grad_o_sorted_3, 1, keepdim=True, dtype=dtype)
            # [s_3, 1, d_o (d_f)]

            grad_bias = torch.zeros_like(bias, dtype=dtype)
            # [m, 1, d_o (d_f)]
            grad_bias[freq_sorted_indices_1] = grad_bias_1
            grad_bias[freq_sorted_indices_2] = grad_bias_2
            grad_bias[freq_sorted_indices_3] = grad_bias_3

        if ffn:
            if not fused:
                dim = input_dim // 2
                i_sorted_1_g = i_sorted_1[:, :, :dim]
                # [s_1, f_max, d_f]
                i_sorted_1_l = i_sorted_1[:, :, dim:]
                i_sorted_2_g = i_sorted_2[:, :, :dim]
                # [s_2, f_x, d_f]
                i_sorted_2_l = i_sorted_2[:, :, dim:]
                i_sorted_3_g = i_sorted_3[:, :, :dim]
                # [s_3, f_y, d_f]
                i_sorted_3_l = i_sorted_3[:, :, dim:]

                output_1_g = torch.bmm(i_sorted_1_g, weight_1[:, 0, ...])
                # [s_1, f_max, d_f] bmm [s_1, d_f, d_fh] = [s_1, f_max, d_fh]
                output_1_l = torch.bmm(i_sorted_1_l, weight_1[:, 1, ...])
                output_2_g = torch.bmm(i_sorted_2_g, weight_2[:, 0, ...])
                # [s_2, f_x, d_f] bmm [s_2, d_f, d_fh] = [s_2, f_x, d_fh]
                output_2_l = torch.bmm(i_sorted_2_l, weight_2[:, 1, ...])
                output_3_g = torch.bmm(i_sorted_3_g, weight_3[:, 0, ...])
                # [s_3, f_y, d_f] bmm [s_3, d_f, d_fh] = [s_3, f_y, d_fh]
                output_3_l = torch.bmm(i_sorted_3_l, weight_3[:, 1, ...])

                output_1_o = F.silu(output_1_g) * output_1_l
                # [s_1, f_max, d_fh]
                output_2_o = F.silu(output_2_g) * output_2_l
                # [s_2, f_x, d_fh]
                output_3_o = F.silu(output_3_g) * output_3_l
                # [s_3, f_y, d_fh]

            grad_output_1_o = torch.bmm(grad_o_sorted_1, weight_1[:, 2, ...])
            # [s_1, f_max, d_f] bmm [s_1, d_f, d_fh] = [s_1, f_max, d_fh]
            grad_output_2_o = torch.bmm(grad_o_sorted_2, weight_2[:, 2, ...])
            # [s_2, f_x, d_f] bmm [s_2, d_f, d_fh] = [s_2, f_x, d_fh]
            grad_output_3_o = torch.bmm(grad_o_sorted_3, weight_3[:, 2, ...])
            # [s_3, f_y, d_f] bmm [s_3, d_y, d_fh] = [s_3, f_y, d_fh]

            grad_weight_1_o = (
                torch.bmm(output_1_o.transpose(1, 2), grad_o_sorted_1)
                .transpose(1, 2)
                .unsqueeze(1)
            )
            # [s_1, d_fh, f_max] bmm [s_1, f_max, d_f] = [s_1, d_fh, d_f]
            # transpose(1, 2).unsqueeze(1) -> [s_1, 1, d_f, d_fh]
            grad_weight_2_o = (
                torch.bmm(output_2_o.transpose(1, 2), grad_o_sorted_2)
                .transpose(1, 2)
                .unsqueeze(1)
            )
            # [s_2, d_fh, f_x] bmm [s_2, f_x, d_f] = [s_2, d_fh, d_f]
            # transpose(1, 2).unsqueeze(1) -> [s_2, 1, d_f, d_fh]
            grad_weight_3_o = (
                torch.bmm(output_3_o.transpose(1, 2), grad_o_sorted_3)
                .transpose(1, 2)
                .unsqueeze(1)
            )
            # [s_3, d_fh, f_y] bmm [s_3, f_y, d_f] = [s_3, d_fh, d_f]
            # transpose(1, 2).unsqueeze(1) -> [s_3, 1, d_f, d_fh]

            grad_output_1_g = grad_output_1_o * output_1_l
            # [s_1, f_max, d_fh]
            output_1_g_s = torch.sigmoid(output_1_g)
            # [s_1, f_max, d_fh]
            grad_output_1_g = grad_output_1_g * (
                output_1_g_s + output_1_g * output_1_g_s * (1 - output_1_g_s)
            )
            # [s_1, f_max, d_fh]
            grad_weight_1_g = torch.bmm(
                i_sorted_1_g.transpose(1, 2), grad_output_1_g
            ).unsqueeze(1)
            # [s_1, d_f, f_max] bmm [s_1, f_max, d_fh] = [s_1, d_f, d_fh]
            # unsqueeze(1) -> [s_1, 1, d_f, d_fh]
            grad_input_1_g = torch.bmm(
                grad_output_1_g, weight_1[:, 0, ...].transpose(1, 2)
            ).flatten(0, 1)
            # [s_1, f_max, d_fh] bmm [s_1, d_fh, d_f] = [s_1, f_max, d_f] -> [s_1 * f_max, d_f]

            grad_output_2_g = grad_output_2_o * output_2_l
            # [s_2, f_x, d_fh]
            output_2_g_s = torch.sigmoid(output_2_g)
            # [s_2, f_x, d_fh]
            grad_output_2_g = grad_output_2_g * (
                output_2_g_s + output_2_g * output_2_g_s * (1 - output_2_g_s)
            )
            # [s_2, f_x, d_fh]
            grad_weight_2_g = torch.bmm(
                i_sorted_2_g.transpose(1, 2), grad_output_2_g
            ).unsqueeze(1)
            # [s_2, d_f, f_x] bmm [s_2, f_x, d_fh] = [s_2, d_f, d_fh]
            # unsqueeze(1) -> [s_2, 1, d_f, d_fh]
            grad_input_2_g = torch.bmm(
                grad_output_2_g, weight_2[:, 0, ...].transpose(1, 2)
            ).flatten(0, 1)
            # [s_2, f_x, d_fh] bmm [s_2, d_fh, d_f] = [s_2, f_x, d_f] -> [s_2 * f_x, d_f]

            grad_output_3_g = grad_output_3_o * output_3_l
            # [s_3, f_y, d_fh]
            output_3_g_s = torch.sigmoid(output_3_g)
            # [s_3, f_y, d_fh]
            grad_output_3_g = grad_output_3_g * (
                output_3_g_s + output_3_g * output_3_g_s * (1 - output_3_g_s)
            )
            # [s_3, f_y, d_fh]
            grad_weight_3_g = torch.bmm(
                i_sorted_3_g.transpose(1, 2), grad_output_3_g
            ).unsqueeze(1)
            # [s_3, d_f, f_y] bmm [s_3, f_y, d_fh] = [s_3, d_f, d_fh]
            # unsqueeze(1) -> [s_3, 1, d_f, d_fh]
            grad_input_3_g = torch.bmm(
                grad_output_3_g, weight_3[:, 0, ...].transpose(1, 2)
            ).flatten(0, 1)
            # [s_3, f_x, d_fh] bmm [s_3, d_fh, d_f] = [s_3, f_y, d_f] -> [s_3 * f_y, d_f]

            grad_output_1_l = grad_output_1_o * output_1_g
            # [s_1, f_max, d_fh]
            grad_output_2_l = grad_output_2_o * output_2_g
            # [s_2, f_x, d_fh]
            grad_output_3_l = grad_output_3_o * output_3_g
            # [s_3, f_y, d_fh]

            grad_weight_1_l = torch.bmm(
                i_sorted_1_l.transpose(1, 2), grad_output_1_l
            ).unsqueeze(1)
            # [s_1, d_f, f_max] bmm [s_1, f_max, d_fh] = [s_1, d_f, d_fh]
            # unsqueeze(1) -> [s_1, 1, d_f, d_fh]
            grad_input_1_l = torch.bmm(
                grad_output_1_l, weight_1[:, 1, ...].transpose(1, 2)
            ).flatten(0, 1)
            # [s_1, f_max, d_fh] bmm [s_1, d_fh, d_f] = [s_1, f_max, d_f] -> [s_1 * f_max, d_f]
            grad_weight_2_l = torch.bmm(
                i_sorted_2_l.transpose(1, 2), grad_output_2_l
            ).unsqueeze(1)
            # [s_2, d_f, f_x] bmm [s_2, f_x, d_fh] = [s_2, d_f, d_fh]
            # unsqueeze(1) -> [s_2, 1, d_f, d_fh]
            grad_input_2_l = torch.bmm(
                grad_output_2_l, weight_2[:, 1, ...].transpose(1, 2)
            ).flatten(0, 1)
            # [s_2, f_x, d_fh] bmm [s_2, d_fh, d_f] = [s_2, f_x, d_f] -> [s_2 * f_x, d_f]
            grad_weight_3_l = torch.bmm(
                i_sorted_3_l.transpose(1, 2), grad_output_3_l
            ).unsqueeze(1)
            # [s_3, d_f, f_y] bmm [s_3, f_y, d_fh] = [s_3, d_f, d_fh]
            # unsqueeze(1) -> [s_3, 1, d_f, d_fh]
            grad_input_3_l = torch.bmm(
                grad_output_3_l, weight_3[:, 1, ...].transpose(1, 2)
            ).flatten(0, 1)
            # [s_3, f_y, d_fh] bmm [s_3, d_fh, d_f] = [s_3, f_y, d_f] -> [s_3 * f_y, d_f]

            grad_input = torch.empty(n_total, input_dim, dtype=dtype, device=device)
            # [n + n_p, 2 * d_f]
            grad_input[order_indices_1, :dim] = grad_input_1_g
            grad_input[order_indices_2, :dim] = grad_input_2_g
            grad_input[order_indices_3, :dim] = grad_input_3_g
            grad_input[order_indices_1, dim:] = grad_input_1_l
            grad_input[order_indices_2, dim:] = grad_input_2_l
            grad_input[order_indices_3, dim:] = grad_input_3_l
            grad_input = grad_input[:n]
            # [n, 2 * d_f]

            grad_weight_1 = torch.cat(
                (grad_weight_1_g, grad_weight_1_l, grad_weight_1_o), 1
            )
            # [s_1, 3, d_f, d_fh]
            grad_weight_2 = torch.cat(
                (grad_weight_2_g, grad_weight_2_l, grad_weight_2_o), 1
            )
            # [s_2, 3, d_f, d_fh]
            grad_weight_3 = torch.cat(
                (grad_weight_3_g, grad_weight_3_l, grad_weight_3_o), 1
            )
            # [s_3, 3, d_f, d_fh]

        else:
            grad_input_1 = torch.bmm(grad_o_sorted_1, weight_1.transpose(1, 2)).flatten(
                0, 1
            )
            # [s_1, f_max, d_o] bmm [s_1, d_o, d_i] = [s_1, f_max, d_i] -> [s_1 * f_max, d_i]
            grad_input_2 = torch.bmm(grad_o_sorted_2, weight_2.transpose(1, 2)).flatten(
                0, 1
            )
            # [s_2, f_x, d_o] bmm [s_2, d_o, d_i] = [s_2, f_x, d_i] -> [s_2 * f_x, d_i]
            grad_input_3 = torch.bmm(grad_o_sorted_3, weight_3.transpose(1, 2)).flatten(
                0, 1
            )
            # [s_3, f_y, d_o] bmm [s_3, d_o, d_i] = [s_3, f_y, d_i] -> [s_3 * f_y, d_i]

            grad_input = torch.empty(n_total, input_dim, dtype=dtype, device=device)
            # [n + n_p, d_i]
            grad_input[order_indices_1] = grad_input_1
            grad_input[order_indices_2] = grad_input_2
            grad_input[order_indices_3] = grad_input_3
            grad_input = grad_input[:n]
            # [n, d_i]

            grad_weight_1 = torch.bmm(i_sorted_1.transpose(1, 2), grad_o_sorted_1)
            # [s_1, d_i, f_max] bmm [s_1, f_max, d_o] = [s_1, d_i, d_o]
            grad_weight_2 = torch.bmm(i_sorted_2.transpose(1, 2), grad_o_sorted_2)
            # [s_2, d_i, f_x] bmm [s_2, f_x, d_o] = [s_2, d_i, d_o]
            grad_weight_3 = torch.bmm(i_sorted_3.transpose(1, 2), grad_o_sorted_3)
            # [s_3, d_i, f_y] bmm [s_3, f_y, d_o] = [s_3, d_i, d_o]

        grad_weight = torch.zeros_like(weight, dtype=dtype)
        # [m, d_i, d_o] or [m, 3, d_f, d_fh]
        grad_weight[freq_sorted_indices_1] = grad_weight_1
        grad_weight[freq_sorted_indices_2] = grad_weight_2
        grad_weight[freq_sorted_indices_3] = grad_weight_3

        if not router:
            grad_input = rearrange(
                grad_input,
                "(s n_m n_i n_o) d -> s n_m n_i n_o d",
                s=seq_len,
                n_m=num_module,
                n_i=num_input,
                n_o=num_output,
            )
            # [s, n_m, n_i, n_o, d_i] or [s, 1, n_fr, k * n_iv, 2 * d_f]
            grad_input = torch.sum(grad_input, 3, dtype=dtype)
            # [s, n_m, n_i, d_i] or [s, 1, n_fr, 2 * d_f]

        return (
            grad_input,
            grad_weight,
            grad_bias,
            grad_probs,
            None,
            None,
            None,
            None,
        )


grouped_gemm = GroupedGEMM.apply


class Router(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_id: int,
        vector_pool: ParamVectorPool,
    ):
        super().__init__()

        self.device = f"cuda:{torch.cuda.current_device()}"
        self.layer_id = layer_id
        self.config = config
        self.dtype = self.config.dtype
        self.vector_pool = vector_pool
        self.n_lm = len(config.linear_module_list)
        self.module_id = torch.arange(self.n_lm, device=self.device).reshape(
            1, -1, 1, 1
        )
        # [1, n_lm, 1, 1]

    def forward(
        self,
        query_linear: torch.Tensor,  # [n_lm, s * n_lr, v]
        query_ffn: torch.Tensor,  # [s, n_fr, v]
    ) -> Tuple[Tuple, Tuple]:
        linear_cluster_values, linear_cluster_indices_original = torch.max(
            torch.bmm(
                query_linear, self.vector_pool.linear_vec["vec_clusters"][self.layer_id]
            ),
            -1,
        )
        # [n_lm, s * n_lr, v] bmm [n_lm, v, n_lc] = [n_lm, s * n_lr, n_lc]
        # [n_lm, s * n_lr, n_lc] (max, dim = -1) -> [n_lm, s * n_lr] * 2
        linear_cluster_indices_original = rearrange(
            linear_cluster_indices_original,
            "n_lm (s n_lr) -> s n_lm n_lr 1",
            s=self.config.seq_len,
        )
        # [s, n_lm, n_lr, 1]
        linear_cluster_indices_flatten = (
            linear_cluster_indices_original
            + self.module_id * self.config.linear_cluster_num
        ).flatten()
        # [s * n_lm * n_lr]
        num_linear_c_indices = linear_cluster_indices_flatten.shape[0]

        query_ffn = query_ffn.flatten(0, 1)
        # [s * n_fr, v]
        ffn_cluster_values, ffn_cluster_indices = torch.max(
            torch.matmul(
                query_ffn, self.vector_pool.ffn_vec["vec_clusters"][self.layer_id]
            ),
            -1,
        )
        # [s * n_fr, v] mm [v, n_fc] = [s * n_fr, n_fc]
        # [s * n_fr, n_fc] (max, dim = -1) -> [s * n_fr] * 2
        ffn_cluster_indices = rearrange(
            ffn_cluster_indices, "(s n_fr) -> s 1 n_fr 1", s=self.config.seq_len
        )
        # [s, 1, n_fr, 1]
        ffn_cluster_indices_flatten = ffn_cluster_indices.flatten()
        # [s * n_fr]

        num_linear_c_bins = self.n_lm * self.config.linear_cluster_num
        num_ffn_c_bins = self.config.ffn_cluster_num
        num_c_bins = num_linear_c_bins + num_ffn_c_bins

        cluster_indices = torch.cat(
            (
                linear_cluster_indices_flatten,
                ffn_cluster_indices_flatten + num_linear_c_bins,
            ),
            0,
        )
        # [s * n_lm * n_lr + s * n_fr]

        query = torch.cat(
            (
                rearrange(
                    query_linear,
                    "n_lm (s n_lr) v -> (s n_lm n_lr) v",
                    s=self.config.seq_len,
                ),
                query_ffn,
            ),
            0,
        )
        # [s * n_lm * n_lr + s * n_fr, v]

        weight = self.vector_pool.vec_pool[self.layer_id]
        # [n_lm * n_lc + n_fc, v, n_cv]
        block_num = weight.shape[0]
        # n_lm * n_lc + n_fc
        freqs = torch.bincount(cluster_indices, minlength=block_num)
        block_valid_num = torch.sum(freqs > 0)

        indices_pre_results = indices_pre_func(
            freqs, block_valid_num, self.config.x_len
        )
        indices_results = indices_func(
            cluster_indices,
            block_num,
            block_valid_num,
            1,
            indices_pre_results,
        )

        vec_scores = grouped_gemm(
            query,
            weight,
            None,
            None,
            False,
            False,
            True,
            indices_results,
        )
        # [s * n_lm * n_lr + s * n_fr, n_cv]

        linear_vec_scores = rearrange(
            vec_scores[:num_linear_c_indices],
            "(s n_lm n_lr) n_cv -> s n_lm n_lr n_cv",
            s=self.config.seq_len,
            n_lm=self.n_lm,
        )
        # [s, n_lm, n_lr, n_cv]
        ffn_vec_scores = rearrange(
            vec_scores[num_linear_c_indices:],
            "(s n_fr) n_cv -> s 1 n_fr n_cv",
            s=self.config.seq_len,
        )
        # [s, 1, n_fr, n_cv]

        linear_vec_values, linear_vec_indices = torch.max(
            linear_vec_scores, -1, keepdim=True
        )
        # [s, n_lm, n_lr, 1] * 2
        ffn_vec_values, ffn_vec_indices = torch.topk(
            ffn_vec_scores, self.config.ffn_top_k, dim=-1
        )
        # [s, 1, n_fr, k] * 2, k = ffn_top_k

        linear_block_indices = self.vector_pool.linear_vec_info["block_indices"][
            self.layer_id
        ][
            self.module_id, linear_cluster_indices_original, linear_vec_indices, :
        ].squeeze(3)
        # [s, n_lm, n_lr, n_iv]
        linear_gain_factors_pool = self.vector_pool.linear_vec["gain_factors"][
            self.layer_id
        ].to(self.dtype)
        linear_vec_gain = linear_gain_factors_pool[
            self.module_id, linear_cluster_indices_original, linear_vec_indices, :
        ].flatten(3)
        # [s, n_lm, n_lr, n_iv]

        ffn_block_indices = self.vector_pool.ffn_vec_info["block_indices"][
            self.layer_id
        ][ffn_cluster_indices, ffn_vec_indices, :]
        # [s, 1, n_fr, k, n_iv]
        ffn_gain_factors_pool = self.vector_pool.ffn_vec["gain_factors"][
            self.layer_id
        ].to(self.dtype)
        ffn_vec_gain = ffn_gain_factors_pool[ffn_cluster_indices, ffn_vec_indices, :]
        # [s, 1, n_fr, k, n_iv]

        linear_c_probs_aux = torch.sigmoid(
            rearrange(
                linear_cluster_values,
                "n_lm (s n_lr) -> (s n_lm n_lr)",
                s=self.config.seq_len,
            )
        )
        # [s * n_lm * n_lr]
        linear_vec_probs_aux = torch.sigmoid(linear_vec_values)
        # [s, n_lm, n_lr, 1]
        linear_vec_probs = linear_vec_probs_aux * torch.sigmoid(linear_vec_gain) * 2
        # [s, n_lm, n_lr, n_iv]

        ffn_c_probs_aux = torch.sigmoid(ffn_cluster_values)
        # [s * n_fr]
        ffn_vec_probs_aux = torch.sigmoid(ffn_vec_values).unsqueeze(-1)
        # [s, 1, n_fr, k, 1]
        ffn_vec_probs = ffn_vec_probs_aux * torch.sigmoid(ffn_vec_gain) * 2
        # [s, 1, n_fr, k, n_iv]

        c_probs_aux = torch.cat(
            (linear_c_probs_aux, ffn_c_probs_aux * self.config.c_ffn_balance_ratio), 0
        )
        # [s * n_lm * n_lr + s * n_fr]
        with torch.no_grad():
            c_indices_bincount = torch.bincount(cluster_indices, minlength=num_c_bins)
        c_probs_bincount = weighted_bincount(cluster_indices, c_probs_aux, num_c_bins)

        c_aux_loss = (
            torch.dot(
                c_indices_bincount.type_as(c_probs_bincount),
                c_probs_bincount,
            )
            / (cluster_indices.numel() * torch.sum(c_probs_bincount))
            * num_c_bins
        )

        vec_probs_aux = torch.cat(
            (
                linear_vec_probs_aux.flatten(),
                ffn_vec_probs_aux.flatten() * self.config.vec_ffn_balance_ratio,
            ),
            0,
        )
        # [s * n_lm * n_lr + s * n_fr * k]

        num_linear_vec_bins = num_linear_c_bins * self.config.c_vec_num
        num_ffn_vec_bins = num_ffn_c_bins * self.config.c_vec_num
        num_vec_bins = num_linear_vec_bins + num_ffn_vec_bins

        linear_vec_indices_aux = (
            linear_vec_indices.flatten()
            + linear_cluster_indices_flatten * self.config.c_vec_num
        )
        # [s * n_lm * n_lr]
        ffn_vec_indices_aux = (
            ffn_vec_indices
            + ffn_cluster_indices * self.config.c_vec_num
            + num_linear_vec_bins
        ).flatten()
        # [s * n_fr * k]
        vec_indices_aux = torch.cat((linear_vec_indices_aux, ffn_vec_indices_aux), 0)
        # [s * n_lm * n_lr + s * n_fr * k]

        with torch.no_grad():
            vec_indices_bincount = torch.bincount(
                vec_indices_aux, minlength=num_vec_bins
            )
        vec_probs_bincount = weighted_bincount(
            vec_indices_aux, vec_probs_aux, num_vec_bins
        )

        vec_aux_loss = (
            torch.dot(
                vec_indices_bincount.type_as(vec_probs_bincount),
                vec_probs_bincount,
            )
            / (vec_indices_aux.numel() * torch.sum(vec_probs_bincount))
            * num_vec_bins
        )

        router_results = (
            linear_block_indices,
            linear_vec_probs,
            ffn_block_indices,
            ffn_vec_probs,
        )

        aux_loss = (
            c_aux_loss,
            vec_aux_loss,
        )

        return router_results, aux_loss


class External(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_id: int,
    ):
        super().__init__()

        have_router, _ = layer_info(layer_id, config)
        input_dim = (
            config.main_dim + 2 * config.compressed_dim
            if layer_id != 0
            else config.main_dim
        )
        self.type1_dim = (
            int(4 * config.compressed_dim / config.linear_block_out_dim)
            * config.vec_dim
        )
        self.type2_dim = (
            int(6 * config.main_dim / config.linear_block_out_dim) * config.vec_dim
        )
        output_dim = (
            self.type1_dim
            + self.type2_dim
            + int(config.main_dim / config.ffn_block_dim) * config.vec_dim
        )

        self.config = config
        self.layer_id = layer_id
        self.dtype = self.config.dtype
        self.external = config.external and have_router
        if self.external:
            self.external_w1 = nn.Parameter(
                torch.zeros(input_dim, config.external_c_dim)
            )
            # [d_m + 2 * d_c, d_ec] or [d_m, d_ec], d_ec = external_c_dim
            self.external_w2 = nn.Parameter(
                torch.zeros(config.external_c_dim, output_dim)
            )
            # [d_ec, d_eo], d_eo = external_output_dim

        factor_num = 0
        if self.external:
            factor_num += 2
            if layer_id == 0:
                factor_num -= 1
        if config.residual_norm:
            factor_num += 1

        self.external_factors = nn.Parameter(torch.zeros(1, factor_num))
        # [1, factor_num]

    def forward(
        self,
        x: torch.Tensor,  # [s, d_m]
        attn_c: Optional[torch.Tensor],  # [s, d_c]
        norm: torch.Tensor,  # [s, 1]
        external_factors: torch.Tensor,  # [1, factor_num]
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        x_external_output = None
        residual_norm_coeff = None

        if self.external_factors.shape[-1] == 0:
            return x_external_output, residual_norm_coeff

        norm_coeff = torch.pow(norm, external_factors).to(self.dtype)  # [s, factor_num]

        if self.config.residual_norm:
            residual_norm_coeff = norm_coeff[:, -1].unsqueeze(-1)
            # [s, 1]

        if not self.external:
            return x_external_output, residual_norm_coeff

        external_x_coeff = norm_coeff[:, 0].unsqueeze(-1)
        if self.layer_id != 0:
            external_attn_coeff = norm_coeff[:, 1].unsqueeze(-1)
            x_cat = torch.cat(
                (x * external_x_coeff, attn_c, attn_c * external_attn_coeff), 1
            )
            # [s, d_m + 2 * d_c]
        else:
            x_cat = x * external_x_coeff
            # [s, d_m]

        x_external = torch.matmul(x_cat, self.external_w1)
        # layer_id != 0: [s, d_m + 2 * d_c] mm [d_m + 2 * d_c, d_ec] = [s, d_ec]
        # layer_id == 0: [s, d_m] mm [d_m, d_ec] = [s, d_ec]
        x_external = torch.matmul(x_external, self.external_w2)
        # [s, d_ec] mm [d_ec, d_eo] = [s, d_eo]

        x_e_linear_type1 = x_external[:, : self.type1_dim]
        # [s, type1_dim]
        x_e_linear_type2 = x_external[
            :, self.type1_dim : self.type1_dim + self.type2_dim
        ]
        # [s, type2_dim]
        x_e_ffn = x_external[:, self.type1_dim + self.type2_dim :]
        # [s, ffn_dim]

        x_e_linear_type1 = rearrange(
            x_e_linear_type1,
            "s (n_1m n_1mh v) -> s n_1m 1 n_1mh v",
            n_1m=4,
            v=self.config.vec_dim,
        )
        # [s, n_1m, 1, n_1mh, v]
        x_e_linear_type2 = rearrange(
            x_e_linear_type2,
            "s (n_2m n_2mh v) -> s n_2m 1 n_2mh v",
            n_2m=6,
            v=self.config.vec_dim,
        )
        # [s, n_2m, 1, n_2mh, v]
        x_e_ffn = rearrange(x_e_ffn, "s (n_fr v) -> s n_fr v", v=self.config.vec_dim)
        # [s, n_fr, v]

        x_external_output = {
            "linear_type1": x_e_linear_type1,
            "linear_type2": x_e_linear_type2,
            "ffn": x_e_ffn,
        }

        return x_external_output, residual_norm_coeff


class RotaryEmb(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        device = f"cuda:{torch.cuda.current_device()}"
        self.config = config
        freqs = (1 / config.theta) ** (
            torch.arange(0, config.attn_block_dim, 2) / config.attn_block_dim
        )
        # [d_a / 2]
        outer_freqs = torch.outer(torch.arange(config.seq_len), freqs)
        # [s, d_a / 2]
        self.freqs_cis = torch.polar(torch.ones_like(outer_freqs), outer_freqs)[
            None, None, :, :
        ].to(device)
        # [1, 1, s, d_a / 2]

    def forward(self, x_qk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_qk_complex = torch.view_as_complex(
            x_qk.float().reshape(*x_qk.shape[:-1], -1, 2)
        )
        # [2, h, s, d_a] -> [2, h, s, d_a / 2, 2] -> [2, h, s, d_a / 2]
        x_qk = (
            torch.view_as_real(x_qk_complex * self.freqs_cis).flatten(3).type_as(x_qk)
        )
        # [2, h, s, d_a / 2] -> [2, h, s, d_a / 2, 2] -> [2, h, s, d_a]
        x_q, x_k = torch.split(x_qk, 1, 0)
        # [1, h, s, d_a] * 2
        return x_q, x_k


class Attention(nn.Module):
    def __init__(self, config: Config, rotary_emb: RotaryEmb):
        super().__init__()

        self.config = config
        self.dtype = self.config.dtype
        self.rotary_emb = rotary_emb

    def forward(
        self,
        x_attn: torch.Tensor,  # [s, 3, d_m]
        block_mask: BlockMask,
    ) -> torch.Tensor:
        x_attn = rearrange(
            x_attn, "s c (h d_a) -> c h s d_a", d_a=self.config.attn_block_dim
        )
        # [3, h, s, d_a], d_a = attn_block_dim, h = head_num
        x_qk = x_attn[:2]
        # [2, h, s, d_a]
        x_v = x_attn[-1].unsqueeze(0)
        # [1, h, s, d_a]

        if self.config.qk_norm:
            x_qk = F.normalize(x_qk, dim=-1, eps=self.config.norm_eps).to(self.dtype)
            # [2, h, s, d_a]

        x_q, x_k = self.rotary_emb(x_qk)
        # [1, h, s, d_a] * 2
        kernel_options = (
            self.config.kernel_options if self.config.gpu == "4090" else None
        )
        x_attn_output = flex_attention(
            x_q,
            x_k,
            x_v,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )
        # [1, h, s, d_a]
        x_attn_output = rearrange(x_attn_output, "c h s d_a -> s c (h d_a)")
        # [s, 1, d_m]
        return x_attn_output


class Layer(nn.Module):
    def __init__(
        self,
        config: Config,
        layer_id: int,
        block_pool: ParamBlockPool,
        vector_pool: ParamVectorPool,
        attention: Attention,
    ):
        super().__init__()

        n_fr = int(config.main_dim / config.ffn_block_dim)
        self.n_1m = 4
        self.n_2m = 6
        self.n_1mh = int(config.compressed_dim / config.linear_block_out_dim)
        self.n_2mh = int(config.main_dim / config.linear_block_out_dim)

        device = f"cuda:{torch.cuda.current_device()}"
        self.have_router, self.vanilla_matmul = layer_info(layer_id, config)
        self.config = config
        self.dtype = self.config.dtype
        self.layer_id = layer_id
        self.vector_pool = vector_pool
        self.block_pool = block_pool
        self.attention = attention
        self.external = External(config, layer_id)
        if self.have_router:
            self.query = Query(config)
            self.router = Router(config, layer_id, vector_pool)

        self.elementwise_main_coeff = nn.Parameter(torch.ones(1, config.main_dim))
        # [1, d_m]
        self.elementwise_c_coeff = nn.Parameter(torch.ones(1, 2, config.compressed_dim))
        # [1, 2, d_c]
        if config.residual_factor:
            self.residual_coeff = nn.Parameter(torch.ones(1, config.main_dim))
            # [1, d_m]
        self.norm_eps = torch.tensor(config.norm_eps, device=device, dtype=self.dtype)

        if self.vanilla_matmul:
            self.vanilla_type1_w = nn.Parameter(
                torch.zeros(self.n_1m, config.main_dim, config.compressed_dim)
            )
            # [n_1m, d_m, d_c]
            self.vanilla_type2_w = nn.Parameter(
                torch.zeros(self.n_2m, config.compressed_dim, config.main_dim)
            )
            # [n_2m, d_c, d_m]
            self.vanilla_ffn_w = nn.Parameter(
                torch.zeros(3, n_fr, config.ffn_block_dim, config.ffn_vanilla_dim)
            )
            # [3, n_fr, d_f, d_fv]

        if not self.vanilla_matmul and config.fixed:
            self.fixed_in_w1 = nn.Parameter(
                torch.zeros(config.main_dim, config.linear_fixed_in_dim)
            )
            # [d_m, d_lfi], d_lf = linear_fixed_in_dim
            self.fixed_in_w2 = nn.Parameter(
                torch.zeros(config.linear_fixed_in_dim, 5 * config.main_dim)
            )
            # [d_lfi, 5 * d_m]
            self.fixed_out_w1 = nn.Parameter(
                torch.zeros(2 * config.main_dim, config.linear_fixed_out_dim)
            )
            # [2 * d_m, d_lfo]
            self.fixed_out_w2 = nn.Parameter(
                torch.zeros(config.linear_fixed_out_dim, config.main_dim)
            )
            # [d_lfo, d_m]
            self.fixed_ffn_w = nn.Parameter(
                torch.zeros(3, n_fr, config.ffn_block_dim, config.ffn_fixed_dim)
            )
            # [3, n_fr, d_f, d_ff]

    def layer_router_info(self, router_results, sublayer):
        indices, probs = router_results[sublayer]
        # [s * n_m * n_mi * n_1mh], [s, n_m, n_mi, n_1mh, 1]
        if self.config.global_linear_pool:
            param_sublayer = "linear"
        else:
            param_sublayer = sublayer
        weight = self.block_pool.weight_pool[param_sublayer]
        bias = self.block_pool.bias_pool[param_sublayer] if self.config.bias else None
        block_num = weight.shape[0]
        num_output = probs.shape[-2]

        return indices, probs, weight, bias, block_num, num_output

    def forward(
        self,
        x: torch.Tensor,  # [s, d_m]
        attn_c: Optional[torch.Tensor],  # [s, d_c]
        router_results: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        block_mask: BlockMask,
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[Tuple], torch.Tensor]:
        norm = (
            torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=self.dtype)
            + self.norm_eps
        )
        # [s, 1]
        x_norm = (x / norm) * self.elementwise_main_coeff.to(self.dtype)
        # [s, d_m]
        external_factors = self.external.external_factors.to(self.dtype)

        if not self.vanilla_matmul:
            indices_1, probs_1, weight_1, bias_1, block_num_1, num_output_1 = (
                self.layer_router_info(router_results, "linear_1")
            )
            indices_2, probs_2, weight_2, bias_2, block_num_2, num_output_2 = (
                self.layer_router_info(router_results, "linear_2")
            )
            (
                indices_ffn,
                probs_ffn,
                weight_ffn,
                bias_ffn,
                block_num_ffn,
                num_output_ffn,
            ) = self.layer_router_info(router_results, "ffn")
            indices_3, probs_3, weight_3, bias_3, block_num_3, num_output_3 = (
                self.layer_router_info(router_results, "linear_3")
            )
            indices_4, probs_4, weight_4, bias_4, block_num_4, num_output_4 = (
                self.layer_router_info(router_results, "linear_4")
            )

            block_num_cumsum = torch.cumsum(
                torch.tensor([block_num_1, block_num_2, block_num_ffn, block_num_3]), 0
            )
            block_num_sum = block_num_cumsum[-1] + block_num_4
            indices_cat = torch.cat(
                (
                    indices_1,
                    indices_2 + block_num_cumsum[0],
                    indices_ffn + block_num_cumsum[1],
                    indices_3 + block_num_cumsum[2],
                    indices_4 + block_num_cumsum[3],
                ),
                0,
            )
            freqs = torch.bincount(indices_cat, minlength=block_num_sum)
            freqs_1, freqs_2, freqs_ffn, freqs_3, freqs_4 = torch.split(
                freqs,
                [block_num_1, block_num_2, block_num_ffn, block_num_3, block_num_4],
                0,
            )

            block_valid_num_1 = torch.sum(freqs_1 > 0).item()
            block_valid_num_2 = torch.sum(freqs_2 > 0).item()
            block_valid_num_ffn = torch.sum(freqs_ffn > 0).item()
            block_valid_num_3 = torch.sum(freqs_3 > 0).item()
            block_valid_num_4 = torch.sum(freqs_4 > 0).item()
            x_len = self.config.x_len

            indices_1_pre_results = indices_pre_func(freqs_1, block_valid_num_1, x_len)
            indices_2_pre_results = indices_pre_func(freqs_2, block_valid_num_2, x_len)
            indices_ffn_pre_results = indices_pre_func(
                freqs_ffn, block_valid_num_ffn, x_len
            )
            indices_3_pre_results = indices_pre_func(freqs_3, block_valid_num_3, x_len)
            indices_4_pre_results = indices_pre_func(freqs_4, block_valid_num_4, x_len)

            x_1_in = repeat(
                x_norm,
                "s (n_mi d_li) -> s 2 n_mi d_li",
                d_li=self.config.linear_block_in_dim,
            )
            # [s, 2, n_mi, d_li], n_mi = linear_main_block_in_num
            indices_1_results = indices_func(
                indices_1,
                block_num_1,
                block_valid_num_1,
                num_output_1,
                indices_1_pre_results,
            )
            x_1_out = grouped_gemm(
                x_1_in,
                weight_1,
                bias_1,
                probs_1,
                False,
                self.config.linear_fused,
                False,
                indices_1_results,
            )
            if self.config.linear_fused:
                # [s, 2, n_1mh, d_lo]
                x_1_out = x_1_out.flatten(2)
                # [s, 2, d_c]
            else:
                # [s, 2, n_mi, n_1mh, d_lo]
                x_1_out = x_1_out * probs_1
                x_1_out = torch.sum(x_1_out, 2, dtype=self.dtype).flatten(2)
                # [s, 2, d_c]

            x_2_attn_in = repeat(
                x_1_out[:, 0, :],
                "s (n_ci d_li) -> s 3 n_ci d_li",
                d_li=self.config.linear_block_in_dim,
            )
            # [s, 3, n_ci, d_li], n_ci = linear_c_block_in_num
            x_2_ffn_in = repeat(
                x_1_out[:, 1, :],
                "s (n_ci d_li) -> s 2 n_ci d_li",
                d_li=self.config.linear_block_in_dim,
            )
            # [s, 2, n_ci, d_li], n_ci = linear_c_block_in_num

            x_2_in = torch.cat((x_2_attn_in, x_2_ffn_in), 1)
            # [s, 5, n_ci, d_li]
            indices_2_results = indices_func(
                indices_2,
                block_num_2,
                block_valid_num_2,
                num_output_2,
                indices_2_pre_results,
            )
            x_2_out = grouped_gemm(
                x_2_in,
                weight_2,
                bias_2,
                probs_2,
                False,
                self.config.linear_fused,
                False,
                indices_2_results,
            )
            # [s, 5, n_ci, n_2mh, d_lo]
            if self.config.linear_fused:
                # [s, 5, n_2mh, d_lo]
                x_2_out = x_2_out.flatten(2)
                # [s, 5, d_m]
            else:
                # [s, 5, n_ci, n_2mh, d_lo]
                x_2_out = x_2_out * probs_2
                x_2_out = torch.sum(x_2_out, 2, dtype=self.dtype).flatten(2)
                # [s, 5, d_m]
            indices_ffn_results = indices_func(
                indices_ffn,
                block_num_ffn,
                block_valid_num_ffn,
                num_output_ffn,
                indices_ffn_pre_results,
            )
            indices_3_results = indices_func(
                indices_3,
                block_num_3,
                block_valid_num_3,
                num_output_3,
                indices_3_pre_results,
            )
            indices_4_results = indices_func(
                indices_4,
                block_num_4,
                block_valid_num_4,
                num_output_4,
                indices_4_pre_results,
            )
            external, residual_norm_coeff = self.external(
                x, attn_c, norm, external_factors
            )
            if self.config.fixed:
                x_fixed = torch.matmul(x_norm, self.fixed_in_w1)
                # [s, d_m] mm [d_m, d_lfi] = [s, d_lfi]
                x_fixed = torch.matmul(x_fixed, self.fixed_in_w2).reshape(
                    -1, 5, self.config.main_dim
                )
                # [s, d_lfi] mm [d_lfi, 5 * d_m] = [s, 5 * d_m] -> [s, 5, d_m]
                x_2_out += x_fixed
            x_attn_in = x_2_out[:, :3, :]
            # [s, 3, d_m]

            x_ffn = x_2_out[:, 3:, :]
            # [s, 2, d_m]
            x_ffn_in = rearrange(
                x_ffn,
                "s c (n_fr d_f) -> s 1 n_fr (c d_f)",
                d_f=self.config.ffn_block_dim,
            )
            # [s, 1, n_fr, 2 * d_f]
            x_attn_out = self.attention(x_attn_in, block_mask)
            # [s, 1, d_m]
            x_ffn_out = grouped_gemm(
                x_ffn_in,
                weight_ffn,
                bias_ffn,
                probs_ffn,
                True,
                self.config.ffn_fused,
                False,
                indices_ffn_results,
            )
            if self.config.ffn_fused:
                # [s, 1, n_fr, d_f]
                x_ffn_out = x_ffn_out.flatten(2)
                # [s, 1, d_m]
            else:
                # [s, 1, n_fr, k * n_iv, d_f]
                x_ffn_out = x_ffn_out * probs_ffn
                x_ffn_out = torch.sum(x_ffn_out, 3, dtype=self.dtype).flatten(2)
                # [s, 1, d_m]
            if self.config.fixed:
                x_ffn_in = rearrange(
                    x_ffn_in.squeeze(1), "s n_fr (c d_f) -> c n_fr s d_f", c=2
                )
                # [2, n_fr, s, d_f]
                x_ffn_fixed_g = torch.bmm(x_ffn_in[0], self.fixed_ffn_w[0])
                # [n_fr, s, d_f] bmm [n_fr, d_f, d_ff] = [n_fr, s, d_ff]
                x_ffn_fixed_l = torch.bmm(x_ffn_in[1], self.fixed_ffn_w[1])
                x_ffn_fixed = F.silu(x_ffn_fixed_g) * x_ffn_fixed_l
                x_ffn_fixed = torch.bmm(
                    x_ffn_fixed, self.fixed_ffn_w[2].transpose(1, 2)
                )
                # [n_fr, s, d_ff] bmm [n_fr, d_ff, d_f] = [n_fr, s, d_f]
                x_ffn_fixed = rearrange(x_ffn_fixed, "n_fr s d_f -> s 1 (n_fr d_f)")
                # [s, 1, d_m]
                x_ffn_out += x_ffn_fixed
            x_3_in = torch.cat((x_attn_out, x_ffn_out), 1)
            # [s, 2, d_m]

            x_3_in = rearrange(
                x_3_in,
                "s c (n_mi d_li) -> s c n_mi d_li",
                d_li=self.config.linear_block_in_dim,
            )
            # [s, 2, n_mi, d_li]
            x_3_out = grouped_gemm(
                x_3_in,
                weight_3,
                bias_3,
                probs_3,
                False,
                self.config.linear_fused,
                False,
                indices_3_results,
            )
            if self.config.linear_fused:
                # [s, 2, n_1mh, d_lo]
                x_3_out = x_3_out.flatten(2)
                # [s, 2, d_c]
            else:
                # [s, 2, n_mi, n_1mh, d_lo]
                x_3_out = x_3_out * probs_3
                x_3_out = torch.sum(x_3_out, 2, dtype=self.dtype).flatten(2)
                # [s, 2, d_c]
            next_attn_c = x_3_out[:, 0, :]
            # [s, d_c]
            x_3_out = x_3_out * self.elementwise_c_coeff.to(self.dtype)
            # [s, 2, d_c]

            x_4_in = torch.sum(x_3_out, 1, keepdim=True, dtype=self.dtype).reshape(
                self.config.seq_len, 1, -1, self.config.linear_block_in_dim
            )
            # [s, 1, n_ci, d_li]
            x_4_out = grouped_gemm(
                x_4_in,
                weight_4,
                bias_4,
                probs_4,
                False,
                self.config.linear_fused,
                False,
                indices_4_results,
            )
            # [s, 1, n_ci, n_2mh, d_lo]
            if self.config.linear_fused:
                # [s, 1, n_2mh, d_lo]
                x_4_out = x_4_out.flatten(2)
                # [s, 1, d_m]
            else:
                x_4_out = x_4_out * probs_4
                x_4_out = torch.sum(x_4_out, 2, dtype=self.dtype).flatten(2)
                # [s, 1, d_m]
            x_4_out = x_4_out.squeeze(1)
            # [s, d_m]
            if self.config.fixed:
                x_fixed = torch.matmul(x_3_in.flatten(1), self.fixed_out_w1)
                # [s, 2 * d_m] mm [2 * d_m, d_lfo] = [s, d_lfo]
                x_fixed = torch.matmul(x_fixed, self.fixed_out_w2)
                # [s, d_lfo] mm [d_lfo, d_m] = [s, d_m]
                x_4_out += x_fixed

        else:
            external, residual_norm_coeff = self.external(
                x, attn_c, norm, external_factors
            )

            x_1_in = repeat(x_norm, "s d_m -> 2 s d_m")
            # [2, s, d_m]
            x_1_out = torch.bmm(x_1_in, self.vanilla_type1_w[:2])
            # [2, s, d_m] bmm [2, d_m, d_c] = [2, s, d_c]
            x_2_attn_in = repeat(x_1_out[0], "s d_c -> 3 s d_c")
            # [3, s, d_c]
            x_2_ffn_in = repeat(x_1_out[1], "s d_c -> 2 s d_c")
            # [2, s, d_c]

            x_2_in = torch.cat((x_2_attn_in, x_2_ffn_in), 0)
            # [5, s, d_c]
            x_2_out = torch.bmm(x_2_in, self.vanilla_type2_w[:5])
            # [5, s, d_c] bmm [5, d_c, d_m] = [5, s, d_m]
            x_attn_in = x_2_out[:3]
            # [3, s, d_m]
            x_attn_out = self.attention(
                x_attn_in.transpose(0, 1), block_mask
            ).transpose(0, 1)
            # [1, s, d_m]

            x_ffn = x_2_out[3:].transpose(0, 1)
            # [s, 2, d_m]
            x_ffn_in = rearrange(
                x_2_out[3:],
                "c s (n_fr d_f) -> c n_fr s d_f",
                d_f=self.config.ffn_block_dim,
            )
            # [2, n_fr, s, d_f]
            x_ffn_g = torch.bmm(x_ffn_in[0], self.vanilla_ffn_w[0])
            # [n_fr, s, d_f] bmm [n_fr, d_f, d_fv] = [n_fr, s, d_fv]
            x_ffn_l = torch.bmm(x_ffn_in[1], self.vanilla_ffn_w[1])
            x_ffn_out = F.silu(x_ffn_g) * x_ffn_l
            x_ffn_out = torch.bmm(x_ffn_out, self.vanilla_ffn_w[2].transpose(1, 2))
            # [n_fr, s, d_fv] bmm [n_fr, d_fv, d_f] = [n_fr, s, d_f]
            x_ffn_out = rearrange(x_ffn_out, "n_fr s d_f -> 1 s (n_fr d_f)")
            # [1, s, d_m]

            x_3_in = torch.cat((x_attn_out, x_ffn_out), 0)
            # [2, s, d_m]
            x_3_out = torch.bmm(x_3_in, self.vanilla_type1_w[2:])
            # [2, s, d_m] bmm [2, d_m, d_c] = [2, s, d_c]
            next_attn_c = x_3_out[0]
            # [s, d_c]
            x_3_out = x_3_out * self.elementwise_c_coeff.transpose(0, 1).to(self.dtype)
            # [2, s, d_c]

            x_4_in = torch.sum(x_3_out, 0, keepdim=True)
            # [1, s, d_c]
            x_4_out = torch.matmul(x_4_in.squeeze(0), self.vanilla_type2_w[-1])
            # [s, d_c] mm [d_c, d_m] = [s, d_m]

        if not self.config.residual_factor and not self.config.residual_norm:
            x_output = x_4_out + x
        if self.config.residual_factor and not self.config.residual_norm:
            x_output = x_4_out + x * self.residual_coeff.to(self.dtype)
        if not self.config.residual_factor and self.config.residual_norm:
            x_output = x_4_out + x * residual_norm_coeff
        if self.config.residual_factor and self.config.residual_norm:
            x_output = (
                x_4_out + x * self.residual_coeff.to(self.dtype) * residual_norm_coeff
            )
        # [s, d_m]

        if not self.vanilla_matmul:
            x_linear_type1 = torch.cat((x_1_in, x_3_in), 1).flatten(2)
            # [s, n_1m, d_m]
            x_linear_type2 = torch.cat((x_2_in, x_4_in), 1).flatten(2)
            # [s, n_2m, d_c]
        else:
            x_linear_type1 = torch.cat((x_1_in, x_3_in), 0).transpose(0, 1)
            # [s, n_1m, d_m]
            x_linear_type2 = torch.cat((x_2_in, x_4_in), 0).transpose(0, 1)
            # [s, n_2m, d_c]

        next_router_results, aux_loss = None, None
        if self.have_router:
            query_linear, query_ffn = self.query(
                x_linear_type1, x_linear_type2, x_ffn, external
            )
            # [n_lm, s * n_lr, v], [s * n_fr, v]
            next_router_results, aux_loss = self.router(query_linear, query_ffn)
            linear_indices, linear_probs, ffn_indices, ffn_probs = next_router_results
            # [s, n_lm, n_lr, n_iv] * 2

            linear_type1_indices = linear_indices[:, : self.n_1m, ...]
            # [s, n_1m, n_lr, n_iv]
            linear_type2_indices = linear_indices[:, self.n_1m :, ...]
            # [s, n_2m, n_lr, n_iv]
            linear_type1_probs = linear_probs[:, : self.n_1m, ...]
            # [s, n_1m, n_lr, n_iv]
            linear_type2_probs = linear_probs[:, self.n_1m :, ...]
            # [s, n_2m, n_lr, n_iv]

            linear_type1_indices = rearrange(
                linear_type1_indices,
                "s n_1m (n_mr n_1mh) n_iv -> s n_1m (n_mr n_iv) n_1mh",
                n_1mh=self.n_1mh,
            )
            # [s, n_1m, n_mi, n_1mh]
            linear_type2_indices = rearrange(
                linear_type2_indices,
                "s n_2m (n_cr n_2mh) n_iv -> s n_2m (n_cr n_iv) n_2mh",
                n_2mh=self.n_2mh,
            )
            # [s, n_2m, n_ci, n_2mh]
            linear_type1_probs = rearrange(
                linear_type1_probs,
                "s n_2m (n_mr n_1mh) n_iv -> s n_2m (n_mr n_iv) n_1mh 1",
                n_1mh=self.n_1mh,
            )
            # [s, n_1m, n_mi, n_1mh, 1]
            linear_type2_probs = rearrange(
                linear_type2_probs,
                "s n_2m (n_cr n_2mh) n_iv -> s n_2m (n_cr n_iv) n_2mh 1",
                n_2mh=self.n_2mh,
            )
            # [s, n_2m, n_ci, n_2mh, 1]
            if self.config.linear_router_prob_norm:
                linear_type1_probs = linear_type1_probs / torch.sum(
                    linear_type1_probs, 2, keepdim=True, dtype=self.dtype
                )
                linear_type2_probs = linear_type2_probs / torch.sum(
                    linear_type2_probs, 2, keepdim=True, dtype=self.dtype
                )

            indices_1 = linear_type1_indices[:, :2, ...].flatten()
            # [s * 2 * n_mi * n_1mh]
            indices_2 = linear_type2_indices[:, :5, ...].flatten()
            # [s * 5 * n_ci * n_2mh]
            indices_3 = linear_type1_indices[:, 2:, ...].flatten()
            # [s * 2 * n_mi * n_1mh]
            indices_4 = linear_type2_indices[:, [-1], ...].flatten()
            # [s * 1 * n_ci * n_2mh]

            probs_1 = linear_type1_probs[:, :2, ...]
            # [s, 2, n_mi, n_1mh, 1]
            probs_2 = linear_type2_probs[:, :5, ...]
            # [s, 5, n_ci, n_2mh, 1]
            probs_3 = linear_type1_probs[:, 2:, ...]
            # [s, 2, n_mi, n_1mh, 1]
            probs_4 = linear_type2_probs[:, [-1], ...]
            # [s, 1, n_ci, n_2mh, 1]

            indices_ffn = ffn_indices.flatten()
            # [s, 1, n_fr, k * n_iv]
            probs_ffn = ffn_probs.flatten(3).unsqueeze(-1)
            # [s, 1, n_fr, k * n_iv, 1]
            if self.config.ffn_router_prob_norm:
                probs_ffn = (probs_ffn / torch.sum(probs_ffn, 3, keepdim=True)).to(
                    self.dtype
                )

            next_router_results = {
                "linear_1": (indices_1, probs_1),
                "linear_2": (indices_2, probs_2),
                "linear_3": (indices_3, probs_3),
                "linear_4": (indices_4, probs_4),
                "ffn": (indices_ffn, probs_ffn),
            }

        return x_output, next_router_results, aux_loss, next_attn_c


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.block_pool = ParamBlockPool(config)
        self.vector_pool = ParamVectorPool(config)
        self.rotary_emb = RotaryEmb(config)
        self.attention = Attention(config, self.rotary_emb)
        self.model = nn.ModuleDict(
            {
                "vocab_emb": nn.Embedding(config.vocab_size, config.vocab_c_dim),
                # [vocab, vocab_c] or [vocab, d_m]
                "vocab_c_in": nn.Linear(config.vocab_c_dim, config.main_dim, bias=False)
                if config.vocab_compressed
                else nn.Identity(),
                # [vocab_c, d_m]
                "layers": nn.ModuleList(
                    [
                        Layer(
                            config,
                            i,
                            self.block_pool,
                            self.vector_pool,
                            self.attention,
                        )
                        for i in range(config.layer_num)
                    ]
                ),
                "vocab_c_out": nn.Linear(
                    config.main_dim, config.vocab_c_dim, bias=False
                )
                if config.vocab_compressed
                else nn.Identity(),
                # [d_m, vocab_c]
                "lm_head": nn.Linear(config.vocab_c_dim, config.vocab_size, bias=False),
                # [vocab_c, vocab] or [d_m, vocab]
            }
        )
        if config.tied_vocab_emb:
            self.model.vocab_emb.weight = self.model.lm_head.weight
        self.generator = torch.Generator()
        self.generator.manual_seed(42)
        self.init_weights()

    def init_weights(self):
        std = self.config.init_std

        for name, param in self.block_pool.named_parameters():
            if "weight_pool" in name:
                nn.init.trunc_normal_(
                    param,
                    mean=0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                    generator=self.generator,
                )

            if "bias_pool" in name:
                nn.init.zeros_(param)

        for name, param in self.vector_pool.named_parameters():
            if name[-3:] == "vec" or "vec_clusters" in name:
                nn.init.trunc_normal_(
                    param,
                    mean=0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                    generator=self.generator,
                )
            if "gain_factors" in name:
                nn.init.zeros_(param)

        for name, param in self.model.named_parameters():
            if name[-1] == "b" in name or "external_factors" in name:
                nn.init.zeros_(param)
            elif "residual_coeff" in name or "elementwise" in name:
                nn.init.ones_(param)
            else:
                nn.init.trunc_normal_(
                    param,
                    mean=0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                    generator=self.generator,
                )

    def forward(
        self,
        idx: torch.LongTensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = f"cuda:{torch.cuda.current_device()}"
        docs = (idx == 50256).cumsum(0)

        def doc_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < self.config.window_len
            return causal_mask & doc_mask & window_mask

        seq_len = self.config.seq_len
        block_mask = create_block_mask(
            doc_causal_mask, None, None, seq_len, seq_len, device=device, _compile=True
        )
        vocab_emb = self.model.vocab_emb(idx)
        # [s] -> [s, vocab_c] or [s, d_m]
        if self.config.vocab_compressed:
            vocab_emb = self.model.vocab_c_in(vocab_emb)
            # [s, vocab_c] -> [s, d_m]
        x, attn_c, router_results = vocab_emb, None, None

        aux_loss_list = []
        for i, layer in enumerate(self.model.layers):
            x, router_results, aux_loss, attn_c = layer(
                x, attn_c, router_results, block_mask
            )
            if aux_loss is not None:
                aux_loss_list.append(aux_loss)
        x = F.normalize(x, dim=-1, eps=self.config.norm_eps)
        # [s, d_m]
        if self.config.vocab_compressed:
            x = self.model.vocab_c_out(x)
            # [s, vocab_c]
        logits = self.model.lm_head(x)
        # [s, vocab]
        loss = F.cross_entropy(logits, targets)

        layer_num = len(aux_loss_list)
        c_aux_loss = (
            torch.sum(sum(aux_loss_list[i][0] for i in range(layer_num)))
            * self.config.c_token_balance_factor
        )
        vec_aux_loss = (
            torch.sum(sum(aux_loss_list[i][1] for i in range(layer_num)))
            * self.config.vec_token_balance_factor
        )

        total_loss = loss + c_aux_loss + vec_aux_loss

        return loss.detach(), total_loss, c_aux_loss.detach(), vec_aux_loss.detach()

    def params_count(self) -> Dict[str, int]:
        block_pool_params = sum(x.numel() for x in self.block_pool.parameters())
        vector_pool_params = sum(x.numel() for x in self.vector_pool.parameters())
        model_params = sum(
            param.numel()
            for name, param in self.model.named_parameters()
            if not "pool" in name
        )
        emb_params = int(
            self.model.vocab_emb.weight.numel() * (2 - self.config.tied_vocab_emb)
        )
        total_params = block_pool_params + vector_pool_params + model_params
        total_params_wo_emb = total_params - emb_params
        model_params_wo_emb = model_params - emb_params

        result = {
            "total_params": total_params,
            "total_params_wo_emb": total_params_wo_emb,
            "block_pool_params": block_pool_params,
            "vector_pool_params": vector_pool_params,
            "model_params": model_params,
            "model_params_wo_emb": model_params_wo_emb,
            "emb_params": emb_params,
        }

        return result

    def activated_params_count(self) -> Dict[str, int]:
        index_num_per_vec = 2 if self.config.paired else 1
        main_router_block_num = int(
            self.config.main_dim / (self.config.linear_block_in_dim * index_num_per_vec)
        )
        # n_mr
        linear_type1_multihead_num = int(
            self.config.compressed_dim / self.config.linear_block_out_dim
        )
        # n_1mh
        linear_module_num = len(self.config.linear_module_list)
        linear_router_num = (
            main_router_block_num * linear_type1_multihead_num * linear_module_num
        )
        ffn_router_num = (
            int(self.config.main_dim / self.config.ffn_block_dim)
            * self.config.ffn_top_k
        )
        layer_num = (
            self.config.layer_num - 2
            if self.config.last_layer_vanilla
            else self.config.layer_num - 1
        )
        linear_router_block_params = (
            self.config.linear_cluster_num + self.config.c_vec_num
        ) * self.config.vec_dim
        ffn_router_block_params = (
            self.config.ffn_cluster_num + self.config.c_vec_num
        ) * self.config.vec_dim
        linear_block_params = (
            self.config.linear_block_in_dim + self.config.bias
        ) * self.config.linear_block_out_dim
        ffn_block_params = (
            3 * self.config.ffn_block_dim * self.config.ffn_hidden_dim
            + self.config.bias * self.config.ffn_block_dim
        )

        model_params = model_params = sum(
            param.numel()
            for name, param in self.model.named_parameters()
            if not "pool" in name
        )
        emb_params = int(
            self.model.vocab_emb.weight.numel() * (2 - self.config.tied_vocab_emb)
        )
        linear_router_params = (
            linear_router_num * linear_router_block_params * layer_num
        )
        ffn_router_params = ffn_router_num * ffn_router_block_params * layer_num
        linear_main_params = (
            linear_router_num * linear_block_params * layer_num * index_num_per_vec
        )
        ffn_main_params = (
            ffn_router_num * ffn_block_params * layer_num * index_num_per_vec
        )

        total_params = (
            model_params
            + linear_router_params
            + ffn_router_params
            + linear_main_params
            + ffn_main_params
        )
        total_params_wo_emb = total_params - emb_params
        model_params_wo_emb = model_params - emb_params

        result = {
            "total_params": total_params,
            "total_params_wo_emb": total_params_wo_emb,
            "ffn_router_params": ffn_router_params,
            "linear_router_params": linear_router_params,
            "ffn_main_params": ffn_main_params,
            "linear_main_params": linear_main_params,
            "model_params": model_params,
            "model_params_wo_emb": model_params_wo_emb,
            "emb_params": emb_params,
        }

        return result


def configure_optimizers(model: Model, config: Config):
    param_dict = {name: param for name, param in model.named_parameters()}
    model_lr = config.model_lr
    pool_lr = config.pool_lr
    weight_decay = config.weight_decay
    betas = config.betas

    decay_model = [
        param
        for name, param in param_dict.items()
        if ("model" in name and param.shape[0] != 1)
    ]
    no_decay_model = [
        param
        for name, param in param_dict.items()
        if ("model" in name and param.shape[0] == 1)
    ]
    decay_pool = [param for name, param in param_dict.items() if "weight_pool" in name]
    no_decay_pool = [
        param
        for name, param in param_dict.items()
        if "vector_pool" in name or "bias_pool" in name
    ]

    if config.no_weight_decay_1d:
        optim_groups = [
            {"params": decay_model, "lr": model_lr, "weight_decay": weight_decay},
            {"params": no_decay_model, "lr": model_lr, "weight_decay": 0.0},
            {"params": decay_pool, "lr": pool_lr, "weight_decay": weight_decay},
            {"params": no_decay_pool, "lr": pool_lr, "weight_decay": 0.0},
        ]
    else:
        optim_groups = [
            {
                "params": decay_model + no_decay_model,
                "lr": model_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": decay_pool + no_decay_pool,
                "lr": pool_lr,
                "weight_decay": weight_decay,
            },
        ]

    if config.zero_optimizer:
        optimizer = ZeroRedundancyOptimizer(
            optim_groups, optimizer_class=torch.optim.AdamW, betas=betas, foreach=True
        )
    else:
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, foreach=True)

    return optimizer


def load_data_shard(filename: str, peek: bool = False):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            print0("magic number mismatch in the data .bin file", log_mode=False)
            exit(1)
        assert header[1] == 1, "unsupported version"
        num_token = header[2]
        if peek:
            return num_token
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(tokens) == num_token, "number of tokens read does not match header"
        return tokens


class DistributedDataLoader:
    def __init__(
        self, filename_pattern: str, seq_len: int, process_rank: int, num_processes: int
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_len = seq_len
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        num_token_total = 0
        for fname in self.files:
            shard_num_token = load_data_shard(fname, peek=True)
            assert shard_num_token >= num_processes * seq_len + 1
            num_token_total += int(shard_num_token)
        self.num_token_total = num_token_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_pos = self.process_rank * self.seq_len
        self.tokens = load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        total_seq_len = self.seq_len * self.num_processes
        buffer = self.tokens[self.current_pos : self.current_pos + self.seq_len + 1]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        x = buffer[:-1]
        y = buffer[1:]
        self.current_pos += total_seq_len

        if self.current_pos + total_seq_len >= len(self.tokens):
            self.advance()

        return x, y


config = Config()

if config.flex_attention_compile:
    flex_attention = torch.compile(flex_attention, dynamic=False)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_bin",
    type=str,
    default="/root/autodl-tmp/llm.c/dev/data/fineweb10B/fineweb_train_*.bin",
)
parser.add_argument(
    "--input_val_bin",
    type=str,
    default="/root/autodl-tmp/llm.c/dev/data/fineweb10B/fineweb_val_*.bin",
)
parser.add_argument("--accumulated_steps", type=int, default=1)
parser.add_argument("--scheduler", type=str, default="wsd")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--cooldown_steps", type=int, default=100)
parser.add_argument("--val_every", type=int, default=25)
parser.add_argument("--val_steps", type=int, default=20)
parser.add_argument("--overfit", type=bool, default=False)
parser.add_argument("--save_every", type=int, default=0)
parser.add_argument("--save_before_cooldown", type=bool, default=True)
parser.add_argument("--log_freq", type=int, default=1)
parser.add_argument("--gpu", type=str, default="4090")

args = parser.parse_args()
for arg, value in vars(args).items():
    setattr(config, arg, value)

assert torch.cuda.is_available()
ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    config.zero_optimizer = False
    device = f"cuda:{torch.cuda.current_device()}"
    master_process = True


def print0(msg, print_mode: bool = True, log_mode: bool = True):
    if master_process:
        if print_mode:
            print(msg)
        if log_mode:
            with open(log_file, "a") as f:
                f.write(msg + "\n")


with open(sys.argv[0]) as f:
    code = f.read()

run_id = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
if master_process:
    log_dir = f"logs/{run_id}/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"logs/{run_id}.txt"

    with open(log_file, "w") as f:
        f.write(code)
        f.write("=" * 100 + "\n")

print0(
    f"running pytorch {torch.version.__version__}"
    f"compiled for cuda {torch.version.cuda}\nnvidia-smi:"
)
result = subprocess.run(
    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
print0(f"{result.stdout}")
print0("=" * 100)

train_loader = DistributedDataLoader(
    config.input_bin, config.seq_len, ddp_rank, ddp_world_size
)
val_loader = DistributedDataLoader(
    config.input_val_bin, config.seq_len, ddp_rank, ddp_world_size
)

step_tokens = ddp_world_size * config.seq_len
accumulated_step_tokens = step_tokens * config.accumulated_steps
assert (
    val_loader.num_token_total >= config.val_steps * step_tokens
), "validation tokens not enough"

print0(
    f"step_tokens: {step_tokens} \
    accumulated_step_tokens: {accumulated_step_tokens}"
)
print0(
    f"training dataloader: total number of tokens: \
    {train_loader.num_token_total} across {len(train_loader.files)} files"
)
print0(
    f"validation dataloader: total number of tokens: \
    {val_loader.num_token_total} across {len(val_loader.files)} files"
)
print0("=" * 100)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = Model(config).to(device)
model.train()
amp_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
scaler = torch.amp.GradScaler()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if config.model_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

optimizer = configure_optimizers(raw_model, config)
original_lr = []
for param_group in optimizer.param_groups:
    original_lr.append(param_group["lr"])

num_steps = config.num_steps
warmup_steps = config.warmup_steps
cooldown_steps = config.cooldown_steps
min_lr_ratio = config.min_lr_ratio


def get_lr_wsd(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step < num_steps - cooldown_steps:
        return 1.0
    else:
        decay_ratio = (num_steps - step) / cooldown_steps
        return max(decay_ratio, min_lr_ratio)


def get_lr_cos(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step < num_steps:
        decay_ratio = (step - warmup_steps) / (num_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)
    else:
        return min_lr_ratio


if config.scheduler == "wsd":
    get_lr = get_lr_wsd
if config.scheduler == "cos":
    get_lr = get_lr_cos

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

total_time_ms = 0.0
time_ms = 0.0
running_avg_time = 0.0
tokens_per_second = 0.0
running_avg_tokens = 0.0
grad_norm = 0.0

train_log = []
train_log_file = f"logs/{run_id}/train_log.csv"
train_layer_log = []
train_layer_log_file = f"logs/{run_id}/train_layer_log.csv"

t0 = time.time()

for step in range(num_steps + 1):
    if step == 0:
        raw_model.vector_pool.vec_init()

    last_step = step == num_steps
    if last_step or (
        config.val_every > 0 and step > 0 and step % config.val_every == 0
    ):
        torch.cuda.synchronize()
        model.eval()
        val_loader.reset()
        val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            with amp_ctx:
                for _ in range(config.val_steps):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    model_loss, _, _, _ = model(x_val, y_val)
                    val_loss += model_loss
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= config.val_steps

        print0(f"step:{step}/{num_steps} val_loss:{val_loss.item():.4f}")

        model.train()
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (
        last_step
        or (config.save_before_cooldown and step == cooldown_steps)
        or (config.save_every > 0 and step % config.save_every == 0)
    ):
        torch.cuda.synchronize()

        checkpoint = {
            "model": raw_model,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"logs/{run_id}/checkpoint_step_{step}.pt")
        # TODO: checkpoint

        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    if config.overfit:
        train_loader.reset()

    if step == 4:
        torch.cuda.memory._record_memory_history()

    profile_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if step == 3
        else nullcontext()
    )

    train_loss = torch.zeros(1, device=device)
    aux_loss = torch.zeros(2, device=device)
    c_aux_loss = torch.zeros(1, device=device)
    vec_aux_loss = torch.zeros(1, device=device)
    c_vec_aux_loss = None
    with profile_ctx as prof:
        for micro_step in range(config.accumulated_steps):
            t_0 = time.time()
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < config.accumulated_steps - 1
                else nullcontext()
            )

            with sync_ctx:
                with amp_ctx:
                    step_train_loss, loss, step_c_aux_loss, step_vec_aux_loss = model(x, y)
                    if micro_step == 0 and c_vec_aux_loss is not None:
                        loss = loss / config.accumulated_steps + c_vec_aux_loss
                    else:
                        loss = loss / config.accumulated_steps
                    train_loss += step_train_loss / config.accumulated_steps
                    c_aux_loss += step_c_aux_loss / config.accumulated_steps
                    vec_aux_loss += step_vec_aux_loss / config.accumulated_steps
                    torch.cuda.empty_cache()
                    if config.scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

        lr = get_lr(step)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr * original_lr[i]

        if config.scaler:
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.grad_clip != 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(c_aux_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(vec_aux_loss, op=dist.ReduceOp.AVG)

        c_vec_aux_loss, c_mean_prob, cand_p = raw_model.vector_pool.training_update()

    if step == 3:
        prof.export_chrome_trace(f"logs/{run_id}/trace.json")

    if step == 4:
        torch.cuda.memory._dump_snapshot(f"logs/{run_id}/snapshot.pickle")
        # TODO: multi step

    torch.cuda.synchronize()

    if master_process:
        time_ms = 1000 * (time.time() - t0)
        running_avg_time = (
            time_ms if step < 10 else 0.8 * running_avg_time + 0.2 * time_ms
        )
        total_time_ms += time_ms
        tokens_per_second = accumulated_step_tokens / time_ms * 1000
        running_avg_tokens = (
            tokens_per_second
            if step < 10
            else 0.8 * running_avg_tokens + 0.2 * tokens_per_second
        )

        residual_coeff_list = []
        residual_norm_coeff_list = []
        layers = raw_model.model["layers"]

        for i in range(config.layer_num):
            if config.residual_factor:
                residual_coeff_list.append(layers[i].residual_coeff.mean().item())
            if config.residual_norm:
                residual_norm_coeff_list.append(
                    layers[i].external.external_factors[0, -1].item()
                )
        residual_coeff_avg = sum(residual_coeff_list) / len(residual_coeff_list)
        residual_norm_coeff_avg = sum(residual_norm_coeff_list) / len(
            residual_norm_coeff_list
        )

        grad_norm = grad_norm.item()
        c_aux_loss = c_aux_loss.item()
        vec_aux_loss = vec_aux_loss.item()

        print0(
            f"step:{step}/{num_steps} "
            f"train_loss:{train_loss.item():.4f} "
            f"time:{total_time_ms / 1000:.0f}s "
            # f"step_time:{time_ms:.1f}ms "
            f"running_avg_time:{running_avg_time:.1f}ms "
            # f"speed: {tokens_per_second:.0f} t/s  "
            f"avg_speed: {running_avg_tokens:.0f} t/s  "
            f"norm:{grad_norm:.3f} "
            f"aux_loss:({c_aux_loss:.3f}, {vec_aux_loss:.3f}, {c_vec_aux_loss:.3f}) "
            f"vec_info: ({cand_p:.3f}, {c_mean_prob:.3f}) "
            f"coeff: ({residual_coeff_avg:.3f}, {residual_norm_coeff_avg:.3f})"
        )

        train_log.append(
            [
                step,
                train_loss.item(),
                total_time_ms / 1000,
                running_avg_time,
                running_avg_tokens,
                grad_norm,
                c_aux_loss,
                vec_aux_loss,
                c_vec_aux_loss,
                cand_p,
                c_mean_prob,
                residual_coeff_avg,
                residual_norm_coeff_avg,
            ]
        )

        train_layer_log.append(residual_coeff_list + [0] + residual_norm_coeff_list)

        if step % config.log_freq == 0:
            df = pd.DataFrame(
                train_log,
                columns=[
                    "step",
                    "train_loss",
                    "total_time(s)",
                    "step_time(ms)",
                    "token_speed(t/s)",
                    "grad_norm",
                    "c_aux_loss",
                    "vec_aux_loss",
                    "c_vec_aux_loss",
                    "cand_p",
                    "c_mean_prob",
                    "residual_coeff_avg",
                    "residual_norm_coeff_avg",
                ],
            )

            df_layer = pd.DataFrame(train_layer_log)

            if step == 0:
                df.to_csv(train_log_file, index=False)
            else:
                df.to_csv(train_log_file, mode="a", header=False, index=False)

            df_layer.to_csv(train_layer_log_file, mode="a", header=False, index=False)

            train_log = []
            train_layer_log = []

    torch.cuda.synchronize()
    t0 = time.time()

if ddp:
    dist.destroy_process_group()
