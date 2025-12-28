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
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from torch.profiler import profile, record_function, ProfilerActivity
from einops import rearrange, repeat, reduce
from grouped_gemm.ops import gmm, permute, unpermute # type: ignore
from fast_pytorch_kmeans import KMeans # type: ignore

class Config:
    
    layer_num: int = 8
    block_dim: int = 128
    vocab_size: int = 50304
    vocab_c_dim: int = 768
    seq_len: int = 64 * 1024
    ffn_vec_dim: int = 32
    linear_vec_dim: int = 16
    sublayer_vec_dim: int = 3
    external_c_dim: int = 32
    linear_fixed_dim: int = 32
    ffn_fixed_dim: int = 32
    ffn_fixed_vanilla_matmul_dim: int = 64
    ffn_hidden_dim: int = 128
    ffn_top_k: int = 3
    init_std: float = 0.02
    theta: float = 10000.0
    pool_lr: float = 1e-4
    model_lr: float = 5e-4
    min_lr_ratio: float = 0.1
    scheduler: str = 'wsd' # 'wsd' or 'cos'
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    norm_eps: float = 1e-6
    adam_eps: float = 1e-8
    window_len: int = 4096
    linear_vec_factor: float = 1.5e3
    ffn_vec_factor: float = 4e3
    capacity_factor: float = 1.2
    training_update_freq: int = 1
    update_cutoff_point: float = 0.9
    balance_factor: float = 1e-2
    router_z_factor: float = 1e-3
    vec_balance_factor: float = 1e-2
    vec_router_z_factor: float = 1e-3

    global_linear_pool: bool = False
    no_compressed: bool = False #
    striped_block: bool = False #
    no_paired: bool = False
    three_layer_router: bool = False #
    two_layer_router: bool = True
    pre_gated: bool = True #
    bias: bool = True
    tied_vocab_emb: bool = True
    vocab_compressed: bool = True
    last_layer_vanilla: bool = False
    info_fusion_router: bool = True
    rnn_router: bool = False #
    retrieval_router: bool = False #
    vanilla_router: bool = False #
    residual_norm: bool = True
    residual_factor: bool = True
    rnn_residual_norm: bool = False #
    gqa: bool = True
    qk_norm: bool = False
    return_logits: bool = False
    zero_optimizer: bool = True
    no_weight_decay_1d: bool = True
    model_compile: bool = True
    scaler: bool = True
    multi_stream: bool = True
    fixed: bool = True
    clear_gather_list: bool = True
    nonzero_prob_norm: bool = False #
    router_detach: bool = True
    fsdp: bool = False #


    other_compile: Dict[str, bool] = {
        'flex_attention': True,
        'gmm': False,
        'permute': False,
        'unpermute': False,
    }

    block_num: Dict[str, int] = {
        'linear_1': 200,
        'linear_2': 300,
        'linear_3': 150,
        'linear_4': 100,
    }
    global_linear_block_num: int = 600
    ffn_block_num: int = 600

    main_size: int = 8 # in units of block_dim, same below
    attn_size: int = 1
    
    q_c_size: int = 2
    kv_c_size: int = 2
    ffn_c_size: int = 3
    q_size: int = 8
    kv_size: int = 4
    ffn_size: int = 10
    oc_size: int = 4

    q_f_size: int = 1
    kv_f_size: int = 1
    ffn_f_size: int = 1
    ffn_of_size: int = 1
    v_of_size: int = 1

    module_list: List[str] = ['q_c', 'kv_c', 'ffn_c', 'q', 'k', 'v', 'ffn_g', 'ffn_l', 'ffn', 'v_oc', 'ffn_oc', 'output']

    module_parent: List[str] = ['linear_1'] * 3 + ['linear_2'] * 5 + ['ffn'] + ['linear_3'] * 2 + ['linear_4']

    input_repeat_num: Dict[str, Dict[str, int]] = {
        'linear_1':{
            'input':3,
        },
        'linear_2':{
            'q_c':1,
            'kv_c':2,
            'ffn_c':2,
        },
        'linear_3':{
            'v_o':1,
            'ffn':1,
        },
        'linear_4':{
            'output_c':1,
        },
    }

    sublayer_fixed_type: Dict[str, int] = {
        'linear_1':1,
        'linear_2':0,
        'linear_3':1,
        'linear_4':0,
    }

    modules = {
        'linear_1':{
            'q_c':(main_size, q_c_size, q_f_size, 'input'),
            'kv_c':(main_size, kv_c_size, kv_f_size, 'input'),
            'ffn_c':(main_size, ffn_c_size, ffn_f_size, 'input'),
        },
        'linear_2':{
            'q':(q_c_size, q_size, q_f_size, 'q_c'),
            'k':(kv_c_size, kv_size, kv_f_size, 'kv_c'),
            'v':(kv_c_size, kv_size, kv_f_size, 'kv_c'),
            'ffn_g':(ffn_c_size, ffn_size, ffn_f_size, 'ffn_c'),
            'ffn_l':(ffn_c_size, ffn_size, ffn_f_size, 'ffn_c'),
        },
        'linear_3':{
            'v_oc':(q_size, oc_size, v_of_size, 'v_o'),
            'ffn_oc':(ffn_size, oc_size, ffn_of_size, 'ffn'),
        },
        'linear_4':{
            'output':(oc_size, main_size, v_of_size + ffn_of_size, 'output_c'),
        },
    }

    m_i_pos: Dict[str, List[int]] = {} # m_i_pos: module_input_position
    m_o_pos: Dict[str, List[int]] = {}
    m_io_pos: Dict[str, List[int]] = {}
    m_io_num: List[int] = []
    for sublayer, sublayer_modules in modules.items():
        m_sizes_i = [0] + [v[0] for v in sublayer_modules.values()]
        m_sizes_o = [0] + [v[1] for v in sublayer_modules.values()]
        m_sizes_io = [0] + [v[0] * v[1] for v in sublayer_modules.values()]
        m_io_num += m_sizes_io[1:]
        m_i_pos[sublayer], m_o_pos[sublayer], m_io_pos[sublayer] = [], [], []
        for i in range(len(m_sizes_i)):
            m_i_pos[sublayer].append(sum(m_sizes_i[:i+1]))
        for i in range(len(m_sizes_o)):
            m_o_pos[sublayer].append(sum(m_sizes_o[:i+1]))
        for i in range(len(m_sizes_io)):
            m_io_pos[sublayer].append(sum(m_sizes_io[:i+1]))

    layer_size_i_sum = 0
    layer_size_o_sum = 0
    layer_size_sum = 0
    layer_size_io_sum = 0
    for sublayer in modules:
        layer_size_i_sum += m_i_pos[sublayer][-1]
        layer_size_o_sum += m_o_pos[sublayer][-1]
        layer_size_io_sum += m_io_pos[sublayer][-1]
    layer_size_sum += layer_size_i_sum + layer_size_o_sum

config = Config()

if config.other_compile['flex_attention']:
    flex_attention = torch.compile(flex_attention, dynamic = False)
if config.other_compile['gmm']:
    gmm = torch.compile(gmm)
if config.other_compile['permute']:
    permute = torch.compile(permute)
if config.other_compile['unpermute']:
    unpermute = torch.compile(unpermute)


def layer_info(layer_id: int, config: Config) -> Tuple[bool, bool]:

    first_layer = layer_id == 0
    last_layer = layer_id == (config.layer_num - 1)
    if config.last_layer_vanilla:
        last_standard_layer = layer_id == (config.layer_num - 2)
        last_layer_vanilla = last_layer
    else:
        last_standard_layer = last_layer
        last_layer_vanilla = False
    have_router = not last_standard_layer and not last_layer_vanilla
    vanilla_matmul = first_layer or last_layer_vanilla

    return have_router, vanilla_matmul


def permute_info(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    indices = indices.view(-1) # [query num]
    index_freqs = torch.bincount(indices) # [< index size]
    filter = torch.nonzero(index_freqs, as_tuple = True)[0] # [vis], vis = valid index size (freq > 0)
    index_freqs = index_freqs[filter] # [vis]

    return index_freqs, filter # [vis] * 2


def router(
    x_queries: torch.Tensor, # [t * m_i * m_o, 2 * v] or [t * f_n, f_v] := [t * m, v]
    vector_pool: 'ParamVectorPool', 
    module: str, 
    layer_id: int, 
    top_k: int,

)-> Tuple[
    Dict[str, torch.Tensor], # router_output: dict, [t * m_i * m_o * i, 1] * 2 or [t * f_n * k * i, 1] * 2 (indices and probs), i = index_num
    torch.Tensor, # balance_loss
    torch.Tensor, # router_z_loss
]:
    # m_i = module_input_size, m_i = module_output_size, f_n = ffn_size, v = linear_vec_dim, f_v = ffn_vec_dim
    # m = m_i * m_o or f_n, v = 2 * v or f_v, n = vec_num_sqrt, n_e = vec_num_sqrt_expand

    cluster_router = torch.transpose(vector_pool.vec[module]['vec_clusters'][layer_id], 0, 1)  # [v, n]
    c_router_indices = torch.argmax(torch.matmul(x_queries, cluster_router), -1, keepdim = True)  # [t * m, v] @ [v, n] = [t * m, n] -> argmax -> [t * m, 1]
    x_queries_sorted, row_id_map = permute(x_queries, c_router_indices)
    # x_queries: [t * m, v]
    # c_router_indices: [t * m, 1]
    # x_queries_sorted: [t * m, v]
    # row_id_map: [t * m]
    index_freqs, filter = permute_info(c_router_indices) # [n_v] * 2, n_v = vec_num_sqrt valid < n
    router_blocks = torch.transpose(vector_pool.vec[module]['vec'][layer_id, filter, :, :], 1, 2) # [n_v, v, n_e]
    router_permuted = gmm(x_queries_sorted, router_blocks, index_freqs, trans_b = False) # [t * m, v] gmm ([n_v, v, n_e], [n_v]) -> [t * m, n_e]
    router = unpermute(router_permuted, row_id_map, torch.ones_like(row_id_map)) # [t * m, n_e]
    router_values, router_indices = torch.topk(router, top_k, dim = -1) # router_values: [t * m, k], router_indices: [t * m, k], k = top_k

    block_indices = vector_pool.vec_buffer[module]['block_indices'][layer_id, c_router_indices, router_indices, :] # [l, n, n_e, i] -> [t * m, k, i], l = layer_num, i = index_num
    gain_factors = vector_pool.vec[module]['gain_factors'][layer_id, c_router_indices, router_indices, :] # [l, n, n_e, i] -> [t * m, k, i]
    sigmoid_factors = vector_pool.vec[module]['sigmoid_factors'][layer_id] # [5]
    sigmoid_factors = torch.max(sigmoid_factors, torch.zeros(5))

    router_probs = 1 / (torch.exp(-1 * sigmoid_factors[0] * router_values) + sigmoid_factors[1]) # [t * m, k]
    router_probs = repeat(router_probs, 'tm k -> tm k c', c = vector_pool.index_num) # [t * m, k] -> [t * m, k, i]
    gain_probs = 1 / (torch.exp(-1 * sigmoid_factors[2] * gain_factors) + sigmoid_factors[3]) # [t * m, k, i]
    block_probs = router_probs * gain_probs * sigmoid_factors[4] # [t * m, k, i]

    block_indices = rearrange(block_indices, 'tm k c -> (tm k c) 1') # [t * m, k, i] -> [t * m * k * i, 1]
    block_probs = rearrange(block_probs, 'tm k c -> (tm k c) 1') # [t * m, k, i] -> [t * m * k * i, 1]

    router_output = {'indices': block_indices, 'probs': block_probs}

    if vector_pool.router_detach:

        router_permuted_detach = gmm(x_queries_sorted.detach(), router_blocks, index_freqs, trans_b = False) # [t * m, v] gmm ([n_v, v, n_e], [n_v]) = [t * m, n_e]
        router = unpermute(router_permuted_detach, row_id_map, torch.ones_like(row_id_map)) # [t * m, n_e]
        router_values, router_indices = torch.topk(router, top_k, dim = -1) # router_values: [t * m, k], router_indices: [t * m, k]
        router_probs = 1 / (torch.exp(-1 * sigmoid_factors[0] * router_values) + sigmoid_factors[1]) # [t * m, k]

    vec_num_s, vec_num_se = vector_pool.vec_info[module]['vec_num']
    vec_num = vec_num_s * vec_num_se
    router_indices = c_router_indices * vec_num_se + router_indices # [t * m, k]
    router_indices = router_indices.view(-1) # [t * n * k]
    router_indices_bincount = torch.bincount(router_indices, minlength = vec_num) # [vec_num]
    router_values_bincount = torch.bincount(router_indices, weights = router_probs.view(-1), minlength = vec_num) # [vec_num]

    balance_loss = torch.sum(router_indices_bincount * router_values_bincount) / (router_indices.numel() * torch.sum(router_values_bincount)) * router_indices_bincount.numel()
    router_z_loss = torch.sum(router_values ** 2) / router_values.numel()

    return router_output, balance_loss, router_z_loss



class ParamBlockPool(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.weight_pool = nn.ParameterDict()
        block_dim = config.block_dim
        hidden_block_dim = block_dim if config.no_paired else int(block_dim / 2)
        if config.global_linear_pool:
            self.weight_pool['linear'] = nn.Parameter(torch.zeros(config.global_linear_block_num, hidden_block_dim, block_dim)) # [b_n_g, d_h, d], b_n_g = global_linear_block_num
        else:
            for sublayer, block_num in config.block_num.items():
                self.weight_pool[sublayer] = nn.Parameter(torch.zeros(block_num, hidden_block_dim, block_dim)) # [b_n_i, d_h, d], b_n_i = subayer_i_block_num
        self.weight_pool['ffn'] = nn.Parameter(torch.zeros(config.ffn_block_num, 3, block_dim, config.ffn_hidden_dim)) # [b_n_f, 3, d, d_f], b_n_f = ffn_block_num

        if config.bias:
            self.bias_pool = nn.ParameterDict()
            if config.global_linear_pool:
                self.bias_pool['linear'] = nn.Parameter(torch.zeros(config.global_linear_block_num, block_dim)) # [b_n_g, d]
            else:
                for sublayer, block_num in config.block_num.items():
                    self.bias_pool[sublayer] = nn.Parameter(torch.zeros(block_num, block_dim)) # [b_n_i, d]
            self.bias_pool['ffn'] = nn.Parameter(torch.zeros(config.ffn_block_num, block_dim)) # [b_n_f, d]



class ParamVectorPool(nn.Module):
    # router -> query, vector = key, block = expert = value, query + key -> value
    
    def __init__(self, config: Config) -> None:
        super().__init__()  
        
        layer_num = config.layer_num - 2 if config.last_layer_vanilla else config.layer_num - 1
        capacity_factor = config.capacity_factor
        index_num = 1 if config.no_paired else 2
        self.vec = nn.ModuleDict()
        self.vec_buffer = {}
        self.vec_info = {}
            
        for module, module_parent in zip(config.module_list, config.module_parent):
            module_vec = nn.ParameterDict()
            module_vec_buffer = {}
            module_vec_info = {}

            if module == 'ffn':
                block_num = config.ffn_block_num
            else:
                block_num = config.global_linear_block_num if config.global_linear_pool else config.block_num[module_parent]

            vec_dim = config.ffn_vec_dim if module == 'ffn' else 2 * config.linear_vec_dim
            
            if module != 'ffn':
                module_info = config.modules[module_parent][module]
                vec_num = int(config.linear_vec_factor * math.sqrt(module_info[0] * module_info[1]))
            else:
                vec_num = int(config.ffn_vec_factor * config.ffn_size)
 
            vec_num_sqrt = int(math.sqrt(vec_num)) # vec_num_square_root
            vec_num_sqrt_expand = int(vec_num_sqrt * capacity_factor)
            vec_num_s = vec_num_sqrt
            vec_num_se = vec_num_sqrt_expand
            
            module_vec['vec_clusters'] = nn.Parameter(torch.zeros(layer_num, vec_num_s, vec_dim)) # [l, n, v], n = vec_num_s, v = vec_dim
            module_vec['vec'] = nn.Parameter(torch.zeros(layer_num, vec_num_s, vec_num_se, vec_dim)) # [l, n, n_e, v], n_e = vec_num_se
            module_vec['gain_factors'] = nn.Parameter(torch.ones(layer_num, vec_num_s, vec_num_se, index_num)) # [l, n, n_e, i], i = index_num
            module_vec['sigmoid_factors'] = nn.Parameter(torch.zeros(layer_num, 5))

            with torch.no_grad():
                module_vec['sigmoid_factors'][:, [0, 2, 4]] = 1
                module_vec['sigmoid_factors'][:, [1, 3]] = 0

            g = torch.Generator()
            g.manual_seed(42)

            module_vec_buffer['block_indices'] = torch.randint(1, block_num, (layer_num, vec_num_s, vec_num_se, index_num), dtype = torch.int32, generator = g) # [l, n, n_e, i]
            module_vec_buffer['repeat_init_vec'] = torch.cat((torch.zeros(vec_num_s, vec_dim + 2 * index_num), torch.arange(vec_num_s)[:, None]), -1) # [n, v_cat], v_cat = v + 2i +1

            module_vec_info['candidates'] = [None for _ in range(layer_num)]
            module_vec_info['vec_num'] = (vec_num_s, vec_num_se)
            module_vec_info['vec_dim'] = vec_dim

            self.vec[module] = module_vec
            self.vec_buffer[module] = module_vec_buffer
            self.vec_info[module] = module_vec_info

        for module, module_buffer in self.vec_buffer.items():
            for name, buffer in module_buffer.items():
                    self.register_buffer(f'{module}_{name}', buffer)
                    

        self.gather_list: List[List[torch.Tensor] | torch.Tensor] = []
        self.cand_list: List[List[torch.Tensor]] = []
        self.cand_num_list: List[torch.Tensor] = []
        self.layer_num = layer_num
        self.module_list = config.module_list
        self.module_num = len(config.module_list)
        self.index_num = index_num
        self.vec_dtype = self.vec['ffn']['vec'].dtype
        self.index_dtye = self.vec_buffer['ffn']['block_indices'].dtype
        self.router_detach = config.router_detach
        self.device = None


    def get_device(self) -> None:
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")


    def get_info(self) -> List[List]:

        info = [[] for _ in range(3)] # (3, layer_num * module_num) list

        for module in self.module_list:
            for i in range(self.layer_num):
                
                cand_num = self.vec_info[module]['candidates'][i].shape[0]
                vec_num = self.vec_info[module]['vec_num'][0] ** 2
                info[0].append(cand_num / vec_num)
                vec_index = self.vec_buffer[module]['block_indices'][i, :, :, 0] # [n, n_e]
                c_vec_p = torch.count_nonzero(vec_index, dim = -1) / vec_index.shape[-1] # [n]
                info[1].append(max(c_vec_p).item())
                info[2].append(min(c_vec_p).item())

        return info


    def get_vec_router_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:

        vec_balance_loss = torch.tensor(0, device = self.device)
        vec_router_z_loss = torch.tensor(0, device = self.device)

        for module in config.module_list:

            vec_clusters = self.vec[module]['vec_clusters'] # [l, n, v]
            vec = self.vec[module]['vec'] # [l, n, n_e, v]
            block_indices = self.vec_buffer[module]['block_indices'][:, :, :, 0] # [l, n, n_e]

            vec_clusters = rearrange(vec_clusters, 'l n v -> (l n) 1 v') # [l, n, v] -> [l * n, 1, v]
            vec = rearrange(vec, 'l n n_e v -> (l n) v n_e') # [l, n, n_e, v] -> [l * n, v, n_e]
            block_indices = rearrange(block_indices, 'l n n_e -> (l n) n_e') # [l, n, n_e] -> [l * n, n_e]
            vec_sum = block_indices.numel() # l * n * n_e

            vec_logits = torch.bmm(vec_clusters, vec) # [l * n, 1, v] bmm [l * n, v, n_e] = [l * n, 1, n_e]
            vec_logits = vec_logits.view(-1, vec_logits.shape[-1]) # [l * n, n_e]
            vec_sigmoids = torch.sum(torch.sigmoid(vec_logits), -1) # [l * n]
            block_indices = torch.sum(block_indices != 0, -1) # [l * n]
            vec_clusters_sum = vec_sigmoids.numel() # l * n

            module_balance_loss = torch.sum(vec_sigmoids * block_indices) / (torch.sum(vec_sigmoids) * vec_sum) * vec_clusters_sum
            module_router_z_loss = torch.sum(vec_logits ** 2) / vec_sum

            vec_balance_loss += module_balance_loss
            vec_router_z_loss += module_router_z_loss

        return vec_balance_loss, vec_router_z_loss


    def vec_cat_func(self, init: bool) -> Dict[str, torch.Tensor]:

        vec_cat_result = {}
        print(self.ffn_block_indices.device)
        print(self.vec_buffer['ffn']['block_indices'].device)
        print(self.ffn_block_indices is self.vec_buffer['ffn']['block_indices'])
        for module in self.module_list:
            vec_cat = torch.cat((
                self.vec[module]['vec'], 
                self.vec[module]['gain_factors'], 
                self.vec_buffer[module]['block_indices'].to(self.vec_dtype), 
                torch.zeros(*self.vec[module]['vec'].shape[:-1], 1, device = self.device))
                , -1) # [l, n, n_e, v_cat]
            if init:
                vec_num_s = vec_cat.shape[1]
                vec_cat = vec_cat[:, :, :vec_num_s, :] # [l, n, n, v_cat]

            vec_cat_result[module] = rearrange(vec_cat, 'l n1 n2 f_v -> l (n1 n2) f_v')
            # init: [l, n * n, v_cat],  else: [l, n * n_e, v_cat]
        return vec_cat_result
    

    def gather_list_init(self, world_size: int) -> None:
            
        list_id = 0
        list = []
        info_dim = 2 * self.index_num + 1

        for i, module in enumerate(self.module_list):
            
            shape = self.vec[module]['vec'].shape
            if i - list_id < world_size:
                list.append(torch.zeros(*shape[:-1], shape[-1] + info_dim, device = self.device))
            else:
                self.gather_list.append(list)
                list = []
                list.append(torch.zeros(*shape[:-1], shape[-1] + info_dim, device = self.device))
                list_id = i
        if list:
            pad_size = world_size - len(list)
            list = list + [torch.tensor([], device = self.device) for _ in range(pad_size)]
            self.gather_list.append(list) 
            # (list_num, world_size) list: [[m_1, ..., m_w], [...], ...]
            # w = world_size, m_i: [l, n, n_e, v_cat]


    def cand_num_list_init(self, world_size: int) -> None:

        list_num = (self.module_num - 1) // world_size + 1
        self.cand_num_list = [torch.zeros(list_num, 2, dtype = torch.int, device = self.device) for _ in range(world_size)] # (world_size, ) list: [(list_num, 2), ...]
    

    def cand_list_init(self, list_num: int) -> None:

        self.cand_list = [[] for _ in range(list_num)]

        for i in range(list_num):
            for cand_num in self.cand_num_list: # cand_num: (list_num, ) list
                if cand_num[i, 0] != 0:
                    self.cand_list[i].append(torch.zeros(int(cand_num[i, 0]), int(cand_num[i, 1]), device = self.device))
                else:
                    self.cand_list[i].append(torch.tensor([], device = self.device))
        # self.cand_list is (list_num, world_size) list: [[(module_cand_num, v_cat), ...,], ...], 


    def vec_filling(self, ddp: bool) -> None:

        if ddp:
            module_vec_list = [module for list in self.gather_list for module in list]
            cand_list = [cand for list in self.cand_list for cand in list]
        else:
            module_vec_list = self.gather_list

        for i, module in enumerate(self.module_list):

            vec_dim = self.vec_info[module]['vec_dim']
            with torch.no_grad():
                self.vec[module]['vec'].data = module_vec_list[i][:, :, :vec_dim] # [l, n, n_e, v]
                self.vec[module]['gain_factors'].data = module_vec_list[i][:, :, vec_dim: vec_dim + self.index_num] # [l, n, n_e, i]
            self.vec_buffer[module]['block_indices'] = module_vec_list[i][:, :, vec_dim + self.index_num : vec_dim + 2 * self.index_num].to(self.index_dtye) # [l, n, n_e, i]

            if ddp:
                cand_cat = cand_list[i] # [n_mc, v_cat], n_mc = module_cand_num
                mask = cand_cat[:, -1] > 0 # [n_mc]
                cand_start_list = torch.nonzero(mask, as_tuple = True)[0]
                cand_end_list = torch.roll(cand_start_list, -1)
                cand_end_list[-1] = cand_cat.shape[0]
                cand_info = torch.stack((cand_cat[mask, -1], cand_start_list, cand_end_list), dim = -1) # [valid_layer_num, 3]

                cand_i = 0
                for j in range(self.layer_num):
                    if cand_i < cand_info.shape[0] and j == cand_info[cand_i, 0]:
                        cand_start = cand_info[cand_i, 1]
                        cand_end = cand_info[cand_i, 2]
                        self.vec_info[module]['candidates'][j] = cand_cat[cand_start : cand_end, :] # [n_mlc, v_cat], n_mlc = module_layer_cand_num
                        cand_i += 1
                    else:
                        self.vec_info[module]['candidates'][j] = torch.tensor([])

            else:
                for j in range(self.layer_num):
                    self.vec_info[module]['candidates'][j] = self.cand_list[i][j] # [n_mlc, v_cat] or torch.tensor([])


    def ml_update( # ml: module & layer
        self, 
        module: str, 
        layer_id: int, 
        vec_dict: Dict[str, torch.Tensor], 
        router_indices_input: Optional[Dict[str, torch.Tensor]], 
        init: bool,

    )-> Tuple[
        torch.Tensor, # vec_updated: [n, n_e, v_cat] 
        torch.Tensor, # vec_cand: [n_mlc, v_cat] or torch.tensor([])
    ]:

        vec_num_s, vec_num_se = self.vec_info[module]['vec_num']
        vec_dim = self.vec_info[module]['vec_dim']

        vec = vec_dict[module][layer_id] # if init: [n * n, v_cat],  else: [n * n_e, v_cat]

        if not init:
            vec = vec[vec[:, -2] != 0]
            vec_old_cand = self.vec_info[module]['candidates'][layer_id].to(self.device)
            vec = torch.cat((vec, vec_old_cand), -1) # [n * n, v_cat]

        if router_indices_input is None:
            router_indices = torch.matmul(vec[:, :vec_dim], torch.transpose(self.vec[module]['vec_clusters'][layer_id], 0, 1)) # [n * n, v] @ [v, n] = [n * n, n]
            router_indices = torch.argmax(router_indices, -1) # [n * n]
        else:
            router_indices = router_indices_input[module][layer_id] # [n * n]

        vec[:, -1] = router_indices

        bincounts = torch.bincount(router_indices, minlength = vec_num_s) # [n]
        bincounts = torch.max(torch.max(bincounts), torch.tensor(vec_num_se, device = self.device)) - bincounts # [n]

        vec_pad = torch.repeat_interleave(self.vec_buffer[module]['repeat_init_vec'], bincounts, dim = 0) # [(n_max - n) * n, v_cat]
        vec_updated = torch.cat((vec, vec_pad), 0) # [n * n_max, v_cat]
        vec_updated, _ = permute(vec_updated, vec_updated[:, [-1]].to(torch.int)) # [n * n_max, v_cat]

        vec_updated = rearrange(vec_updated, '(n1 n2) v -> n1 n2 v', n1 = vec_num_s) # [n * n_max, v_cat] -> [n, n_max, v_cat]
        vec_cand = rearrange(vec_updated[:, vec_num_se:, :], 'n1 n2 v -> (n1 n2) v') # [n * (n_max - n_e), v_cat]
        if vec_cand.shape[0] != 0 and torch.any(vec_cand[:, -2] != 0):
            vec_cand = vec_cand[vec_cand[:, -2] != 0] # [n_mlc, v_cat]
            vec_cand[:, -1] = 0
            vec_cand[0, -1] = layer_id
        else:
            vec_cand = torch.tensor([], device = self.device)

        vec_updated = vec_updated[:, :vec_num_se, :]
        return vec_updated, vec_cand


    def update(
        self, 
        vec_dict: Dict[str, torch.Tensor], 
        router_indices: Optional[Dict[str, torch.Tensor]], 
        init: bool,
    )-> None:

        for module in self.module_list:

            module_vec_updated = torch.tensor([], device = self.device)
            module_vec_cand = []

            for i in range(self.layer_num):

                ml_vec_updated, ml_vec_cand = self.ml_update(module, i, vec_dict, router_indices, init) # ml: module & layer
                # vec_updated: [n, n_e, v_cat], vec_cand: [n_mlc, v_cat] or torch.tensor([])
                module_vec_updated = torch.stack((module_vec_updated, ml_vec_updated), 0) # [l, n, n_e, v_cat]
                module_vec_cand.append(ml_vec_cand) # list, each [n_mlc, v_cat] or torch.tensor([])

            self.gather_list.append(module_vec_updated) # list, each [l, n, n_e, v_cat]
            self.cand_list.append(module_vec_cand) # (module_num, layer_num) list

        ddp = False
        self.vec_filling(ddp)


    def update_ddp(
        self, 
        vec_dict: Dict[str, torch.Tensor], 
        router_indices: Optional[Dict[str, torch.Tensor]], 
        init: bool, 
        rank: int, 
        world_size: int,
    )-> None:

        list_num = (self.module_num - 1) // world_size + 1

        vec_updated_list = []
        vec_cand_list = []

        for i, module in enumerate(self.module_list):

            if i % world_size == rank:

                module_vec_updated = torch.tensor([], device = self.device)
                module_vec_cand = torch.tensor([], device = self.device)

                for j in range(self.layer_num):

                    ml_vec_updated, ml_vec_cand = self.ml_update(module, j, vec_dict, router_indices, init)
                    # ml_vec_updated: [n, n_e, v_cat], ml_vec_cand: [n_mlc, v_cat] or torch.tensor([])
                    module_vec_updated = torch.stack((module_vec_updated, ml_vec_updated), 0) # [l, n, n_e, v_cat]
                    module_vec_cand = torch.cat((module_vec_cand, ml_vec_cand), 0) # [l, n_mlc, v_cat]
                
                vec_updated_list.append(module_vec_updated) # list, each [l, n, n_e, v_cat]
                vec_cand_list.append(module_vec_cand) # list, each [n_mc, v_cat] or torch.tensor([]), n_mc = module_cand_num

        vec_updated_list += [torch.tensor([], device = self.device) for _ in range(list_num - len(vec_updated_list))] # padding length to list_num
        vec_cand_list += [torch.tensor([], device = self.device) for _ in range(list_num - len(vec_cand_list))]

        vec_cand_num = torch.zeros(list_num, 2, dtype = torch.int, device = self.device) # [list_num, 2]
        for i, vec in enumerate(vec_cand_list):
            if vec:
                vec_cand_num[i, 0] = vec.shape[0]
                vec_cand_num[i, 1] = vec.shape[1]
            else:
                vec_cand_num[i, :] = 0

        torch.cuda.synchronize()

        dist.all_gather(self.cand_num_list, vec_cand_num)
        self.cand_list_init(list_num)

        for i in range(list_num):
            dist.all_gather(self.gather_list[i], vec_updated_list[i])
            dist.all_gather(self.cand_list[i], vec_cand_list[i])

        ddp = True
        self.vec_filling(ddp)


    def vec_init(
        self, 
        rank: int, 
        world_size: int, 
        ddp: bool,
    )-> None:

        init = True
        vec_dict = self.vec_cat_func(init) # dict, each [l, n * n, v_cat]
        router_indices = {}

        for module in self.module_list:

            vec_dim = self.vec_info[module]['vec_dim'] # v
            vec_num_s, _ = self.vec_info[module]['vec_num'] # n
            k_means = KMeans(n_clusters = vec_num_s, verbose = 1)

            router_indices[module] = torch.zeros(*vec_dict[module].shape[:-1], dtype = torch.int, device = self.device) # [l, n * n]
            results_centers = torch.zeros(self.layer_num, vec_num_s, vec_dim, device = self.device) # [l, n, v]

            for i in range(self.layer_num):
                vec = vec_dict[module][i, :, :vec_dim] # [n * n, v]
                layer_result_labels = k_means.fit_predict(vec) # [n * n, v] -> [n * n]
                router_indices[module][i] = layer_result_labels
                for j in range(vec_num_s):
                    index = layer_result_labels == j # [n * n]
                    center = vec[index, :].mean(0) # [v]
                    results_centers[i, j, :] = center

            with torch.no_grad():
                self.vec[module]['vec_clusters'].data = results_centers # [l, n, v]
        
        if ddp:
            self.update_ddp(vec_dict, router_indices, init, rank, world_size)
        else:
            self.update(vec_dict, router_indices, init)


    def training_update(
        self, 
        rank: int, 
        world_size: int, 
        ddp: bool,
    )-> None:

        init = False
        router_indices = None
        vec_dict = self.vec_cat_func(init) # dict, each [l, n * n_e, v_cat]

        if ddp:
            self.update_ddp(vec_dict, router_indices, init, rank, world_size)
        else:
            self.update(vec_dict, router_indices, init)



class Linear(nn.Module):

    def __init__(
        self, 
        config: Config, 
        sublayer: str, 
        layer_id: int,
    )-> None:
        super().__init__()
        
        block_dim = config.block_dim # d
        vec_dim = config.linear_vec_dim # v
        s_vec_dim = config.sublayer_vec_dim # s_v
        s_i_size = config.m_i_pos[sublayer][-1] # sublayer_size_input, s_i
        s_o_size = config.m_o_pos[sublayer][-1] # s_o
        linear_fixed_dim = config.linear_fixed_dim # d_lf
        
        m_vec_dim = int(vec_dim * math.sqrt(s_o_size / s_i_size)) ## m_v, m: multihead intermediate
        vec_dim_sum = vec_dim + m_vec_dim + s_vec_dim # v_sum = v + m_v + s_v

        self.have_router, self.vanilla_matmul = layer_info(layer_id, config)
            
        if self.have_router: # qw = query_weight, qb = query_bias, qw2 = query_weight_2
            # unified_qw = blockwise_qw + multihead_qw + sublayer_qw
            self.unified_qw = nn.Parameter(torch.zeros(s_i_size, block_dim, vec_dim_sum)) # [s_i, d, v_sum]
            self.blockwise_qb = nn.Parameter(torch.zeros(1, s_i_size * vec_dim)) # [1, s_i * v]
            self.multihead_qb = nn.Parameter(torch.zeros(1, s_o_size * vec_dim)) # [1, s_o * v]
            self.sublayer_qw2 = nn.Parameter(torch.zeros(s_i_size * s_vec_dim, (s_i_size + s_o_size) * vec_dim)) # [s_i * s_v, (s_i + s_o) * v]
        self.multihead_qw2 = nn.ParameterDict()
        self.fixed_w = nn.ParameterDict()

        sublayer_modules = config.modules[sublayer]
        fixed_type = config.sublayer_fixed_type[sublayer]
        for module, info in sublayer_modules.items():
            m_i_size, m_o_size, m_f_size = info[0], info[1], info[2]
            
            if self.have_router:
                self.multihead_qw2[module] = nn.Parameter(torch.zeros(m_i_size * m_vec_dim, m_o_size * vec_dim)) # [m_i * m_v, m_o * v]
            if self.vanilla_matmul:
                self.fixed_w[module] = nn.Parameter(torch.zeros(m_i_size * block_dim, m_o_size * block_dim)) # [m_i * d, m_o * d]
            elif config.fixed:  
                if fixed_type == 1:
                    self.fixed_w[module] = nn.Parameter(torch.zeros(m_i_size * block_dim, m_f_size * linear_fixed_dim)) # [m_i * d, m_f * d_lf]
                else:
                    self.fixed_w[module] = nn.Parameter(torch.zeros(m_f_size * linear_fixed_dim, m_o_size * block_dim)) # [m_f * d_lf, m_o * d]

        self.sublayer = sublayer
        self.layer_id = layer_id
        self.block_dim = block_dim
        self.vec_dim = vec_dim
        self.m_vec_dim = m_vec_dim
        self.s_vec_dim = s_vec_dim
        self.s_i_size = s_i_size
        self.s_o_size = s_o_size
        self.sublayer_input_repeat_num = config.input_repeat_num[sublayer]
        self.m_i_pos = config.m_i_pos[sublayer]
        self.m_o_pos = config.m_o_pos[sublayer]
        self.m_io_pos = config.m_io_pos[sublayer]
        self.sublayer_modules = sublayer_modules
        self.fixed = config.fixed
        self.fixed_type = fixed_type
        self.stream = config.multi_stream
        self.bias = config.bias
        self.info_fusion_router = config.info_fusion_router
        self.global_linear_pool = config.global_linear_pool
        self.no_paired = config.no_paired
        self.seq_len = config.seq_len

    
    def forward(
        self, 
        x: Dict[str, torch.Tensor], # (previous_module_num, ) dict, each [t, pre_m_o * d], pre_m_o = previous_module_size_output
        x_fixed: Optional[Dict[str, torch.Tensor]], # (module_num, ) dict or None, each [t, m_f * d]
        external_input: Optional[Dict[str, Dict[str, torch.Tensor] | torch.Tensor]], # (sublayer_num, 2) dict or None, total [t, layer_size_sum * v + f_n * f_v], each [t, s_i * v] or [t, s_o * v]
        block_pool: ParamBlockPool, 
        vector_pool: ParamVectorPool,
        router: Callable,
        router_result: Optional[Dict[str, Dict[str, torch.Tensor]]], # (sublayer_num, 2) dict or None, each [t * s_io * i, 1], s_io = (m_io)_sum

    )-> Tuple[
        Dict[str, torch.Tensor], # x_output: dict, each [t, m_o * d], total [t, s_o * d]
        Optional[Dict[str, torch.Tensor]], # x_fixed_output: dict, each [t, m_f * d] or None
        Optional[Dict[str, torch.Tensor]], # router_output: [t * s_io * i, 1] * 2 or None
        torch.Tensor, # balance_loss
        torch.Tensor, # router_z_loss
    ]:
        
        s_router = torch.cuda.Stream()
        s_main = torch.cuda.Stream()
        s_fixed = torch.cuda.Stream()

        router_ctx = torch.cuda.stream(s_router) if self.stream else nullcontext()
        main_ctx = torch.cuda.stream(s_main) if self.stream else nullcontext()
        fixed_ctx = torch.cuda.stream(s_fixed) if self.stream else nullcontext()

        device = block_pool.weight_pool['ffn'].device

        x_cat = torch.tensor(self.seq_len, 0, device = device)
        for module, x_module in x.items():
            num = self.sublayer_input_repeat_num[module]
            x_cat = torch.cat((x_cat, x_module.repeat(1, num)), -1) # [t, m_i * d] -> [t, m_r * m_i * d] -> [t, (m_r * m_i * d)_sum], m_r = module_input_repeat_num
        x_cat = rearrange(x_cat, 't (s_i d) -> s_i t d', d = self.block_dim) # [t, (m_r * m_i * d)_sum] -> [(m_r * m_i)_sum, t, d] = [s_i, t, d], (m_r * m_i)_sum = s_i

        if self.have_router:
            
            with router_ctx:

                if self.info_fusion_router:
                    assert isinstance(external_input, dict)
                    external = external_input[self.sublayer] # (2, ) dict: [t, s_i * v], [t, s_o * v]
                    assert isinstance(external, dict)
                else:
                    external = {'blockwise': torch.tensor(0, device = device), 'multihead': torch.tensor(0, device = device)}
                
                q_unified = torch.bmm(x_cat, self.unified_qw) # [s_i, t, d] bmm [s_i, d, v_sum] = [s_i, t, v_sum]
                q_blockwise = q_unified[:, :, : self.vec_dim] # [s_i, t, v]
                q_multihead_1 = q_unified[:, :, self.vec_dim : self.vec_dim + self.m_vec_dim] # [s_i, t, m_v]
                q_sublayer_1 = q_unified[:, :, self.vec_dim + self.m_vec_dim:] # [s_i, t, s_v]

                q_blockwise = rearrange(q_blockwise, 's_i t v -> t (s_i v)') # [s_i, t, v] -> [t, s_i * v]
                q_multihead_1 = rearrange(q_multihead_1, 's_i t v -> t (s_i v)') # [s_i, t, m_v] -> [t, s_i * m_v]
                q_sublayer_1 = rearrange(q_sublayer_1, 's_i t v -> t (s_i v)') # [s_i, t, s_v] -> [t, s_i * s_v]
                    
                q_sublayer_2 = torch.matmul(q_sublayer_1, self.sublayer_qw2) # [t, s_i * s_v] @ [s_i * s_v, (s_i + s_o) * v] = [t, (s_i + s_o) * v]
                q_blockwise = q_blockwise + self.blockwise_qb + q_sublayer_2[:, : self.s_i_size * self.vec_dim] + external['blockwise'] # [t, s_i * v]
        
                q_multihead = torch.tensor(self.seq_len, 0, device = device)
                for i, module in enumerate(self.sublayer_modules):
                    index_l = self.m_i_pos[i] * self.m_vec_dim
                    index_r = self.m_i_pos[i+1] * self.m_vec_dim # index_r - index_l = m_i * m_v
                    q_multihead_2 = torch.matmul(q_multihead_1[:, index_l : index_r], self.multihead_qw2[module]) # [t, m_i * m_v] @ [m_i * m_v, m_o * v] = [t, m_o * v]
                    q_multihead = torch.cat((q_multihead, q_multihead_2), -1) # [t, (m_o * v)_sum] = [t, s_o * v]
                q_multihead = q_multihead + self.multihead_qb + q_sublayer_2[:, self.s_i_size * self.vec_dim:] + external['multihead'] # [t, s_o * v]
        
                x_queries = {}
                for i, module in enumerate(self.sublayer_modules):
                    index_il = self.m_i_pos[i] * self.vec_dim
                    index_ir = self.m_i_pos[i+1] * self.vec_dim # index_ir - index_il = m_i * v
                    index_ol = self.m_o_pos[i] * self.vec_dim
                    index_or = self.m_o_pos[i+1] * self.vec_dim # index_or - index_ol = m_o * v
                    m_i_size = self.m_i_pos[i+1] - self.m_i_pos[i]
                    m_o_size = self.m_o_pos[i+1] - self.m_o_pos[i]

                    q_blockwise = repeat(q_blockwise[:, index_il : index_ir], 't (m_i v) -> (t m_i m_o) v', v = self.vec_dim, m_o = m_o_size) # [t, m_i * v] -> [t * m_i * m_o, v]
                    q_multihead = repeat(q_multihead[:, index_ol : index_or], 't (m_o v) -> (t m_i m_o) v', v = self.vec_dim, m_i = m_i_size) # [t, m_o * v] -> [t * m_i * m_o, v]
                    x_queries[module] = torch.cat((q_blockwise, q_multihead), -1) # [t * m_i * m_o, 2 * v]
        
                router_output: Optional[Dict[str, torch.Tensor]] = None
                balance_loss = torch.tensor(0, device = device)
                router_z_loss = torch.tensor(0, device = device)
                router_output_index = torch.tensor(0, 1, device = device)
                router_output_prob = torch.tensor(0, 1, device = device)
                top_k = 1

                for module, x_query in x_queries.items(): # x_queries: dict, each [t * m_i * m_o, 2 * v]
                    m_router_output, m_balance_loss, m_router_z_loss = router(x_query, vector_pool, module, self.layer_id, top_k) # [t * m_i * m_o * i, 1] * 2
                    router_output_index = torch.cat((router_output_index, m_router_output['indices']), 0) # [(t * m_i * m_o)_sum * i, 1] = [t * s_io * i, 1]
                    router_output_prob = torch.cat((router_output_prob, m_router_output['probs']), 0) # [(t * m_i * m_o)_sum * i, 1] = [t * s_io * i, 1]
                    balance_loss += m_balance_loss
                    router_z_loss += m_router_z_loss
                router_output = {'indices': router_output_index, 'probs': router_output_prob}


        if not self.vanilla_matmul: 
            
            with main_ctx:

                assert router_result is not None
                block_indices = router_result[self.sublayer]['indices'] # [t * s_io * i, 1]
                block_probs = router_result[self.sublayer]['probs'] # [t * s_io * i, 1]
               
                index_num = 1 if self.no_paired else 2
                hidden_block_dim = int(self.block_dim / index_num)
                x_input = torch.tensor(0, hidden_block_dim, device = device)
                for i, module in enumerate(self.sublayer_modules): # x_cat: [s_i, t, d]
                    m_o_size = self.sublayer_modules[module][1]
                    x_module = x_cat[self.m_i_pos[i] : self.m_i_pos[i+1], :, :] # [m_i, t, d]
                    x_input = torch.cat((x_input, repeat(x_module, 'm_i t (i d) -> (t m_i m_o i) d', m_o = m_o_size, i = index_num)), 0) # [m_i, t, d] -> [t * m_i * m_o * i, d/i] -> [(t * m_i * m_o)_sum * i, d/i] = [t * s_io * i, d_h], d_h = hidden_block_dim = d/i
                x_input_cat = torch.cat((x_input, block_indices), -1) # [t * s_io * i, d_h + 1],

                x_sorted_cat, row_id_map = permute(x_input_cat, block_indices)
                # x_input_cat: [t * s_io * i, d_h + 1]
                # block_indices: [t * s_io * i, 1]
                # x_sorted_cat: [t * s_io * i, d_h + 1]
                # row_id_map: [t * s_io * i]
                x_sorted = x_sorted_cat[:, :-1] # [t * s_io * i, d_h]
                block_indices_sorted = x_sorted_cat[:, -1] # [t * s_io * i]

                index_freq, filter = permute_info(block_indices) # [vis] * 2, vis = valid index size

                sublayer = 'linear' if self.global_linear_pool else self.sublayer
                blocks = block_pool.weight_pool[sublayer][filter, :, :] # [vis, d_h, d]
                x_permuted = gmm(x_sorted, blocks, index_freq, trans_b = False) # [t * s_io * i, d_h] gmm ([vis, d_h, d], [vis]) = [t * s_io * i, d]

                if self.bias:
                    bias = torch.index_select(block_pool.bias_pool[sublayer], 0, block_indices_sorted) # [t * s_io * i, d]
                    x_permuted += bias # [t * s_io * i, d]
                x_unpermuted = unpermute(x_permuted, row_id_map, block_probs) # [t * s_io * i, d]
                
                x_output: Dict[str, torch.Tensor] = {}
                for i, module in enumerate(self.sublayer_modules):
                    m_i_size = self.sublayer_modules[module][0]
                    m_o_size = self.sublayer_modules[module][1]
                    index_l = self.m_io_pos[i] * self.seq_len * index_num
                    index_r = self.m_io_pos[i+1] * self.seq_len * index_num # index_r - index_l = t * m_io * i = t * m_i * m_o * i
                    x_output[module] = reduce(x_unpermuted[index_l : index_r, :], '(t m_i m_o i) d -> t (m_o d)', 'sum', m_i = m_i_size, m_o = m_o_size, i = index_num) # [t * m_i * m_o * i, d] -> [t, m_o * d]  


        if self.vanilla_matmul or self.fixed:
            
            with fixed_ctx:
                
                x_fixed_input = {}
                for module, info in self.sublayer_modules.items():
                    condition = self.fixed_type == 1 or self.vanilla_matmul
                    if condition:
                        x_fixed_input[module] = x[info[3]]
                    else:
                        assert x_fixed is not None
                        x_fixed_input[module] = x_fixed[info[3]]
                # if vanilla_matmul or fixed_type == 1: [t, m_i * d] else: [t, m_f * d_lf]
        
                x_fixed_output = {}
                for i, module in enumerate(self.sublayer_modules):
                    x_fixed_output[module] = torch.matmul(x_fixed_input[module], self.fixed_w[module])
                # if vanilla_matmul: [t, m_i * d] @ [m_i * d, m_o * d] = [t, m_o * d]
                # if fixed_type == 1: [t, m_i * d] @ [m_i * d, m_f * d_lf] = [t, m_f * d_lf]
                # if fixed_type == 0: [t, m_f * d_lf] @ [m_f * d, m_o * d] = [t, m_o * d]
        else:
            x_fixed_output = None

        torch.cuda.synchronize()

        if not self.vanilla_matmul and self.fixed and self.fixed_type == 0:
            assert x_fixed_output is not None
            for module in self.sublayer_modules:
                x_output[module] += x_fixed_output[module] # [t, m_o * d]
            x_fixed_output = None

        if not self.have_router:
            router_output = None
            balance_loss = torch.tensor(0, device = device)
            router_z_loss = torch.tensor(0, device = device)

        if self.vanilla_matmul:
            assert x_fixed_output is not None
            x_output = x_fixed_output # dict, each [t, m_o * d], total [t, s_o * d]
            x_fixed_output = None

        return x_output, x_fixed_output, router_output, balance_loss, router_z_loss
    


class FFN(nn.Module):

    def __init__(
        self, 
        config: Config, 
        layer_id: int,
    )-> None:
        super().__init__()

        block_dim = config.block_dim # d
        ffn_fixed_dim = config.ffn_fixed_dim # d_ff
        ffn_fixed_vanilla_matmul_dim = config.ffn_fixed_vanilla_matmul_dim # d_ffvm
        vec_dim = config.ffn_vec_dim # f_v
        s_vec_dim = config.sublayer_vec_dim # s_v
        ffn_size = config.ffn_size # f_n
        vec_dim_sum = vec_dim + s_vec_dim # v_sum = f_v + s_v

        self.have_router, self.vanilla_matmul = layer_info(layer_id, config)

        if self.have_router:
            # unified_qw = blockwise_qw + sublayer_qw
            self.unified_qw = nn.Parameter(torch.zeros(ffn_size, block_dim * 2, vec_dim_sum)) # [f_n, 2 * d, v_sum]
            self.blockwise_qb = nn.Parameter(torch.zeros(1, ffn_size * vec_dim)) # [1, f_n * f_v]
            self.sublayer_qw2 = nn.Parameter(torch.zeros(ffn_size * s_vec_dim, ffn_size * vec_dim)) # [f_n * s_v, f_n * f_v]
        if self.vanilla_matmul:
            self.fixed_w = nn.Parameter(torch.zeros(ffn_size, 3, block_dim, ffn_fixed_vanilla_matmul_dim)) # [f_n, 3, d, d_ffvm]
        else:
            self.fixed_w = nn.Parameter(torch.zeros(ffn_size, 3, block_dim, ffn_fixed_dim)) # [f_n, 3, d, d_ff]

        self.layer_id = layer_id
        self.block_dim = block_dim
        self.ffn_hidden_dim = config.ffn_hidden_dim
        self.vec_dim = vec_dim
        self.ffn_size = ffn_size
        self.stream = config.multi_stream
        self.bias = config.bias
        self.fixed = config.fixed
        self.top_k = config.ffn_top_k # k
        self.info_fusion_router = config.info_fusion_router
        self.no_paired = config.no_paired


    def forward(
        self, 
        x_input_dict: Dict[str, torch.Tensor], # dict, ffn_l & ffn_g: [t, f_n * d]
        external_input: Optional[Dict[str, Dict[str, torch.Tensor] | torch.Tensor]], # (sublayer_num, 2) dict, total [t, layer_size_sum * v + f_n * f_v], each [t, s_i * v] or [t, s_o * v] or None
        block_pool: ParamBlockPool, 
        vector_pool: ParamVectorPool,
        router: Callable, 
        router_result: Optional[Dict[str, Dict[str, torch.Tensor]]], # dict, ffn: [t * f_n * k * i, 1] * 2 or None

    )-> Tuple[
        Dict[str, torch.Tensor], # x_output: [t, f_n * d]
        Optional[Dict[str, torch.Tensor]], # router_output: [t * f_n * k * i, 1] * 2 or None
        torch.Tensor, # balance_loss
        torch.Tensor, # router_z_loss
    ]:
       
        s_router = torch.cuda.Stream()
        s_main = torch.cuda.Stream()
        s_fixed = torch.cuda.Stream()

        router_ctx = torch.cuda.stream(s_router) if self.stream else nullcontext()
        main_ctx = torch.cuda.stream(s_main) if self.stream else nullcontext()
        fixed_ctx = torch.cuda.stream(s_fixed) if self.stream else nullcontext()

        device = block_pool.weight_pool['ffn'].device

        x_ffn_g = rearrange(x_input_dict['ffn_g'], 't (f_n d) -> f_n t d', d = self.block_dim) # [t, f_n * d] -> [f_n, t, d]
        x_ffn_l = rearrange(x_input_dict['ffn_l'], 't (f_n d) -> f_n t d', d = self.block_dim) # [t, f_n * d] -> [f_n, t, d]
        x = torch.cat((x_ffn_g, x_ffn_l), 2) # [f_n, t, 2d]

        if self.have_router:
            
            with router_ctx:

                if self.info_fusion_router:
                    assert isinstance(external_input, dict)
                    external = external_input['ffn'] # [t, f_n * f_v]
                    assert isinstance(external, dict)
                else:
                    external = torch.tensor(0, device = device)

                q_unified = torch.bmm(x, self.unified_qw) # [f_n, t, 2d] bmm [f_n, 2d, v_sum] = [f_n, t, v_sum]
                q_sublayer_1 = rearrange(q_unified[:, :, self.vec_dim:], 'f_n t s_v -> t (f_n s_v)') # [f_n, t, s_v] -> [t, f_n * s_v]
                q_sublayer_2 = torch.matmul(q_sublayer_1, self.sublayer_qw2) # [t, f_n * s_v] @ [f_n * s_v, f_n * f_v] = [t, f_n * f_v]
                q_other = q_sublayer_2 + self.blockwise_qb + external # [t, f_n * f_v]
                x_queries = rearrange(q_unified[:, :, :self.vec_dim], 'f_n t f_v -> (t f_n) f_v')  + rearrange(q_other, 't (f_n f_v) -> (t f_n) f_v', f_v = self.vec_dim) # [t * f_n, f_v]
                router_output, balance_loss, router_z_loss = router(x_queries, vector_pool, 'ffn', self.layer_id, self.top_k) # [t * f_n * k * i, 1] * 2
                
        if not self.vanilla_matmul:
        
            with main_ctx:

                assert router_result is not None
                block_indices = router_result['ffn']['indices'] # [t * f_n * k * i, 1]
                block_probs = router_result['ffn']['probs'] # [t * f_n * k * i, 1]
                
                index_num = 1 if self.no_paired else 2
                x_input = repeat(x, 'f_n t d -> (t f_n k) d', k = index_num * self.top_k) # [f_n, t, 2d] -> [t * f_n * k * i, 2d]
                x_input_cat = torch.cat((x_input, block_indices), -1) # [t * f_n * k * i, 2d + 1]
                x_sorted_cat, row_id_map = permute(x_input_cat, block_indices)
                # x_input_cat: [t * f_n * k * i, 2d + 1]
                # block_indices: [t * f_n * k * i, 1]
                # x_sorted_cat: [t * f_n * k * i, 2d + 1]
                # row_id_map: [t * f_n * k * i]
                x_sorted = x_sorted_cat[:, :-1] # [t * f_n * k * i, 2d]
                block_indices_sorted = x_sorted_cat[:, -1] # [t * f_n * k * i]

                index_freq, filter = permute_info(block_indices) # [vis] * 2, vis = valid index size

                blocks = block_pool.weight_pool['ffn'][filter, :, :, :] # [vis, 3, d, d_f]
                block_3t = torch.transpose(blocks[:, 2, :, :], 1, 2) # [vis, d_f, d]

                x_permuted_1 = gmm(x_sorted[:, :self.block_dim], blocks[:, 0, :, :], index_freq, trans_b = False) # [t * f_n * k * i, d] gmm ([vis, d, d_f] , [vis]) = [t * f_n * k * i, d_f]
                x_permuted_2 = gmm(x_sorted[:, self.block_dim:], blocks[:, 1, :, :], index_freq, trans_b = False) # [t * f_n * k * i, d_f]
                x_permuted = F.silu(x_permuted_1) * x_permuted_2 # [t * f_n * k * i, d_f]
                x_permuted = gmm(x_permuted, block_3t, index_freq, trans_b = False) # [t * f_n * k * i, d]

                if self.bias:
                    bias = torch.index_select(block_pool.bias_pool['ffn'], 0, block_indices_sorted) # [t * f_n * k * i, d]
                    x_permuted += bias # [t * f_n * k * i, d]
                x_output = unpermute(x_permuted, row_id_map, block_probs) # [t * f_n * k * i, d]
                x_output = reduce(x_output, '(t f_n k i) d -> t (f_n d)', 'sum', f_n = self.ffn_size, k = self.top_k, i = index_num) # [t, f_n * d]


        if self.vanilla_matmul or self.fixed:

            with fixed_ctx:

                fixed_1 = torch.bmm(x[:, :, :self.block_dim], self.fixed_w[:, 0, :, :]) 
                # vanilla_matmul: [f_n, t, d] bmm [f_n, d, d_ffvm] = [f_n, t, d_ffvm]
                # not vanilla_matmul: [f_n, t, d] bmm [f_n, d, d_ff] = [f_n, t, d_ff]
                fixed_2 = torch.bmm(x[:, :, self.block_dim:], self.fixed_w[:, 1, :, :]) # same as above
                fixed = F.silu(fixed_1) * fixed_2
                # [f_n, t, d_ffvm] or [f_n, t, d_ff]
                fixed_w3t = torch.transpose(self.fixed_w[:, 2, :, :], 1, 2) # [f_n, v, d] or [f_n, d_ff, d]
                fixed = torch.bmm(fixed, fixed_w3t)
                # vanilla_matmul: [f_n, t, d_ffvm] bmm [f_n, d_ffvm, d] = [f_n, t, d]
                # not vanilla_matmul: [f_n, t, d_ff] bmm [f_n, d_ff, d] = [f_n, t, d]
                fixed = rearrange(fixed, 'f_n t d -> t (f_n d)') # [f_n, t, d] -> [t, f_n * d]
        else:
            fixed = None

        torch.cuda.synchronize()

        if not self.vanilla_matmul and self.fixed:
            assert fixed is not None
            x_output += fixed # [t, f_n * d]

        if not self.have_router:
            router_output = None
            balance_loss = torch.tensor(0, device = device)
            router_z_loss = torch.tensor(0, device = device)

        if self.vanilla_matmul:
            assert fixed is not None
            x_output = fixed # [t, f_n * d]
        
        return x_output, router_output, balance_loss, router_z_loss
    


class External(nn.Module):

    def __init__(
        self, 
        config: Config, 
        layer_id: int,
    )-> None:
        super().__init__()

        have_router, _ = layer_info(layer_id, config)

        external_c_dim = config.external_c_dim # e_d
        if layer_id != 0:
            input_size = (config.main_size + config.oc_size * 2) * config.block_dim # (n + 2 * oc_n) * d, n = config.main_size
        else:
            input_size = config.main_size * config.block_dim # n * d
        output_size = config.layer_size_sum * config.linear_vec_dim + config.ffn_size * config.ffn_vec_dim # l_sum * v + f_n * f_v, l_sum = layer_size_sum

        external_condition = config.info_fusion_router and have_router
        if external_condition:
            self.external_w1 = nn.Parameter(torch.zeros(input_size, external_c_dim)) # [(n + 2 * oc_n) * d, e_d] or [n * d, e_d]
            self.external_w2 = nn.Parameter(torch.zeros(external_c_dim, output_size)) # [e_d, l_sum * v + f_n * f_v]

        factor_num = 0
        if external_condition:
            factor_num += 2
            if layer_id == 0:
                factor_num -= 1
        if config.residual_norm:
            factor_num += 1

        self.external_factors = nn.Parameter(torch.zeros(1, factor_num))

        self.sublayers = config.modules
        self.m_i_pos = config.m_i_pos
        self.m_o_pos = config.m_o_pos
        self.vec_dim = config.linear_vec_dim
        self.layer_id = layer_id
        self.info_fusion_router = config.info_fusion_router
        self.residual_norm = config.residual_norm
        self.factor_num = factor_num
        self.external_condition = external_condition

    
    def forward(
        self, 
        x: torch.Tensor, # [t, n * d]  attn_c: [t, oc_n * d], norm: [t, 1]
        attn_c: torch.Tensor, 
        norm: torch.Tensor,

    )-> Tuple[
        Optional[Dict[str, Dict[str, torch.Tensor] | torch.Tensor]], # external_output: dict, total [t, l_sum * v + f_n * f_v]
        Optional[torch.Tensor], # residual_norm_coeff: [t, 1]
    ]:
        
        norm_coeff = torch.pow(norm, self.external_factors) # [t, factor_num]
        router_x_coeff = norm_coeff[:, [0]] if self.external_condition else None # [t, 1] or None
        router_attn_coeff = norm_coeff[:, [1]] if self.external_condition and self.layer_id != 0 else None # [t, 1] or None
        residual_norm_coeff = norm_coeff[:, [-1]] if self.residual_norm else None # [t, 1] or None

        if not self.external_condition:
            external_output = None
            return None, residual_norm_coeff

        x_cat = torch.cat((x * router_x_coeff, attn_c, attn_c * router_attn_coeff), 1) if self.layer_id != 0 else x * router_x_coeff
        # if layer_id != 0: [t, (n + 2 * oc_n) * d] else [t, n * d]
        external = torch.matmul(x_cat, self.external_w1)
        # if layer_id != 0: [t, (n + 2 * oc_n) * d] @ [(n + 2 * oc_n) * d, e_d] = [t, e_d]
        # else: [t, n * d] @ [nd, e_d] = [t, e_d]
        external = torch.matmul(external, self.external_w2) # [t, e_d] @ [e_d, l_sum * v + f_n * f_v] = [t, l_sum * v + f_n * f_v]

        pos = 0
        external_output = {}
        for sublayer in self.sublayers:
            external_output[sublayer] = {}
            pos_i = self.m_i_pos[sublayer][-1] * self.vec_dim
            external_output[sublayer]['blockwise'] = external[:, pos : pos + pos_i] # [t, s_i * v]
            pos += pos_i
            pos_o = self.m_o_pos[sublayer][-1] * self.vec_dim
            external_output[sublayer]['multihead'] = external[:, pos : pos + pos_o] # [t, s_o * v]
            pos += pos_o

        external_output['ffn'] = external[:, pos:] # [t, f_n * f_v]

        return external_output, residual_norm_coeff 
    


class RotaryEmb(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()

        attn_dim = config.attn_size * config.block_dim # a_n * d, a_n = attn_size
        theta = config.theta
        self.freqs = 1.0 / (theta ** torch.arange(0, attn_dim, 2) / attn_dim) # [a_n * d/2]
        self.seq_len: Optional[int] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # xq: [1, h_q, t, a_n * d], xk: [1, h_k, t, a_n * d], h_q = q_head_num

        seq_len = xq.shape[2] # t
        if self.seq_len != seq_len:
            self.seq_len = seq_len
            outer_freqs = torch.outer(torch.arange(self.seq_len), self.freqs) # [t, a_n * d/2]
            self.freqs_cis = torch.polar(torch.ones_like(outer_freqs), outer_freqs)[None, None, :, :] # [1, 1, t, a_n * d/2]

        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # [1, h_q, t, a_n * d] -> [1, h_q, t, a_n * d/2, 2] -> [1, h_q, t, a_n * d/2]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # [1, h_k, t, a_n * d] -> [1, h_k, t, a_n * d/2, 2] -> [1, h_k, t, a_n * d/2]
        xq_output = torch.view_as_real(xq_ * self.freqs_cis).flatten(3) # [1, h_q, t, a_n * d/2] -> [1, h_q, t, a_n * d/2, 2] -> flatten(3) -> [1, h_q, t, a_n * d]
        xk_output = torch.view_as_real(xk_ * self.freqs_cis).flatten(3) # same as above

        return xq_output.type_as(xq), xk_output.type_as(xk) # xq: [1, h_q, t, a_n * d], xk: [1, h_k, t, a_n * d]
    


class Attention(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.attn_dim = config.attn_size * config.block_dim # a_n * d
        self.seq_len = config.seq_len # t
        self.gqa = config.gqa
        self.qk_norm = config.qk_norm
        self.norm_eps = config.norm_eps

    def forward(
        self, 
        x: Dict[str, torch.Tensor], 
        rotary_emb: RotaryEmb, 
        block_mask: BlockMask,
    )-> torch.Tensor:
        # x: dict, 'q' 'k' 'v': [t, h * a_n * d] and 'ffn_l' 'ffn_g'

        x_attn = {}
        for module, x_module in x.items():
            if module in ('q', 'k', 'v'):
                x_attn[module] = rearrange(x_module, 't (h d) -> 1 h t d', d = self.attn_dim) # [t, h * a_n * d] -> [1, h, t, a_n * d]
        xq, xk, xv = x_attn['q'], x_attn['k'], x_attn['v']

        if self.qk_norm:
            xq = F.normalize(xq, dim = -1, eps = self.norm_eps) # [1, h_q, t, a_n * d]
            xk = F.normalize(xk, dim = -1, eps = self.norm_eps) # [1, h_k, t, a_n * d]
        
        xq, xk = rotary_emb(xq, xk) # [1, h_q, t, a_n * d], [1, h_k, t, a_n * d]
        y_output = flex_attention(xq, xk, xv, block_mask = block_mask, enable_gqa = self.gqa) # [1, h_q, t, a_n * d]
        assert isinstance(y_output, torch.Tensor)
        y_output = rearrange(y_output, 'b h t d -> (b t) (h d)') # [t, h_q * a_n * d]

        return y_output # [t, h_q * a_n * d]
    


class Layer(nn.Module):

    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()

        self.sublayers = nn.ModuleDict({
            'linear_1': Linear(config, 'linear_1', layer_id),
            'linear_2': Linear(config, 'linear_2', layer_id),
            'ffn': FFN(config, layer_id),
            'linear_3': Linear(config, 'linear_3', layer_id),
            'linear_4': Linear(config, 'linear_4', layer_id), 
        })

        self.attention = Attention(config)
        self.external = External(config, layer_id)
        block_dim = config.block_dim

        self.elementwise_coeffs = nn.ParameterList([nn.Parameter(torch.ones(1, config.main_size * block_dim)), nn.Parameter(torch.ones(1, config.oc_size * block_dim)), nn.Parameter(torch.ones(1, config.oc_size * block_dim))]) # [1, n * d], [1, oc_n * d], [1, oc_n * d]
        if config.residual_factor:
            self.residual_coeff = nn.Parameter(torch.ones(1))
        self.norm_eps = torch.tensor(config.norm_eps)
        self.residual_factor = config.residual_factor
        self.residual_norm = config.residual_norm

    
    def forward(
        self,
        x: torch.Tensor, # [t, n * d]
        attn_c: Optional[torch.Tensor], # [t, oc_n * d]
        block_pool: ParamBlockPool, 
        vector_pool: ParamVectorPool, 
        router_result: Optional[Dict[str, Dict[str, torch.Tensor]]], # dict or None, each [t * s_io * i, 1] * 2 or [t * f_n * k * i, 1] * 2
        rotary_emb: RotaryEmb, 
        block_mask: BlockMask,

    )-> Tuple[
        torch.Tensor, # x_output: [t, n * d]
        torch.Tensor, # attn_c_output: [t, oc_n * d]
        Optional[Dict[str, torch.Tensor]], # router_output: dict or None, each [t * s_io * i, 1] * 2 or [t * f_n * k * i, 1] * 2
        torch.Tensor, # balance_loss
        torch.Tensor, # router_z_loss
    ]:
        
        norm = torch.linalg.vector_norm(x, dim = -1, keepdim = True) + self.norm_eps # [t, 1]
        external, residual_norm_coeff = self.external(x, attn_c, norm) # external: dict or None, total [t, l_sum * v + f_n * f_v]  residual_norm_coeff: [t, 1] or None, l_sum = layer_size_sum
        x = (x / norm) * self.elementwise_coeffs[0] # [t, n * d] * [1, n * d] = [t, n * d]
        x_0, x_fixed_0 = {'input': x}, None
        device = x.device

        def forward_func(sublayer, x, x_fixed = None):
            if sublayer != 'ffn':
                return self.sublayers[sublayer](x, x_fixed, external, block_pool, vector_pool, router, router_result)
            else:
                return self.sublayers[sublayer](x, external, block_pool, vector_pool, router, router_result)
            
        output = [None for _ in range(5)]
        # output[i] = (x_i, (x_fixed_i), router_output_i, balance_loss_i, router_z_loss_i)
        # x_i: dict, each [t, pre_o_sum * d]
        # x_fixed_i: dict, each [t, m_f * d] or None
        # external: dict, total [t, l_sum * v + f_n * f_v] or None
        # router_result: dict, each [t * s_io * i, 1] * 2 or [t * f_n * k * i, 1] * 2 or None

        output[0] = forward_func('linear_1', x_0, x_fixed_0)

        x_1, x_fixed_1 = output[0][0], output[0][1]
        output[1] = forward_func('linear_2', x_1, x_fixed_1)

        x_2 = output[1][0]
        attn_output = self.attention(x_2, rotary_emb, block_mask)
        output[2] = forward_func('ffn', x_2)

        x_af, x_fixed_af = {'v_o': attn_output, 'ffn': output[2][0]}, None
        output[3] = forward_func('linear_3', x_af, x_fixed_af)

        x_3, coeffs_1, coeffs_2 = output[3][0], self.elementwise_coeffs[1], self.elementwise_coeffs[2]
        x_3 = x_3['v_oc'] * coeffs_1 + x_3['ffn_oc'] * coeffs_2
        x_3, x_fixed_3 = {'output_c': x_3}, output[3][1]
        if x_fixed_3:
            x_fixed_3 = {'output_c': torch.cat((x_fixed_3['v_oc'], x_fixed_3['ffn_oc']), 1)}
        output[4] = forward_func('linear_4', x_3, x_fixed_3)

        residual_coeff = self.residual_coeff if self.residual_factor else torch.tensor(1.0, device = device)
        if not self.residual_norm:
            residual_norm_coeff = torch.tensor(1.0, device = device)
        x_4, attn_c_output = output[4][0], x_3['v_oc']
        x_output = x * residual_norm_coeff * residual_coeff + x_4['output']
        router_output = {sublayer: output[i][-3] for i, sublayer in enumerate(self.sublayers)} if output[i][-3] else None
        balance_loss = sum(output[i][-2] for i in range(5))
        router_z_loss = sum(output[i][-1] for i in range(5))

        return x_output, attn_c_output, router_output, balance_loss, router_z_loss
    


class Model(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.main_dim = config.main_size * config.block_dim # n * d
        self.vocab_compressed = config.vocab_compressed
        self.emb_dim = config.vocab_c_dim if self.vocab_compressed else self.main_dim # vocab_c or n * d
        self.init_std = config.init_std
        self.return_logits = config.return_logits
        self.norm_eps = config.norm_eps
        self.window_len = config.window_len
        self.bias = config.bias

        self.balance_factor = config.balance_factor
        self.router_z_factor = config.router_z_factor
        self.vec_balance_factor = config.vec_balance_factor
        self.vec_router_z_factor = config.vec_router_z_factor

        self.rotary_emb = RotaryEmb(config)
        self.block_pool = ParamBlockPool(config)
        self.vector_pool = ParamVectorPool(config)
        self.model = nn.ModuleDict({
            'vocab_emb': nn.Embedding(config.vocab_size, self.emb_dim), # [vocab, vocab_c] or [vocab, n * d]
            'vocab_c_in': nn.Linear(self.emb_dim, self.main_dim, bias = False) if self.vocab_compressed else nn.Identity(), # [vocab_c, n * d]
            'layers': nn.ModuleList([Layer(config, i) for i in range(config.layer_num)]),
            'vocab_c_out': nn.Linear(self.main_dim, self.emb_dim, bias = False) if self.vocab_compressed else nn.Identity(), # [n * d, vocab_c]
            'lm_head': nn.Linear(self.emb_dim, config.vocab_size, bias = False), # [vocab_c, vocab] or [n * d, vocab]
        })

        if config.tied_vocab_emb:
            self.model.vocab_emb.weight = self.model.lm_head.weight
            
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.init_weights()

        self.index_num = 1 if config.no_paired else 2
        self.ffn_size = config.ffn_size
        self.ffn_top_k = config.ffn_top_k
        self.layer_num = config.layer_num - 2 if config.last_layer_vanilla else config.layer_num - 1
        self.m_io_num = config.m_io_num
        self.module_list = config.module_list
        self.linear_block_params = (config.block_dim / self.index_num + config.bias) * config.block_dim
        self.ffn_block_params = 3 * config.block_dim * config.ffn_hidden_dim + config.bias * config.block_dim
        self.tied_vocab_emb = config.tied_vocab_emb


    def init_weights(self) -> None:

        std = self.init_std

        for name, param in self.block_pool.named_parameters():
            if 'weight_pool' in name:
                nn.init.trunc_normal_(param, mean = 0, std = std, a = -3 * std, b = 3 * std, generator = self.g)

            if 'bias_pool' in name:
                nn.init.zeros_(param)

        for name, param in self.vector_pool.named_parameters():
            if name[-3:] == 'vec' or 'vec_clusters' in name:
                nn.init.trunc_normal_(param, mean = 0, std = std, a = -3 * std, b = 3 * std, generator = self.g)
            if 'gain_factors' in name:
                nn.init.ones_(param)
                
        for name, param in self.model.named_parameters():
            if name[-1] == 'b' in name:
                nn.init.zeros_(param)
            elif 'external_factors' in name:
                nn.init.zeros_(param)
            elif 'residual_factor' in name:
                nn.init.ones_(param)
            elif 'elementwise_coeffs' in name:
                nn.init.ones_(param)
            else:
                nn.init.trunc_normal_(param, mean = 0, std = std, a = -3 * std, b = 3 * std, generator = self.g)
                

    def params_count(self) -> Dict[str, int]:
        
        block_pool_params = sum(x.numel() for x in self.block_pool.parameters())
        vector_pool_params = sum(x.numel() for x in self.vector_pool.parameters())
        model_params = sum(x.numel() for x in self.model.parameters())
        emb_params = self.model.vocab_emb.weight.numel() * (2 - self.tied_vocab_emb)
        total_params = block_pool_params + vector_pool_params + model_params
        total_params_wo_emb = total_params - emb_params
        model_params_wo_emb = model_params - emb_params

        result = {
            'total_params': total_params, 
            'total_params_wo_emb': total_params_wo_emb, 
            'block_pool_params': block_pool_params, 
            'vector_pool_params': vector_pool_params, 
            'model_params': model_params, 
            'model_params_wo_emb': model_params_wo_emb, 
            'emb_params': emb_params,
        }
        
        return result
    

    def activated_params_count(self) -> Dict[str, int]:

        model_params = sum(x.numel() for x in self.model.parameters())
        emb_params = self.model.vocab_emb.weight.numel() * (2 - self.tied_vocab_emb)
        m_io_num_list = self.m_io_num
        m_io_num_list.insert(8, self.ffn_size * self.ffn_top_k)
        
        router_params, main_params = [], []
        for i, module in enumerate(self.module_list):
            vec_info = self.vector_pool.vec_info[module]
            vec_num_s, vec_num_se = vec_info['vec_num']
            vec_dim = vec_info['vec_dim']

            m_router_params = m_io_num_list[i] * (vec_num_s * vec_dim + vec_num_se * (vec_dim + 2 * self.index_num)) * self.layer_num
            router_params.append(int(m_router_params))

            block_params = self.ffn_block_params if i == 8 else self.linear_block_params
            m_main_params = m_io_num_list[i] * self.index_num * block_params * self.layer_num
            main_params.append(int(m_main_params))
        
        ffn_router_params = router_params[8]
        linear_router_params = sum(router_params) - ffn_router_params
        ffn_main_params = main_params[8]
        linear_main_params = sum(main_params) - ffn_main_params

        total_params = model_params + ffn_router_params + linear_router_params + ffn_main_params + linear_main_params
        total_params_wo_emb = total_params - emb_params
        model_params_wo_emb = model_params - emb_params

        result = {
            'total_params': total_params, 
            'total_params_wo_emb': total_params_wo_emb, 
            'ffn_router_params': ffn_router_params, 
            'linear_router_params': linear_router_params, 
            'ffn_main_params': ffn_main_params, 
            'linear_main_params': linear_main_params, 
            'model_params': model_params, 
            'model_params_wo_emb': model_params_wo_emb, 
            'emb_params': emb_params,
        }

        return result


    def forward(self, idx: torch.LongTensor, targets: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:

        docs = (idx == 50256).cumsum(0)
        
        def doc_causal_mask(b, h, q_idx, kv_idx):
            
            causal_mask = q_idx >= kv_idx
            doc_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < self.window_len
            return causal_mask & doc_mask & window_mask

        seq_len = len(idx)
        block_mask = create_block_mask(doc_causal_mask, None, None, seq_len, seq_len, device = 'cuda', _compile = True)
        
        vocab_emb = self.model.vocab_emb(idx) # [t] -> [t, vocab_c] or [t, n * d]
        
        if self.vocab_compressed:
            vocab_emb = self.model.vocab_c_in(vocab_emb) # [t, vocab_c] -> [t, n * d]
        x, attn_c, router_result = vocab_emb, None, None
        device = x.device

        balance_loss = torch.tensor(0, device = device)
        router_z_loss = torch.tensor(0, device = device)

        for layer in self.model.layers:
            x, attn_c, router_result, layer_balance_loss, layer_router_z_loss = layer(x, attn_c, self.block_pool, self.vector_pool, router_result, self.rotary_emb, block_mask)
            balance_loss += layer_balance_loss
            router_z_loss += layer_router_z_loss
            # x: [t, n * d]  attn_c: [t, oc_n * d] or None
        x = F.normalize(x, dim = -1, eps = self.norm_eps) # [t, n * d]
        
        if self.vocab_compressed:
            x = self.model.vocab_c_out(x) # [t, n * d] -> [t, vocab_c]

        logits = self.model.lm_head(x) # [t, n * d] or [t, vocab_c] -> [t, vocab]
        loss = F.cross_entropy(logits, targets)

        vec_balance_loss, vec_router_z_loss = self.vector_pool.get_vec_router_loss()
        auxiliary_loss = (
            self.balance_factor * balance_loss 
            + self.router_z_factor * router_z_loss 
            + self.vec_balance_factor * vec_balance_loss 
            + self.vec_router_z_factor * vec_router_z_loss
        )

        total_loss = loss + auxiliary_loss

        auxiliary_loss = {
            'balance_loss': balance_loss,
            'router_z_loss': router_z_loss,
            'vec_balance_loss': vec_balance_loss,
            'vec_router_z_loss': vec_router_z_loss,
        }

        if not self.return_logits:
            logits = None

        return logits, total_loss, auxiliary_loss


def configure_optimizers(model: Model, config: Config):

    param_dict = {name: param for name, param in model.named_parameters()}
    model_lr = config.model_lr
    pool_lr = config.pool_lr
    weight_decay = config.weight_decay
    betas = config.betas

    decay_model = [param for name, param in param_dict.items() if ('model' in name and param.shape[0] != 1)]
    no_decay_model = [param for name, param in param_dict.items() if ('model' in name and param.shape[0] == 1)]
    decay_pool = [param for name, param in param_dict.items() if 'weight_pool' in name]
    no_decay_pool = [param for name, param in param_dict.items() if 'vector_pool' in name or 'bias_pool' in name]
    
    if config.no_weight_decay_1d:
        optim_groups = [
            {'params': decay_model, 'lr': model_lr, 'weight_decay': weight_decay},
            {'params': no_decay_model, 'lr': model_lr, 'weight_decay': 0.0},
            {'params': decay_pool, 'lr': pool_lr, 'weight_decay': weight_decay},
            {'params': no_decay_pool, 'lr': pool_lr, 'weight_decay': 0.0},
        ]
    else:
        optim_groups = [
            {'params': decay_model + no_decay_model, 'lr': model_lr, 'weight_decay': weight_decay},
            {'params': decay_pool + no_decay_pool, 'lr': pool_lr, 'weight_decay': weight_decay},
        ]

    if config.zero_optimizer:
        optimizer = ZeroRedundancyOptimizer(optim_groups, optimizer_class = torch.optim.AdamW, betas = betas, fused = True)
    else:
        optimizer = torch.optim.AdamW(optim_groups, betas = betas, fused = True)

    return optimizer


def load_data_shard(filename: str, peek: bool = False):
    
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype = np.int32)
        if header[0] != 20240520:
            print0("magic number mismatch in the data .bin file", log_mode = False)
            exit(1)
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        if peek:
            return ntok
        tokens = np.frombuffer(f.read(), dtype = np.uint16)
        assert len(tokens) == ntok, "number of tokens read does not match header"
        return tokens
    

class DistributedDataLoader:
    
    def __init__(self, filename_pattern: str, seq_len: int, process_rank: int, num_processes: int) -> None:
        
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_len = seq_len
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = load_data_shard(fname, peek = True)
            assert shard_ntok >= num_processes * seq_len + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

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
        buf = self.tokens[self.current_pos : self.current_pos + self.seq_len + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype = torch.long)
        x = buf[:-1]
        y = buf[1:]
        self.current_pos += total_seq_len
        
        if self.current_pos + total_seq_len >= len(self.tokens):
            self.advance()
            
        return x, y
    

parser = argparse.ArgumentParser()
parser.add_argument('--input_bin', type = str, default = "/root/autodl-tmp/llm.c/dev/data/fineweb10B/fineweb_train_*.bin")
parser.add_argument('--input_val_bin', type = str, default = "/root/autodl-tmp/llm.c/dev/data/fineweb10B/fineweb_val_*.bin")
parser.add_argument('--accumulated_steps', type = int, default = 1)
parser.add_argument('--scheduler', type = str, default = 'wsd')
parser.add_argument("--seq_len", type = int, default = 64 * 1024)
parser.add_argument("--num_steps", type = int, default = 1000)
parser.add_argument("--warmup_steps", type = int, default = 50)
parser.add_argument("--cooldown_steps", type = int, default = 200)
parser.add_argument("--val_every", type = int, default = 100)
parser.add_argument("--val_steps", type = int, default = 20)
parser.add_argument("--overfit", type = bool, default = False)
parser.add_argument("--save_every", type = int, default = 0)
parser.add_argument("--save_before_cooldown", type = bool, default = True)
parser.add_argument("--log_freq", type = int, default = 20)

args = parser.parse_args()
for arg, value in vars(args).items():
    setattr(config, arg, value)

assert torch.cuda.is_available()
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    dist.init_process_group(backend = 'nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    config.zero_optimizer = False
    device = 'cuda'
    master_process = True

def print0(msg, print_mode: bool = True, log_mode: bool = True) -> None:
    if master_process:
        if print_mode:
            print(msg)
        if log_mode:
            with open(log_file, "a") as f:
                f.write(msg + '\n')

with open(sys.argv[0]) as f:
    code = f.read()
    
run_id = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
if master_process:
    log_dir = f'logs/{run_id}/'
    os.makedirs(log_dir, exist_ok = True)
    log_file = f'logs/{run_id}.txt'
    
    with open(log_file, "w") as f:
        f.write(code)
        f.write('=' * 100 + '\n')

print0(f"running pytorch {torch.version.__version__} compiled for cuda {torch.version.cuda}\nnvidia-smi:")
result = subprocess.run(['nvidia-smi'], stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
print0(f'{result.stdout}')
print0('=' * 100)

train_loader = DistributedDataLoader(config.input_bin, config.seq_len, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(config.input_val_bin, config.seq_len, ddp_rank, ddp_world_size)

step_tokens = ddp_world_size * config.seq_len
accumulated_step_tokens = step_tokens * config.accumulated_steps
assert val_loader.ntok_total >= config.val_steps * step_tokens, "validation tokens not enough"
print0(f"step_tokens: {step_tokens} accumulated_step_tokens: {accumulated_step_tokens}")
print0(f"training dataloader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"validation dataloader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('=' * 100)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = Model(config).to(device)
model.train()
amp_ctx = torch.autocast(device_type = device, dtype = torch.bfloat16)
scaler = torch.amp.GradScaler()
torch.backends.cuda.matmul.allow_tf32 = True

if config.model_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model
optimizer = configure_optimizers(raw_model, config)
original_lr = []
for param_group in optimizer.param_groups:
    original_lr.append(param_group['lr'])

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

if config.scheduler == 'wsd':
    get_lr = get_lr_wsd
if config.scheduler == 'cos':
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
train_log_file = f'logs/{run_id}/train_log.csv'
train_layer_log = []
train_layer_log_file = f'logs/{run_id}/train_layer_log.csv'

t0 = time.time()

for step in range(num_steps + 1):

    if step == 0:
        raw_model.vector_pool.get_device()
        if ddp:
            raw_model.vector_pool.gather_list_init(ddp_world_size)
            raw_model.vector_pool.cand_num_list_init(ddp_world_size)
        raw_model.vector_pool.vec_init(ddp_local_rank, ddp_world_size, ddp)
        if config.clear_gather_list:
            raw_model.vector_pool.gather_list = []
        raw_model.vector_pool.cand_list = []
    

    last_step = (step == num_steps)
    if (last_step or (config.val_every > 0 and step % config.val_every == 0)):
        
        torch.cuda.synchronize()
        model.eval()
        val_loader.reset()
        val_loss = torch.tensor(0.0)
        with torch.no_grad():
            for _ in range(config.val_steps):
                x_val, y_val = val_loader.next_batch()
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, loss, _ = model(x_val, y_val)
                val_loss += loss
        dist.all_reduce(val_loss, op = dist.ReduceOp.AVG)
        val_loss /= config.val_steps
        
        print0(f'step:{step}/{num_steps} val_loss:{val_loss.item():.4f}')

        model.train()
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (config.save_before_cooldown and step == cooldown_steps) or (config.save_every > 0 and step % config.save_every == 0)):

        torch.cuda.synchronize()
        
        checkpoint = {
            'model': raw_model,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'logs/{run_id}/checkpoint/step_{step}.pt')
        
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    if config.overfit:
        train_loader.reset()

    train_loss = torch.tensor(0.0) 
    for micro_step in range(config.accumulated_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        sync_ctx = model.no_sync() if micro_step < config.accumulated_steps - 1 else nullcontext()
        with sync_ctx:
            with amp_ctx:

                if step == 10 or step % 100 == 0:
                    with profile(activities = [ProfilerActivity.CUDA], record_shapes = True, profile_memory = True, with_stack = True) as prof:
                        with record_function('model_inference'):
                            _, loss, a_loss = model(x, y)
                            print0(prof.key_averages().table(sort_by = 'cuda_time_total', row_limit = 20))
                            print0(prof.key_averages(group_by_input_shape = True).table(sort_by = 'cuda_time_total', row_limit = 20))
                            print0(prof.key_averages().table(sort_by = 'self_cuda_memory_usage', row_limit = 20))
                            print0(prof.key_averages().table(sort_by = 'cuda_memory_usage', row_limit = 20))
                            prof.export_chrome_trace(f'logs/{run_id}/trace/step_{step}.json')
                else:
                    _, loss, a_loss = model(x, y)
                loss = loss / config.accumulated_steps
                train_loss += loss.detach()
                if config.scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

    lr = get_lr(step)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * original_lr[i]
        
    if config.scaler:
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
    else:
        if config.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
        optimizer.step()
    optimizer.zero_grad(set_to_none = True)
    if ddp:
        dist.all_reduce(train_loss, op = dist.ReduceOp.AVG)

    if step % config.training_update_freq == 0 and step < int(config.update_cutoff_point * num_steps):

        if ddp and config.clear_gather_list:
            raw_model.vector_pool.gather_list_init(ddp_world_size)
        raw_model.vector_pool.training_update(ddp_local_rank, ddp_world_size, ddp)
        if config.clear_gather_list:
            raw_model.vector_pool.gather_list = []
        raw_model.vector_pool.cand_list = []

    torch.cuda.synchronize()

    if master_process:

        time_ms = 1000 * (time.time() - t0)
        running_avg_time = time_ms if step < 10 else 0.8 * running_avg_time + 0.2 * time_ms
        total_time_ms += time_ms
        tokens_per_second = accumulated_step_tokens / time_ms * 1000
        running_avg_tokens = tokens_per_second if step < 10 else 0.8 * running_avg_tokens + 0.2 * tokens_per_second

        a_loss_list = [loss.item() for loss in a_loss.values()]
        vec_info_list = raw_model.vector_pool.get_info()
        vec_len = len(vec_info_list)
        cand_p_avg = sum(vec_info_list[0]) / vec_len
        max_c_vec_p_avg = sum(vec_info_list[1]) / vec_len
        min_c_vec_p_avg = sum(vec_info_list[2]) / vec_len

        residual_coeff_list = []
        residual_norm_coeff_list = []
        layers = raw_model.model['layers']

        for i in range(config.layer_num):
            if config.residual_factor:
                residual_coeff_list.append(layers[i].residual_coeff.item())
            if config.residual_norm:
                residual_norm_coeff_list.append(layers[i].external.external_factors[0, -1].item())
        residual_coeff_avg = sum(residual_coeff_list) / len(residual_coeff_list)
        residual_norm_coeff_avg = sum(residual_norm_coeff_list) / len(residual_norm_coeff_list)
        
        sigmoid_factors = torch.zeros(5)
        for module in config.module_list:
            sigmoid_factors += raw_model.vector_pool.vec[module]['sigmoid_factors'].mean(0)
        sigmoid_factors /= len(config.module_list)
        sigmoid_factors = [factor.item() for factor in sigmoid_factors]
        
        print0(f'step:{step}/{num_steps} train_loss:{train_loss.item():.4f} time:{total_time_ms / 1000:.0f}ms step_time:{running_avg_time:.1f}ms {running_avg_tokens:.0f} t/s norm:{grad_norm:.3f} a_loss:({a_loss_list[0]:.3f}, {a_loss_list[1]:.3f}, {a_loss_list[2]:.3f}, {a_loss_list[3]:.3f}) vec_info: ({cand_p_avg:.3f}, {max_c_vec_p_avg:.3f}, {min_c_vec_p_avg:3f}) coeff: ({residual_coeff_avg:.3f}, {residual_norm_coeff_avg:.3f})')

        train_log.append([step, train_loss.item(), total_time_ms / 1000, running_avg_time, running_avg_tokens, grad_norm, cand_p_avg, max_c_vec_p_avg, min_c_vec_p_avg, residual_coeff_avg, residual_norm_coeff_avg] + a_loss_list + sigmoid_factors)

        train_layer_log.append(vec_info_list[0] + [0] + vec_info_list[1] + [0] + vec_info_list[2] + [0] + residual_coeff_list + [0] + residual_norm_coeff_list)

        if step % config.log_freq == 0:

            df = pd.DataFrame(train_log, columns = ['step', 'train_loss', 'total_time(s)', 'step_time(ms)', 'token_speed(t/s)', 'grad_norm', 'cand_p_avg', 'max_c_vec_p_avg', 'min_c_vec_p_avg', 'residual_coeff_avg', 'residual_norm_coeff_avg', 'balance_loss', 'router_z_loss', 'vec_balance_loss', 'vec_router_z_loss', 'sigmoid_factors_1', 'sigmoid_factors_2', 'sigmoid_factors_3', 'sigmoid_factors_4', 'sigmoid_factors_5'])

            df_layer = pd.DataFrame(train_layer_log)

            if step == 0:
                df.to_csv(train_log_file, index = False)
            else:
                df.to_csv(train_log_file, mode = 'a', header = False, index = False)

            df_layer.to_csv(train_layer_log,  mode = 'a', header = False, index = False)

            train_log = []
            train_layer_log = []

    torch.cuda.synchronize()
    t0 = time.time()

print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} mib")
if ddp:
    dist.destroy_process_group()
