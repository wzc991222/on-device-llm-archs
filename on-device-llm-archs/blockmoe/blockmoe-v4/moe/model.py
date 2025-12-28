import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
from einops import rearrange
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts as checkpoint_contexts,
)
import functools
import os
import torch.distributed as dist

flex_attention = torch.compile(flex_attention, dynamic=False)
from config import Config
from utils import GroupedGEMM, grouped_gemm_func_original
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
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
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


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        dim_s = config.shared_expert_dim
        dim_e = config.expert_dim
        en = config.routed_expert_num
        self.dim = dim
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_experts = nn.Parameter(torch.randn(3, en, dim, dim_e))
        self.keys_router = nn.Parameter(torch.randn(dim, en))
        self.register_buffer("stat", torch.zeros(en))
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("logits_max", torch.tensor(0, dtype=torch.float32))
        self.register_buffer("logits_mean", torch.tensor(0, dtype=torch.float32))
        if config.keys_gate:
            self.keys_gate = nn.Parameter(torch.randn(dim, en))
        if config.keys_bias:
            self.keys_bias = nn.Parameter(torch.zeros(en))
        if config.shared_expert:
            self.ffn_up = nn.Linear(dim, dim_s * 2, bias=False)
            self.ffn_down = nn.Linear(dim_s, dim, bias=False)
            if config.expert_post_norm:
                self.shared_expert_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        if config.expert_post_norm:
            self.output_coeff = nn.Parameter(torch.ones(1, 1, dim))

    def reset(self):
        self.stat -= self.stat
        self.count -= self.count
        self.logits_max -= self.logits_max
        self.logits_mean -= self.logits_mean

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

        et = config.expert_num_per_token
        router_logits = (x_ffn @ self.keys_router).flatten(0, 1)
        router_values, indices = router_logits.topk(et, -1, sorted=False)
        values = router_values
        if config.keys_gate:
            gate_logits = (x_ffn @ self.keys_gate).flatten(0, 1)
            gate_values = torch.gather(gate_logits, -1, indices)
            values = values + gate_values
        if config.keys_bias:
            values = values + self.keys_bias[indices]

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
            dim_s = config.shared_expert_dim
            x1, x2 = torch.split(self.ffn_up(x_ffn), dim_s, -1)
            x3 = F.silu(x1) * x2
            y_shared = self.ffn_down(x3)
            if config.expert_post_norm:
                y_shared = self.shared_expert_norm(y_shared)
            y = y + y_shared

        y = y + x_ffn_input

        en = config.routed_expert_num
        indices_bin_o = F.one_hot(indices, en).type_as(x).mean(0) * en
        self.stat += indices_bin_o
        self.count += 1
        self.logits_max += router_logits.amax(-1).mean().detach()
        self.logits_mean += router_logits.mean().detach()
        ddp = int(os.environ.get("RANK", -1)) != -1
        if ddp:
            dist.all_reduce(self.stat, op=dist.ReduceOp.AVG)
        indices_bin = self.stat / self.count
        values_bin = router_logits.softmax(-1).mean(0)
        balance_loss = (indices_bin * values_bin).sum()

        return y, balance_loss
    

class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        attn_dim = config.main_attn_dim
        an = config.main_attn_num
        ln = config.moe_layer_num
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense

        self.vocab_emb = nn.Embedding(config.vocab_size, dim)
        if first_k > 0:
            self.first_dense_layers = nn.ModuleList(
                [Layer(config) for _ in range(first_k)]
            )
        if ln > 0:
            self.moe_layers = nn.ModuleList([MoELayer(config) for _ in range(ln)])
        if last_k > 0:
            self.last_dense_layers = nn.ModuleList(
                [Layer(config) for _ in range(last_k)]
            )
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)
        self.final_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        input: torch.LongTensor,
        target: torch.LongTensor,
        config: Config,
    ):
        b, s = input.shape
        device = input.device
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense
        ln = config.moe_layer_num
        eot = input == config.eot_idx
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = doc[b, q_idx] == doc[b, kv_idx]
            return causal_mask & doc_mask
        
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        x = self.vocab_emb(input)
        if first_k > 0:
            for layer in self.first_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        balance_loss = torch.tensor(0, device=device)
        if ln > 0:
            for layer in self.moe_layers:
                x, layer_b_loss = layer(x, block_mask, self.attention, config)
                balance_loss = balance_loss + layer_b_loss

        if last_k > 0:
            for layer in self.last_dense_layers:
                x = layer(x, block_mask, self.attention, config)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        main_loss = F.cross_entropy(logits.flatten(0, -2), target.flatten())
        balance_loss = balance_loss * config.balance_loss_factor
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
        total_params = sum(x.numel() for x in self.parameters())
        ffn_experts_not_a_params = (
            3
            * config.main_dim
            * config.expert_dim
            * (config.routed_expert_num - config.expert_num_per_token)
            * config.moe_layer_num
        )
        total_a_params = total_params - ffn_experts_not_a_params
        params_result = {
            "total_params": total_params,
            "total_a_params": total_a_params,
        }
        return params_result
    
    def get_optimizer(self, config: Config):
        lr = config.max_lr
        wd = config.weight_decay
        betas = config.adam_betas
        params = [param for param in self.parameters()]
        param_groups = [
            {"params": params, "lr": lr, "weight_decay": wd},
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