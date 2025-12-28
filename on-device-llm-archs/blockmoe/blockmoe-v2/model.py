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
        eps = config.norm_eps
        xq = rearrange(xq, "b s (h a) -> b h s a", h=self.attn_num)
        xk = rearrange(xk, "b s (h a) -> b h s a", h=self.attn_num)
        xv = rearrange(xv, "b s (h a) -> b h s a", h=self.attn_num)
        if config.qk_norm:
            xq = F.normalize(xq, dim=-1, eps=eps).type_as(xv)
            xk = F.normalize(xk, dim=-1, eps=eps).type_as(xv)
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)
        kernel_options = config.kernel_options if config.flex_attn_spec_kernel else None
        x_attn = flex_attention(
            xq,
            xk,
            xv,
            block_mask=block_mask,
            enable_gqa=True,
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
        doc: torch.Tensor,
        config: Config,
    ):
        n = config.block_size

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // n
            kv_b_idx = kv_idx // n
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & doc_mask

        b, s = x.shape[:2]
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
        self.router_token = nn.Parameter(torch.randn(1, config.router_size, dim))
        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(layer_num)])
        self.ouput = nn.Linear(dim, config.aux_output_dim, bias=False)
        self.attention = Attention(RotaryEmb(attn_dim, config.theta_s), an)

    def forward(
        self,
        x: torch.Tensor,
        config: Config,
        next: bool,
    ):
        n = config.block_size
        d = config.aux_dim
        a = config.aux_attn_num
        ln = config.moe_layer_num
        hn = config.aux_router_head_num
        re = config.aux_router_expert_num
        dr = config.aux_router_dim
        dg = config.aux_gate_dim
        l = config.block_len

        x = rearrange(x, "b (l n) d -> (b l) n d", n=n)
        router_token = self.router_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, router_token], 1)

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            return causal_mask
        # TODO

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        x = self.ouput(x[:, n:])

        xr, xg = x.split([dr, dg], -1)
        xr = rearrange(
            xr, "(b l) (ln re hn) dr -> ln b l re (hn dr)", l=l, ln=ln, re=re
        )
        xg = rearrange(
            xg, "(b l) (ln re hn) dg -> ln b l re (hn dg)", l=l, ln=ln, re=re
        )

        if not next:
            xr = xr.roll(1, 2)
            xg = xg.roll(1, 2)
        xr = xr.flatten(1, 3)
        xg = xg.flatten(1, 3)

        return xr, xg


class AuxModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        encoder_ln, next_encoder_ln = config.aux_encoder_layer_num
        router_ln, next_router_ln = config.aux_router_layer_num
        ln = config.moe_layer_num
        hn = config.aux_router_head_num
        en = config.routed_expert_num
        dr = config.aux_router_dim
        dg = config.aux_gate_dim

        self.aux_encoder = AuxEncoder(config, encoder_ln)
        self.aux_router = AuxRouter(config, router_ln)
        self.keys_router = nn.Parameter(torch.randn(ln, hn * dr, en))
        self.keys_gate = nn.Parameter(torch.randn(ln, hn * dg, en))

        if config.aux_pred:
            self.next_aux_encoder = AuxEncoder(config, next_encoder_ln)
            self.next_aux_router = AuxRouter(config, next_router_ln)

        if config.balance_loss_free:
            self.register_buffer("bias", torch.zeros(ln, 1, en))
            self.register_buffer("stat", torch.ones(ln * en))

    def forward(
        self,
        x: torch.Tensor,
        aux_mask: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        x_router, x_gate = self.aux_router(
            self.aux_encoder(x, doc, config), config, False
        )
        x_router_logits = torch.bmm(x_router, self.keys_router)
        x_gate_logits = torch.bmm(x_gate, self.keys_gate)

        router_logits, gate_logits = x_router_logits, x_gate_logits
        if config.aux_pred:
            y_router, y_gate = self.next_aux_router(
                self.next_aux_encoder(x, doc, config), config, True
            )
            y_router_logits = torch.bmm(y_router, self.keys_router)
            y_gate_logits = torch.bmm(y_gate, self.keys_gate)
            router_logits, gate_logits = y_router_logits, y_gate_logits

        router_logits_m = router_logits
        if config.balance_loss_free:
            router_logits_m = router_logits + self.bias

        et = config.expert_num_per_token
        if config.aux_expert_wise_router:
            indices = router_logits_m.max(-1, keepdim=True)[1]
        else:
            indices = router_logits_m.topk(et, -1)[1]

        router_values = torch.gather(router_logits, -1, indices).flatten(1)
        gate_values = torch.gather(gate_logits, -1, indices).flatten(1)
        values = router_values + gate_values
        indices = indices.flatten(1)

        aux_pred_loss = 0
        if config.aux_pred:
            x_factor, y_factor = config.aux_pred_loss_factor
            if config.aux_pred_loss_func == "cross_entropy":
                xr_logits_f = x_router_logits.flatten(0, 1)
                xr_softmax = xr_logits_f.softmax(-1)
                yr_logits_f = y_router_logits.flatten(0, 1)
                yr_softmax = yr_logits_f.softmax(-1)
                x_aux_pred_loss = F.cross_entropy(
                    xr_logits_f, yr_softmax.detach(), reduction="none"
                )
                y_aux_pred_loss = F.cross_entropy(
                    yr_logits_f, xr_softmax.detach(), reduction="none"
                )
            if config.aux_pred_loss_func == "mse":
                xr_fn = F.normalize(x_router.flatten(0, 1), dim=-1)
                yr_fn = F.normalize(y_router.flatten(0, 1), dim=-1)
                x_aux_pred_loss = F.mse_loss(
                    xr_fn, yr_fn.detach(), reduction="none"
                ).sum(-1)
                y_aux_pred_loss = F.mse_loss(
                    yr_fn, xr_fn.detach(), reduction="none"
                ).sum(-1)

            x_aux_pred_loss = (
                (x_aux_pred_loss * aux_mask).sum() / aux_mask.sum() * x_factor
            )
            y_aux_pred_loss = (
                (y_aux_pred_loss * aux_mask).sum() / aux_mask.sum() * y_factor
            )
            aux_pred_loss = x_aux_pred_loss + y_aux_pred_loss

        balance_loss = 0
        ln = config.moe_layer_num
        en = config.routed_expert_num
        df = config.balance_decay_factor
        indices_m = indices + torch.arange(ln, device=indices.device).unsqueeze(-1) * en
        indices_b = torch.bincount(indices_m.flatten(), minlength=ln * en)
        if config.balance_loss_free:
            indices_bn = indices_b / indices.shape[1] * en
            self.stat = self.stat * df + indices_bn * (1 - df)
            error = rearrange(self.stat - 1, "(ln en) -> ln 1 en", ln=ln)
            with torch.no_grad():
                self.bias -= error * config.balance_update_rate
        else:
            router_s = router_logits.softmax(-1).sum(1).flatten()
            balance_loss = (
                torch.dot(indices_b.type_as(router_s), router_s)
                / (indices.shape[1] * router_logits.shape[1])
                * en
                * config.balance_loss_factor
            )

        return indices, values, aux_pred_loss, balance_loss


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        dim_s = config.shared_expert_dim
        ten = config.total_layer_expert_num
        self.dim = dim
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_experts = nn.Parameter(torch.randn(3, ten, dim, config.expert_dim))
        self.keys = nn.Parameter(torch.randn(dim, ten))
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

        token_values = (x_ffn @ self.keys).flatten(0, 1)
        token_values = torch.gather(token_values, -1, indices)
        values = values + token_values
        if config.gate_func == "softmax":
            scores = values.softmax(-1)
        if config.gate_func == "sigmoid":
            scores = values.sigmoid()
        if config.gate_func == "sigmoid_norm":
            scores = values.sigmoid()
            scores = values / values.sum(-1, keepdim=True)
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
        l = config.moe_layer_num
        k = config.first_k_layer_dense
        an = config.main_attn_num

        if k > 0:
            self.dense_layers = nn.ModuleList([Layer(config, dim) for _ in range(k)])
        self.moe_layers = nn.ModuleList([MoELayer(config) for _ in range(l)])
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        main_doc: torch.Tensor,
        indices: torch.LongTensor,
        values: torch.Tensor,
        config: Config,
    ):
        k = config.first_k_layer_dense

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            doc_mask = main_doc[b, q_idx] == main_doc[b, kv_idx]
            return causal_mask & doc_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        if k > 0:
            for layer in self.dense_layers:
                x = layer(x, block_mask, self.attention, config)
        for i, layer in enumerate(self.moe_layers):
            x = layer(x, indices[i], values[i], block_mask, self.attention, config)
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
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        self.init_weights(config)

    def forward(
        self,
        tokens: torch.LongTensor,
        config: Config,
    ):
        b, s = tokens.shape
        n = config.block_size
        ln = config.moe_layer_num
        re = config.aux_router_expert_num
        et = config.expert_num_per_token
        l = config.block_len

        assert (s - 1) % n == 0
        input = rearrange(tokens[:, :-1], "b (l n) -> b l n", n=n)
        eot = (input == config.eot_idx).any(-1)
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        main_doc = repeat(doc, "b l -> b (l n)", n=n)
        pad = input == config.pad_idx
        mask = (~eot).roll(1, 1)
        mask[:, 0] = False
        aux_mask = repeat(mask, "b l -> (ln b l re)", ln=ln, re=re)
        router_mask = repeat(mask, "b l -> ln (b l et)", ln=ln, et=et)

        x = self.vocab_emb(input.flatten(1))
        x_aux = self.main_to_aux(x)
        indices, values, aux_pred_loss, balance_loss = self.aux_model(
            x_aux, aux_mask, doc, config
        )

        bos_indices = torch.arange(
            config.routed_expert_num,
            config.total_layer_expert_num,
            device=indices.device,
        )
        bos_indices = repeat(bos_indices, "et -> ln (bl et)", ln=ln, bl=b * l)
        indices = torch.where(router_mask, indices, bos_indices)
        values = torch.where(router_mask, values, 0)
        indices = repeat(indices, "ln (bl et) -> ln (bl n) et", n=n, et=et)
        values = repeat(values, "ln (bl et) -> ln (bl n) et", n=n, et=et)

        y = self.main_model(x, main_doc, indices, values, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(
            logits.flatten(0, -2), tokens[:, 1:].flatten(), reduction="none"
        )
        main_mask = ~pad.flatten()
        main_loss = (main_loss * main_mask).sum() / main_mask.sum()
        loss = main_loss + aux_pred_loss + balance_loss
        token_num = b * s - pad.sum()
        info = torch.tensor([main_loss, aux_pred_loss, balance_loss, token_num])
        return loss, info

    def init_weights(self, config: Config):
        std = config.init_std
        g = torch.Generator()
        g.manual_seed(42)

        for name, param in self.named_parameters():
            if "norm" in name:
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
        if config.aux_pred:
            next_aux_router_params = sum(
                x.numel() for x in self.aux_model.next_aux_router.parameters()
            )
            aux_router_params += next_aux_router_params
        aux_model_a_params = aux_model_params + int(
            aux_router_params * config.router_size / config.block_size
        )
        en = config.total_layer_expert_num - config.expert_num_per_token
        ffn_experts_not_a_params = (
            3 * config.main_dim * config.expert_dim * en * config.moe_layer_num
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
        lr = config.max_lr
        wd = config.weight_decay
        betas = config.adam_betas
        no_wd_params = [
            param
            for name, param in self.named_parameters()
            if "norm" in name or "vocab_emb" in name or "lm_head" in name
        ]
        wd_params = [
            param
            for name, param in self.named_parameters()
            if "norm" not in name and "vocab_emb" not in name and "lm_head" not in name
        ]
        param_groups = [
            {"params": wd_params, "lr": lr, "weight_decay": wd},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0},
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
