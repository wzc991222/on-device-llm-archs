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
        self.router_token = nn.Parameter(torch.randn(1, config.router_size, dim))
        self.layers = nn.ModuleList([Layer(config, dim) for _ in range(layer_num)])
        self.ouput = nn.Linear(dim, config.aux_output_dim, bias=False)
        self.attention = Attention(RotaryEmb(attn_dim, config.theta), an)

    def forward(
        self,
        x: torch.Tensor,
        config: Config,
    ):
        n = config.block_size
        rm = config.group_router_mul_factor
        dr = config.aux_router_dim
        dg = config.aux_gate_dim
        l = config.block_len

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
        xr = repeat(xr, "(b l) r dr -> (r rm) b l dr", l=l, rm=rm)
        xg = repeat(xg, "(b l) r dg -> (r rm) b l dg", l=l, rm=rm)
        xr = xr.roll(1, 2).flatten(1, 2)
        xg = xg.roll(1, 2).flatten(1, 2)

        return xr, xg


class AuxModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        encoder_ln = config.aux_encoder_layer_num
        router_ln = config.aux_router_layer_num
        ln = config.moe_layer_num
        en = config.routed_expert_num
        an = config.actual_group_expert_num
        gn = config.router_group_num
        dr = config.aux_router_dim
        dg = config.aux_gate_dim
        gc = gn if config.group_wise_coeff else 1

        self.aux_encoder = AuxEncoder(config, encoder_ln)
        self.aux_router = AuxRouter(config, router_ln)
        self.keys_router = nn.Parameter(torch.randn(ln * gn, dr, an))
        self.keys_gate = nn.Parameter(torch.randn(ln * gn, dg, an))

        if config.norm_routing:
            self.router_coeff = nn.Parameter(torch.ones(ln * gc, 1, 1))
            self.gate_coeff = nn.Parameter(torch.ones(ln * gc, 1, 1))
        if config.balance_loss_free:
            self.register_buffer("bias", torch.zeros(ln * en))
            self.register_buffer("stat", torch.ones(ln * en))

    def update(self, config: Config):
        ln = config.moe_layer_num
        error = self.stat - 1
        self.bias -= error * config.balance_update_rate
        stat = rearrange(self.stat, "(ln en) -> ln en", ln=ln)
        print(f"stat: {stat[4]}")
        self.stat -= self.stat

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
        aux_mode: bool,
    ):
        gt = config.group_expert_num_per_token
        ln = config.moe_layer_num
        en = config.routed_expert_num
        gn = config.router_group_num
        ge = config.group_expert_num
        nv = config.router_norm_value
        nf = config.router_norm_factor
        gm = 1 if config.group_wise_coeff else gn
        coeff = config.balance_loss_coeff
        eps = config.norm_eps
        aa = config.aux_accumulated_steps
        device = x.device
        global_attn = True if config.aux_encoder_global_attn and not aux_mode else False

        x = self.aux_encoder(x, doc, config, global_attn)
        x_router, x_gate = self.aux_router(x, config)
        keys_router = self.keys_router
        keys_gate = self.keys_gate

        if config.norm_routing:
            x_router = F.normalize(x_router, dim=-1)
            x_gate = F.normalize(x_gate, dim=-1)
            keys_router = F.normalize(keys_router, dim=1)
            keys_gate = F.normalize(keys_gate, dim=1)

        router_logits = torch.bmm(x_router, keys_router)
        router_logits_m = router_logits
        if config.balance_loss_free:
            if config.sole_group_wise_router:
                bias = rearrange(self.bias, "(c ge) -> c 1 ge", ge=ge)
            else:
                bias = repeat(self.bias, "(ln en) -> (ln gn) 1 en", ln=ln, gn=gn)
            router_logits_m = router_logits + bias

        indices_o = router_logits_m.topk(gt, -1, sorted=False)[1]
        indices = rearrange(indices_o, "(ln gn) bl gt -> ln bl gn gt", ln=ln)
        if config.sole_group_wise_router:
            offset = torch.arange(gn, device=device)[None, None, :, None] * ge
            indices = indices + offset
        indices = indices.flatten(1)
        router_values_o = torch.gather(router_logits, -1, indices_o)

        balance_loss = 0
        if aux_mode:
            indices_bin = F.one_hot(indices, en).type_as(x).mean(1).flatten()
            if config.sole_group_wise_router:
                values_aux = rearrange(
                    router_logits, "(ln gn) bl ge -> ln bl (gn ge)", ln=ln
                )
            else:
                values_aux = rearrange(
                    router_logits, "(ln gn) bl en -> ln (bl gn) en", ln=ln
                )
            print(
                f"values_aux max: {values_aux.amax(-1).mean()}, mean: {values_aux.mean()}"
            )
            if config.balance_loss_free:
                self.stat += indices_bin * en / aa
            else:
                values_bin = (values_aux * coeff).softmax(-1).mean(1).flatten()
                balance_loss = (
                    (indices_bin * values_bin).sum() * en * config.balance_loss_factor
                )
                indices_bin = rearrange(indices_bin, "(ln en) -> ln en", ln=ln)
                print(f"indices_bin: {indices_bin[4]}")
                print(indices_bin.amax(1).mean())
            return balance_loss

        gate_logits = torch.bmm(x_gate, keys_gate)
        gate_values = torch.gather(gate_logits, -1, indices_o)
        if config.norm_routing:
            router_coeff = repeat(self.router_coeff, "c1 c2 c3 -> (c1 gm) c2 c3", gm=gm)
            gate_coeff = repeat(self.gate_coeff, "c1 c2 c3 -> (c1 gm) c2 c3", gm=gm)
            values = router_values_o * router_coeff + gate_values * gate_coeff
        else:
            values = router_values_o + gate_values
        values = rearrange(values, "(ln gn) bl gt -> ln (bl gn gt)", ln=ln)

        norm_loss = 0
        if config.norm_routing and nf > 0:
            router_norm = self.keys_router.norm(dim=1)
            gate_norm = self.keys_gate.norm(dim=1)
            router_norm_loss = router_norm / nv + nv / (router_norm + eps) - 2
            gate_norm_loss = gate_norm / nv + nv / (gate_norm + eps) - 2
            norm_loss = router_norm_loss.sum() + gate_norm_loss.sum()

        return indices, values, norm_loss


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.main_dim
        dim_s = config.shared_expert_dim
        te = config.total_layer_expert_num
        gn = config.router_group_num
        gc = gn if config.group_wise_coeff else 1
        self.dim = dim
        self.dim_coeff = dim**0.5
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm_coeff = nn.Parameter(torch.ones(1, 1, dim))
        self.ffn_experts = nn.Parameter(torch.randn(3, te, dim, config.expert_dim))
        self.keys = nn.Parameter(torch.randn(dim, te))
        if config.norm_routing:
            self.router_coeff = nn.Parameter(torch.ones(1, gc))
            self.bias = nn.Parameter(torch.zeros(1, gc))
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
        x_ffn_norm = F.normalize(x_ffn_input, dim=-1)
        x_ffn = x_ffn_norm * self.ffn_norm_coeff * self.dim_coeff

        x_ffn_router = x_ffn_norm if config.norm_routing else x_ffn
        keys = F.normalize(self.keys, dim=0) if config.norm_routing else self.keys
        token_values = (x_ffn_router @ keys).flatten(0, 1)
        token_values = torch.gather(token_values, -1, indices)
        if config.norm_routing:
            gt = config.group_expert_num_per_token
            gn = config.router_group_num
            gm = gt if config.group_wise_coeff else gn * gt
            router_coeff = repeat(self.router_coeff, "c1 c2 -> c1 (c2 gm)", gm=gm)
            bias = repeat(self.bias, "c1 c2 -> c1 (c2 gm)", gm=gm)
            values = values + token_values * router_coeff + bias
        else:
            values = values + token_values

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
        first_k = config.first_k_layer_dense
        last_k = config.last_k_layer_dense
        an = config.main_attn_num

        if first_k > 0:
            self.first_dense_layers = nn.ModuleList(
                [Layer(config, dim) for _ in range(first_k)]
            )
        self.moe_layers = nn.ModuleList([MoELayer(config) for _ in range(l)])
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
            x = layer(x, indices[i], values[i], block_mask, self.attention, config)

        if last_k > 0:
            for layer in self.last_dense_layers:
                x = layer(x, block_mask, self.attention, config)

        nf = config.router_norm_factor
        nv = config.router_norm_value
        eps = config.norm_eps
        norm_loss = 0
        if config.norm_routing and nf > 0:
            keys_list = []
            for layer in self.moe_layers:
                keys_list.append(layer.keys)
                keys = torch.cat(keys_list, -1)
                keys_norm = keys.norm(dim=0)
                norm_loss = keys_norm / nv + nv / (keys_norm + eps) - 2
                norm_loss = norm_loss.sum()

        return x, norm_loss


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
        input: torch.LongTensor,
        target: torch.LongTensor,
        config: Config,
        aux_mode: bool = False,
    ):
        b, s = input.shape
        n = config.block_size
        ln = config.moe_layer_num
        an = config.actual_group_expert_num
        gn = config.router_group_num
        te = config.total_layer_expert_num
        et = config.expert_num_per_token
        l = config.block_len

        eot = input == config.eot_idx
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        eot = rearrange(eot, "b (l n) -> b l n", n=n).any(-1)
        mask = eot | eot.roll(1, 1)
        mask[:, 0] = True
        mask = repeat(mask, "b l -> ln (b l et)", ln=ln, et=et)

        x = self.vocab_emb(input)
        x_aux = self.main_to_aux(x)

        if aux_mode:
            balance_loss = self.aux_model(x_aux, doc, config, aux_mode)
            return balance_loss
        else:
            indices, values, aux_norm_loss = self.aux_model(
                x_aux, doc, config, aux_mode
            )

        default_idx = torch.arange(
            config.routed_expert_num,
            config.total_layer_expert_num,
            device=indices.device,
        )
        default_idx = repeat(default_idx, "et -> ln (bl et)", ln=ln, bl=b * l)
        indices = torch.where(mask, default_idx, indices)
        values = torch.where(mask, self.default_bias, values)
        indices = repeat(indices, "ln (bl et) -> ln (bl n) et", n=n, et=et)
        values = repeat(values, "ln (bl et) -> ln (bl n) et", n=n, et=et)

        y, main_norm_loss = self.main_model(x, doc, indices, values, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(logits.flatten(0, -2), target.flatten())
        norm_loss = (
            (main_norm_loss + aux_norm_loss)
            / (2 * ln * gn * an + ln * te)
            * config.router_norm_factor
        )
        loss = main_loss + norm_loss
        info = torch.tensor([main_loss, norm_loss])
        return loss, info

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
            aux_router_params * config.router_size / config.block_size
        )
        en = config.routed_expert_num
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
