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
from utils import GroupedGEMM, Wbincount, grouped_gemm_func_original
from config import Config

grouped_gemm = GroupedGEMM.apply
w_bincount = Wbincount.apply


def grouped_gemm_func(
    x: torch.Tensor,
    block: torch.Tensor,
    score: torch.Tensor,
    index: torch.Tensor,
    config: Config,
    type: str,
):
    inputs = x, block, score, index, config, type
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
    def __init__(self, config: Config):
        super().__init__()
        dim = config.attn_head_dim
        self.register_buffer(
            "inv_freq", (1 / config.theta) ** (torch.arange(0, dim, 2) / dim)
        )
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
        x_output = torch.cat((y1, y2), -1).type_as(x)
        return x_output


class Attention(nn.Module):
    def __init__(self, rotary_emb: RotaryEmb):
        super().__init__()
        self.rotary_emb = rotary_emb

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_mask: BlockMask,
        config: Config,
    ):
        dim = config.attn_head_dim
        eps = config.norm_eps
        xq = rearrange(xq, "b s (h a) -> b h s a", a=dim)
        xk = rearrange(xk, "b s (h a) -> b h s a", a=dim)
        xv = rearrange(xv, "b s (h a) -> b h s a", a=dim)
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


class AuxLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = nn.Linear(config.aux_dim, config.aux_dim * 3, bias=False)
        self.attn_o = nn.Linear(config.aux_dim, config.aux_dim, bias=False)
        self.ffn_up = nn.Linear(
            config.aux_dim, config.aux_ffn_hidden_dim * 2, bias=False
        )
        self.ffn_down = nn.Linear(config.aux_ffn_hidden_dim, config.aux_dim, bias=False)
        self.attn_norm = nn.RMSNorm(config.aux_dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.aux_dim, eps=config.norm_eps)

    def forward(
        self,
        x_input: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
        x_ffn: bool = False,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), config.aux_dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config)
        x_attn_o = self.attn_o(x_attn)

        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)
        x1, x2 = torch.split(self.ffn_up(x_ffn), config.aux_ffn_hidden_dim, -1)
        x3 = F.silu(x1) * x2
        x_ffn_out = self.ffn_down(x3)
        x_output = x_ffn_out + x_ffn_input
        if x_ffn:
            return x_output, x_ffn
        else:
            return x_output


class AuxEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [AuxLayer(config) for _ in range(config.aux_encoder_layer_num)]
        )
        self.attention = Attention(RotaryEmb(config))
        if not config.only_original_token:
            self.latent_w1 = nn.Linear(
                config.aux_dim, config.latent_size * config.latent_head_dim, bias=False
            )
            self.latent_w2 = nn.Linear(
                config.block_size * config.latent_head_dim, config.aux_dim, bias=False
            )

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        n = config.block_size
        m = config.latent_size
        if config.only_original_token:
            x = x_input.flatten(1, 2)
            e = n
        else:
            x_latent = rearrange(
                self.latent_w1(x_input), "b c n (m v) -> b c m (n v)", m=m
            )
            x_latent = self.latent_w2(x_latent)
            if config.include_original_token:
                x = torch.cat([x_input, x_latent], 2).flatten(1, 2)
                e = m + n
            else:
                x = x_latent.flatten(1, 2)
                e = m

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.encoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        if config.include_original_token:
            x = rearrange(x, "b (c e) d -> b c e d", e=e)
            x = x[:, :, n:].flatten(1, 2)
        return x


class AuxBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [AuxLayer(config) for _ in range(config.aux_block_layer_num)]
        )

    def forward(
        self,
        x: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
    ):
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, attention, config)
        return x


class AuxPredictor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.reuse_aux_block:
            self.block = AuxBlock(config)
        else:
            self.block_1 = AuxBlock(config)
            self.block_2 = AuxBlock(config)
        self.attention = Attention(RotaryEmb(config))

    def forward(
        self,
        x0: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        e = config.predictor_size

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & doc_mask

        b, s = x0.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        if config.reuse_aux_block:
            x1 = self.block(x0, block_mask, self.attention, config)
            x2 = self.block(x1, block_mask, self.attention, config)
        else:
            x1 = self.block_1(x0, block_mask, self.attention, config)
            x2 = self.block_2(x1, block_mask, self.attention, config)
        return x1, x2


class AuxDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [AuxLayer(config) for _ in range(config.aux_decoder_layer_num)]
        )
        self.attention = Attention(RotaryEmb(config))
        self.decoder_w1 = nn.Linear(
            config.aux_dim, config.router_size * config.decoder_head_dim, bias=False
        )
        self.decoder_w2 = nn.Linear(
            config.predictor_size * config.decoder_head_dim, config.aux_dim, bias=False
        )
        if config.decoder_pos_emb:
            self.decoder_pos_emb = nn.Parameter(
                torch.randn(1, 1, config.router_size, config.aux_dim)
            )

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        m = config.predictor_size
        k = config.router_size
        x = rearrange(
            self.decoder_w1(x_input), "b (c m) (k v) -> b c k (m v)", m=m, k=k
        )
        x = self.decoder_w2(x)
        if config.decoder_pos_emb:
            x = x + self.decoder_pos_emb
        if config.decoder_include_original_token:
            x_input = rearrange(x_input, "b (c m) d -> b c m d", m=m)
            x = torch.cat([x_input, x], 2).flatten(1, 2)
            e = m + k
        else:
            x = x.flatten(1, 2)
            e = k

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.decoder_global_original_attn:
                causal_mask = q_b_idx >= kv_b_idx
                kv_mask = (kv_idx % e) < k
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                block_mask = q_b_idx == kv_b_idx
                return (causal_mask & kv_mask & doc_mask) | block_mask
            if config.decoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        x = rearrange(x, "b (c e) d -> (b c) e d", e=e)
        if config.decoder_include_original_token:
            x = x[:, m:]
        return x


class AuxRouter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [AuxLayer(config) for _ in range(config.aux_router_layer_num)]
        )
        if config.router_pos_emb:
            self.router_pos_emb = nn.Parameter(
                torch.randn(1, config.router_size, config.aux_dim)
            )
        self.aux_final_norm = nn.RMSNorm(config.aux_dim, eps=config.norm_eps)
        self.aux_keys = nn.Parameter(
            torch.randn(
                config.main_layer_num * config.aux_key_head_num,
                config.aux_dim,
                config.routed_expert_num,
            )
        )
        self.aux_probs = nn.Parameter(
            torch.full(
                (
                    1,
                    config.main_layer_num,
                    config.aux_router_expert_num,
                    config.routed_expert_num,
                    config.aux_head_num,
                ),
                1 / config.aux_head_num,
            )
        )
        self.attention = Attention(RotaryEmb(config))

    def forward(
        self,
        x: torch.Tensor,
        config: Config,
    ):
        if config.router_pos_emb:
            x = x + self.router_pos_emb

        def mask_mod(b, h, q_idx, kv_idx):
            return True

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        if not config.aux_layer_wise_router:
            x_ffn_list = []
            for _, layer in enumerate(self.layers):
                x, x_ffn = layer(x, block_mask, self.attention, config, True)
                x_ffn_list.append(x_ffn)
            q = torch.cat(x_ffn_list, 1)
        else:
            for _, layer in enumerate(self.layers):
                x = layer(x, block_mask, self.attention, config)
            q = x

        l = config.main_layer_num
        et = config.expert_num_per_token
        rh = config.aux_router_head_num
        kh = config.aux_key_head_num
        rc = config.aux_router_repeat_num
        kc = config.aux_key_repeat_num
        re = config.aux_router_expert_num
        ah = config.aux_head_num
        e = config.routed_expert_num

        q = repeat(q, "b_c (l re rh) d -> (l rh rc) (b_c re) d", l=l, rh=rh, rc=rc)
        aux_keys = repeat(self.aux_keys, "l_kh d e -> (l_kh kc) d e", kc=kc)
        scores = torch.bmm(q, aux_keys)
        scores = rearrange(scores, "(l ah) (b_c re) e -> l b_c re e ah", l=l, re=re)
        probs = (torch.sigmoid(scores) * self.aux_probs).sum(-1, keepdim=True)
        if re > 1:
            values, indices = probs.max(-2, keepdim=True)
        else:
            values, indices = probs.topk(et, -2, sorted=False)
        indices_expand = indices.expand(-1, -1, -1, -1, ah)
        scores = torch.gather(scores, -2, indices_expand).flatten(2, 3)
        # TODO: check
        indices = indices.squeeze(-1)

        weight = values.flatten()
        indices_flatten = indices.flatten()
        bin_weight = w_bincount(indices_flatten, weight, e)
        bin_index_num = torch.bincount(indices_flatten, minlength=e)
        aux_loss = (
            torch.dot(bin_weight, bin_index_num.type_as(bin_weight))
            / (weight.sum() * indices_flatten.numel())
            * config.routed_expert_num
            * config.aux_loss_factor
            * config.main_layer_num
        )
        return indices, scores, aux_loss


class AuxModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.aux_encoder = AuxEncoder(config)
        self.aux_predictor = AuxPredictor(config)
        self.aux_decoder = AuxDecoder(config)
        self.aux_router = AuxRouter(config)
        if not config.latent_l2_loss:
            self.latent_head = nn.Linear(
                config.aux_dim, config.latent_vocab_size, bias=False
            )
        self.x0_norm = nn.RMSNorm(config.aux_dim, eps=config.norm_eps)
        self.x1_norm = nn.RMSNorm(config.aux_dim, eps=config.norm_eps)
        self.x2_norm = nn.RMSNorm(config.aux_dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        aux_weight: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        m = config.latent_size
        f = config.free_latent_token_num

        x0 = self.aux_encoder(x, doc, config)
        x1, x2 = self.aux_predictor(x0, doc, config)
        latent_loss = [0, 0]
        latent_aux_loss = 0
        if config.latent_loss:
            x0n = self.x0_norm(x0)
            x1n = self.x1_norm(x1)
            x2n = self.x2_norm(x2)
            if f > 0:
                x0n = rearrange(x0n, "b (c m) d -> b c m d", m=m)
                x1n = rearrange(x1n, "b (c m) d -> b c m d", m=m)
                x2n = rearrange(x2n, "b (c m) d -> b c m d", m=m)
                x0n = x0n[:, 2:, f:].flatten(0, -2)
                x1n = x0n[:, 1:-1, f:].flatten(0, -2)
                x2n = x0n[:, :-2, f:].flatten(0, -2)
            else:
                x0n = x0n[:, 2 * m :].flatten(0, 1)
                x1n = x1n[:, m:-m].flatten(0, 1)
                x2n = x1n[:, : -2 * m].flatten(0, 1)
            if not config.latent_l2_loss:
                x0n = self.latent_head(x0n).softmax(-1)
                x1n = self.latent_head(x1n)
                x2n = self.latent_head(x2n)
            if config.latent_target_grad_detach:
                x0n = x0n.detach()
            y1n = x0n
            y2n = x1n.softmax(-1) if config.adjacent_latent_loss else x0n
            if config.latent_l2_loss:
                latent_loss[0] = F.mse_loss(x1n, y1n, reduction="none")
                latent_loss[1] = F.mse_loss(x2n, y2n, reduction="none")
            else:
                latent_loss[0] = F.cross_entropy(x1n, y1n, reduction="none")
                latent_loss[1] = F.cross_entropy(x2n, y2n, reduction="none")
            for i, loss in enumerate(latent_loss):
                loss = (
                    (loss * aux_weight).sum()
                    / aux_weight.sum()
                    * config.latent_loss_factor[i]
                )
            if not config.latent_l2_loss:
                x_gate = x0n
                x_gate = x_gate.sum(0) / x_gate.sum()
                latent_aux_loss = (
                    torch.dot(x_gate, x_gate)
                    * config.latent_vocab_size
                    * config.aux_loss_factor
                )

        x_router = self.aux_decoder(x2, doc, config)
        indices, scores, aux_loss = self.aux_router(x_router, config)
        return indices, scores, aux_loss, latent_loss, latent_aux_loss


class MainLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = nn.Linear(config.main_dim, config.main_dim * 3, bias=False)
        self.attn_output = nn.Linear(config.main_dim, config.main_dim, bias=False)
        self.attn_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.ffn_experts = nn.Parameter(
            torch.randn(
                3,
                config.total_layer_expert_num,
                config.main_dim,
                config.expert_dim,
            )
        )
        self.keys = nn.Parameter(
            torch.randn(
                config.main_key_head_num,
                config.main_dim,
                config.total_layer_expert_num,
            )
        )
        self.head_probs = nn.Parameter(
            torch.full(
                (
                    1,
                    config.aux_router_expert_num,
                    config.total_layer_expert_num,
                    config.main_head_num,
                ),
                1 / config.main_head_num,
            )
        )
        self.score_probs = nn.Parameter(
            torch.full(
                (
                    2,
                    1,
                    config.aux_router_expert_num,
                    config.total_layer_expert_num,
                    config.main_head_num,
                ),
                1 / 2,
            )
        )

    def forward(
        self,
        x_input: torch.Tensor,
        indices: torch.LongTensor,
        scores: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), config.main_dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config)
        x_attn_output = self.attn_output(x_attn)
        x_ffn_input = x_attn_output + x_input
        x_ffn = self.ffn_norm(x_ffn_input)

        kc = config.main_key_repeat_num
        h = config.main_head_num
        n = config.block_size
        b_c = indices.shape[0]
        keys = repeat(self.keys, "kh d e -> (kh kc) d e", kc=kc)
        q = repeat(x_ffn, "b s d -> h (b s) d")
        main_scores = torch.bmm(q, keys)
        scores_indices = repeat(indices.flatten(1), "b_c et -> h (b_c n) et", h=h, n=n)
        main_scores = torch.gather(main_scores, -1, scores_indices)
        main_scores = rearrange(main_scores, "h b_s et -> b_s et h")
        score_probs_indices = repeat(indices, "b_c re de -> 2 b_c re de h", h=h)
        score_probs = self.score_probs.expand(-1, b_c, -1, -1, -1)
        score_probs = torch.gather(score_probs, -2, score_probs_indices)
        score_probs = repeat(
            score_probs.flatten(2, 3), "c b_c et h -> c (b_c n) et h", n=n
        )
        scores = score_probs[0] * main_scores + score_probs[1] * scores
        scores = torch.sigmoid(scores)
        head_probs = self.head_probs.expand(b_c, -1, -1, -1)
        head_probs_indices = score_probs_indices[0]
        head_probs = torch.gather(head_probs, -2, head_probs_indices)
        head_probs = repeat(head_probs.flatten(1, 2), "b_c et h -> (b_c n) et h", n=n)
        scores = (scores * head_probs).sum(-1, keepdim=True).flatten(0, -2)
        indices = repeat(indices.flatten(1), "b_c et -> (b_c n et)", n=n)

        x_ffn_output = grouped_gemm_func(
            x_ffn, self.ffn_experts, scores, indices, config, "ffn"
        )
        x_output = x_ffn_output + x_ffn_input
        return x_output


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [MainLayer(config) for _ in range(config.main_layer_num)]
        )
        self.attention = Attention(RotaryEmb(config))

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor,
        indices: torch.LongTensor,
        scores: torch.Tensor,
        config: Config,
    ):
        n = config.block_size

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // n
            kv_b_idx = kv_idx // n
            causal_mask = q_idx >= kv_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & doc_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for i, layer in enumerate(self.layers):
            x = layer(x, indices[i], scores[i], block_mask, self.attention, config)
        return x


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.vocab_emb = nn.Embedding(config.vocab_size, config.main_dim)
        self.main_to_aux = nn.Linear(config.main_dim, config.aux_dim, bias=False)
        self.main_model = MainModel(config)
        self.aux_model = AuxModel(config)
        self.final_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.main_dim, config.vocab_size, bias=False)
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        if config.token_pos_emb:
            self.token_pos_emb = nn.Parameter(
                torch.randn(1, 1, config.block_size, config.aux_dim)
            )
        self.init_weights(config)

    def forward(
        self,
        tokens: torch.LongTensor,
        config: Config,
    ):
        b, s = input.shape
        n = config.block_size
        m = config.latent_size
        f = config.free_latent_token_num

        assert (s - 1) % config.block_size == 0
        input = rearrange(tokens[:, :-1], "b (c n) -> b c n", n=n)
        eot = (input == config.eot_idx).any(-1)
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        pad = input == config.pad_idx
        pad_num = pad.sum(-1)

        mask = torch.full_like(eot, False)
        mask[:, :2] = True
        mask[:, 1:] |= eot[:, :-1]
        mask[:, 2:] |= eot[:, :-2]
        # TODO: check
        aux_weight = torch.where(mask, 0, (n - pad_num) / n)
        aux_weight = repeat(aux_weight[:, 2:], "b c -> (b c e)", e=m - f)

        x = self.vocab_emb(input)
        x_aux = self.main_to_aux(x)
        if config.token_pos_emb:
            x_aux = x_aux + self.token_pos_emb
        indices, scores, aux_loss, latent_loss, latent_aux_loss = self.aux_model(
            x_aux, aux_weight, doc, config
        )
        x = x.flatten(1, 2)
        
        mask = mask.flatten()[None, :, None]
        mask_indices = torch.arange(
            config.routed_expert_num,
            config.total_layer_expert_num,
            dtype=torch.long,
            device=indices.device,
        )
        if indices.shape[-1] == 1:
            mask_indices = mask_indices[None, None, :, None]
        else:
            mask_indices = mask_indices[None, None, None, :]
        indices_offset = torch.zeros_like(indices)
        if indices.shape[-1] == 1:
            indices_offset[:, :, 2:] = indices[:, :, :-2]
        else:
            indices_offset[:, :, :, 2:] = indices[:, :, :, :-2]
        # check
        indices = torch.where(mask, mask_indices, indices_offset)
        scores = torch.where(mask.unsqueeze(-1), 0, scores)
        ac = config.aux_repeat_num
        n = config.block_size
        scores = repeat(scores, "b_c re ah -> (b_c n) re (ah ac)", n=n, ac=ac)

        y = self.main_model(x, doc, indices, scores, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        main_loss = F.cross_entropy(
            logits.flatten(0, -2), tokens[:, 1:].flatten(), reduction="none"
        )
        main_mask = ~pad.flatten()
        main_loss = (main_loss * main_mask).sum() / main_mask.sum()
        loss = main_loss + aux_loss + latent_loss + latent_aux_loss
        real_token_num = b * s - pad_num.sum()
        info = torch.tensor(
            [main_loss, aux_loss, latent_loss, latent_aux_loss, real_token_num]
        )
        return loss, info

    def init_weights(self, config: Config):
        std = config.init_std
        g = torch.Generator()
        g.manual_seed(42)

        for name, param in self.named_parameters():
            if "norm" in name or "probs" in name:
                continue
            else:
                nn.init.trunc_normal_(
                    param, mean=0, std=std, a=-3 * std, b=3 * std, generator=g
                )

    def params_count(self, config: Config):
        aux_encoder_params = sum(
            x.numel() for x in self.aux_model.aux_encoder.parameters()
        )
        aux_predictor_params = sum(
            x.numel() for x in self.aux_model.aux_predictor.parameters()
        )
        aux_decoder_params = sum(
            x.numel() for x in self.aux_model.aux_decoder.parameters()
        )
        aux_router_params = sum(
            x.numel() for x in self.aux_model.aux_router.parameters()
        )
        aux_model_params = sum(x.numel() for x in self.aux_model.parameters())
        main_model_params = sum(x.numel() for x in self.main_model.parameters())
        total_params = sum(x.numel() for x in self.parameters())

        params_result = {
            "aux_encoder_params": aux_encoder_params,
            "aux_predictor_params": aux_predictor_params,
            "aux_decoder_params": aux_decoder_params,
            "aux_router_params": aux_router_params,
            "aux_model_params": aux_model_params,
            "main_model_params": main_model_params,
            "total_params": total_params,
        }

        n = config.block_size
        m = config.latent_size
        k = config.router_size
        f = config.free_latent_token_num

        if config.encoder_only_original_token:
            aux_encoder_a_params = aux_encoder_params
        elif config.encoder_include_original_token:
            aux_encoder_a_params = aux_encoder_params * (m + n) / n
        else:
            aux_encoder_a_params = aux_encoder_params * m / n
        aux_encoder_a_params = int(aux_encoder_a_params)
        coeff = 2 if config.reuse_aux_block else 1
        aux_predictor_a_params = int(aux_predictor_params * m / n * coeff)
        if config.decoder_include_original_token:
            aux_decoder_a_params = aux_decoder_params * (m + k) / n
        else:
            aux_decoder_a_params = aux_decoder_params * k / n
        aux_decoder_a_params = int(aux_decoder_a_params)
        aux_router_a_params = int(aux_router_params * k / n)
        aux_latent_vocab_a_params = (
            0
            if config.latent_l2_loss
            else int(3 * self.aux_model.latent_head.weight.numel() * (m - f) / n)
        )
        aux_model_a_params = (
            aux_encoder_a_params
            + aux_predictor_a_params
            + aux_decoder_a_params
            + aux_router_a_params
            + aux_latent_vocab_a_params
        )

        d = config.main_dim
        n_layer = config.main_layer_num
        d_e = config.expert_dim
        et = config.expert_num_per_token
        h = config.main_head_num
        main_layer_a_params = n_layer * (d * d * 4 + d * d_e * et * 3 + d * et * h)
        main_vocab_a_params = 2 * self.vocab_emb.weight.numel()
        main_other_a_params = self.main_to_aux.weight.numel()
        if config.token_pos_emb:
            main_other_a_params += self.token_pos_emb.weight.numel()
        main_model_a_params = main_layer_a_params + main_vocab_a_params + main_other_a_params
        total_a_params = main_model_a_params + aux_model_a_params

        activated_params_result = {
            "aux_encoder_a_params": aux_encoder_a_params,
            "aux_predictor_a_params": aux_predictor_a_params,
            "aux_decoder_a_params": aux_decoder_a_params,
            "aux_router_a_params": aux_router_a_params,
            "aux_latent_vocab_a_params": aux_latent_vocab_a_params,
            "aux_model_a_params": aux_model_a_params,
            "main_layer_a_params": main_layer_a_params,
            "main_vocab_a_params": main_vocab_a_params,
            "main_other_a_params": main_other_a_params,
            "main_model_a_params": main_model_a_params,
            "total_a_params": total_a_params,
        }

        return params_result, activated_params_result

    def get_optimizer(self, config: Config):
        lr = config.max_lr
        wd = config.weight_decay
        betas = config.adam_betas
        params = [param for param in self.parameters()]
        param_groups = [{"params": params, "lr": lr, "weight_decay": wd}]
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
