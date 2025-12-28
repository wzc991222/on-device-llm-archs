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
from fast_pytorch_kmeans import KMeans  # type: ignore
from config import Config

flex_attention = torch.compile(flex_attention, dynamic=False)


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
        attn_num: int,
    ):
        eps = config.norm_eps
        xq = rearrange(xq, "b s (h a) -> b h s a", h=attn_num)
        xk = rearrange(xk, "b s (h a) -> b h s a", h=attn_num)
        xv = rearrange(xv, "b s (h a) -> b h s a", h=attn_num)
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
        ffn_h_dim = int(dim * config.ffn_factor)
        self.attn = nn.Linear(dim, dim * 3, bias=False)
        self.attn_o = nn.Linear(dim, dim, bias=False)
        self.ffn_up = nn.Linear(dim, ffn_h_dim * 2, bias=False)
        self.ffn_down = nn.Linear(ffn_h_dim, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

    def forward(
        self,
        x_input: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
        dim: int,
        attn_num: int,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config, attn_num)
        x_attn_o = self.attn_o(x_attn)

        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)
        ffn_h_dim = int(dim * config.ffn_factor)
        x1, x2 = torch.split(self.ffn_up(x_ffn), ffn_h_dim, -1)
        x3 = F.silu(x1) * x2
        x_ffn_out = self.ffn_down(x3)
        x_output = x_ffn_out + x_ffn_input
        return x_output


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d = config.main_dim
        m = config.latent_size
        n = config.block_size

        self.layers = nn.ModuleList(
            [Layer(config, d) for _ in range(config.encoder_layer_num)]
        )

        if config.encoder_token_pos_emb:
            self.token_pos_emb = nn.Parameter(torch.randn(1, 1, n, d))

        if not config.encoder_inplace:
            self.latent_token = nn.Parameter(torch.randn(1, 1, m, d))

        theta = config.theta if config.encoder_global_attn else config.theta_s
        self.attention = Attention(RotaryEmb(config.main_attn_dim, theta))

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        d = config.main_dim
        an = config.main_attn_num
        n = config.block_size
        m = config.latent_size
        b = config.batch_size
        l = config.block_num

        if config.encoder_token_pos_emb:
            x_input = x_input + self.token_pos_emb

        if config.encoder_inplace:
            x = x_input.flatten(1, 2)
            e = n
        else:
            latent_token = self.latent_token.expand(b, l, -1, -1)
            x = torch.cat([x_input, latent_token], 2).flatten(1, 2)
            e = n + m

        old = config.encoder_old_full_attn
        new = config.encoder_new_full_attn

        def mask_mod_1(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.encoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            block_mask = q_b_idx == kv_b_idx
            if config.encoder_global_attn:
                q_r_idx = q_idx % e
                kv_r_idx = kv_idx % e
                q_old = q_r_idx < n
                q_new = q_r_idx >= n
                kv_old = kv_r_idx < n
                kv_cond = True if new else kv_old
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return (
                    q_old & kv_old & causal_mask & doc_mask
                    | q_new & kv_cond & block_mask
                )
            else:
                if old and new:
                    return block_mask
                if not old and not new:
                    kv_old = (kv_idx % e) < n
                    return block_mask & kv_old
                if old and not new:
                    q_new = (q_idx % e) >= n
                    kv_new = (kv_idx % e) >= n
                    return block_mask & ~(q_new & kv_new)
                if not old and new:
                    q_old = (q_idx % e) < n
                    kv_new = (kv_idx % e) >= n
                    return block_mask & ~(q_old & kv_new)

        b, s = x.shape[:2]
        mask_mod = mask_mod_1 if config.encoder_inplace else mask_mod_2
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, d, an)
        if not config.encoder_inplace:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, n:].flatten(1, 2)
        return x


class AuxDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d = config.main_dim
        n = config.block_size
        ms = config.main_size

        self.layers = nn.ModuleList(
            [Layer(config, d) for _ in range(config.aux_decoder_layer_num)]
        )

        if config.aux_decoder_token_pos_emb:
            self.token_pos_emb = nn.Parameter(torch.randn(1, 1, ms, d))

        if not config.encoder_inplace:
            self.output_token = nn.Parameter(torch.randn(1, 1, n, d))

        theta = config.theta if config.aux_decoder_global_attn else config.theta_s
        self.attention = Attention(RotaryEmb(config.main_attn_dim, theta))
        # TODO: input & residual dropout, attn random mask, add noise

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        d = config.main_dim
        an = config.main_attn_num
        n = config.block_size
        m = config.latent_size
        b = config.batch_size
        l = config.block_num

        x_input = rearrange(x_input, "b (l ms) d -> b l ms d", l=l)

        if config.aux_decoder_token_pos_emb:
            x_input = x_input + self.token_pos_emb

        if config.encoder_inplace:
            x = x_input.flatten(1, 2)
            e = n
        else:
            output_token = self.output_token.expand(b, l, -1, -1)
            x = torch.cat([x_input, output_token], 2).flatten(1, 2)
            e = m + n

        old = config.aux_decoder_old_full_attn
        new = config.aux_decoder_new_full_attn

        def mask_mod_1(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.aux_decoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            block_mask = q_b_idx == kv_b_idx
            if config.aux_decoder_global_attn:
                q_r_idx = q_idx % e
                kv_r_idx = kv_idx % e
                q_old = q_r_idx < m
                q_new = q_r_idx >= m
                kv_old = kv_r_idx < m
                kv_cond = True if new else kv_old
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return (
                    q_old & kv_old & causal_mask & doc_mask
                    | q_new & kv_cond & block_mask
                )
            else:
                if old and new:
                    return block_mask
                if not old and not new:
                    kv_old = (kv_idx % e) < m
                    return block_mask & kv_old
                if old and not new:
                    q_new = (q_idx % e) >= m
                    kv_new = (kv_idx % e) >= m
                    return block_mask & ~(q_new & kv_new)
                if not old and new:
                    q_old = (q_idx % e) < m
                    kv_new = (kv_idx % e) >= m
                    return block_mask & ~(q_old & kv_new)

        b, s = x.shape[:2]
        mask_mod = mask_mod_1 if config.encoder_inplace else mask_mod_2
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, d, an)
        if not config.encoder_inplace:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, m:].flatten(1, 2)
        return x


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d = config.main_dim
        ms = config.main_size

        self.layers = nn.ModuleList(
            [Layer(config, d) for _ in range(config.main_layer_num)]
        )

        if config.main_token_pos_emb:
            self.token_pos_emb = nn.Parameter(torch.randn(1, 1, ms, d))

        self.attention = Attention(RotaryEmb(config.main_attn_dim, config.theta))

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        d = config.main_dim
        an = config.main_attn_num
        ms = config.main_size
        l = config.block_num
        e = ms

        if config.main_token_pos_emb:
            x = x + self.token_pos_emb.expand(-1, l, -1, -1).flatten(1, 2)

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & doc_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, d, an)
        return x


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d = config.main_dim
        ms = config.main_size
        k = config.decoder_size

        self.layers = nn.ModuleList(
            [Layer(config, d) for _ in range(config.decoder_layer_num)]
        )

        if config.decoder_token_pos_emb:
            self.token_pos_emb = nn.Parameter(torch.randn(1, 1, ms, d))

        if not config.decoder_inplace:
            self.output_token = nn.Parameter(torch.randn(1, 1, k, d))

        theta = config.theta if config.decoder_global_attn else config.theta_s
        self.attention = Attention(RotaryEmb(config.main_attn_dim, theta))

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        d = config.main_dim
        an = config.main_attn_num
        ms = config.main_size
        k = config.decoder_size
        b = config.batch_size
        l = config.block_num

        x_input = rearrange(x_input, "b (l ms) d -> b l ms d", l=l)

        if config.decoder_token_pos_emb:
            x_input = x_input + self.token_pos_emb

        if config.decoder_inplace:
            x = x_input.flatten(1, 2)
            e = ms
        else:
            output_token = self.output_token.expand(b, l, -1, -1)
            x = torch.cat([x_input, output_token], 2).flatten(1, 2)
            e = ms + k

        old = config.decoder_old_full_attn
        new = config.decoder_new_full_attn

        def mask_mod_1(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.decoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            block_mask = q_b_idx == kv_b_idx
            if config.decoder_global_attn:
                q_r_idx = q_idx % e
                kv_r_idx = kv_idx % e
                q_old = q_r_idx < ms
                q_new = q_r_idx >= ms
                kv_old = kv_r_idx < ms
                kv_cond = True if new else kv_old
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return (
                    q_old & kv_old & causal_mask & doc_mask
                    | q_new & kv_cond & block_mask
                )
            else:
                if old and new:
                    return block_mask
                if not old and not new:
                    kv_old = (kv_idx % e) < ms
                    return block_mask & kv_old
                if old and not new:
                    q_new = (q_idx % e) >= ms
                    kv_new = (kv_idx % e) >= ms
                    return block_mask & ~(q_new & kv_new)
                if not old and new:
                    q_old = (q_idx % e) < ms
                    kv_new = (kv_idx % e) >= ms
                    return block_mask & ~(q_old & kv_new)

        b, s = x.shape[:2]
        mask_mod = mask_mod_1 if config.decoder_inplace else mask_mod_2
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, d, an)
        if not config.decoder_inplace:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, ms:].flatten(1, 2)
        return x


class Parallel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d = config.main_dim
        pre_p = config.pre_parallel_num
        post_p = config.post_parallel_num

        self.layers = nn.ModuleList(
            [Layer(config, d) for _ in range(config.parallel_layer_num)]
        )

        self.pre_parallel_linear = nn.Linear(d, pre_p * d, bias=False)
        self.post_parallel_linear = nn.Linear(d, post_p * d, bias=False)
        self.parallel_linear = (
            self.post_parallel_linear
            if config.parallel_mode
            else self.pre_parallel_linear
        )

        self.attention = Attention(RotaryEmb(config.main_attn_dim, config.theta_s))

    def init(self, config: Config):
        with torch.no_grad():
            params = self.pre_parallel_linear.weight.data
            params = repeat(params, "pd d -> (c pd) d", c=config.parallel_factor)
            self.post_parallel_linear.weight.data = params

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        d = config.main_dim
        an = config.main_attn_num
        p = config.parallel_num
        o = config.output_size
        e = o

        x = self.parallel_linear(x_input)
        x = rearrange(x, "b s (p d) -> (b p) s d", p=p)

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            block_mask = q_b_idx == kv_b_idx
            return block_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, d, an)
        return x


class ARDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d = config.main_dim
        ad = config.ar_decoder_dim
        eps = config.norm_eps
        num = config.ar_iteration_num

        self.layers = nn.ModuleList(
            [Layer(config, ad) for _ in range(config.ar_decoder_layer_num)]
        )

        if config.ar_old_norm:
            self.ar_old_norm = nn.RMSNorm(d, eps=eps)
        if config.ar_new_norm:
            self.ar_new_norm = nn.RMSNorm(d, eps=eps)

        self.x_old_linear = nn.Linear(d, ad, bias=False)
        self.x_ar_linear = nn.Linear(d, ad, bias=False)
        self.x_ar_linear_list = nn.ModuleList(
            [nn.Linear(d, ad, bias=False) for _ in range(num - 1)]
        )
        if config.ar_decoder_input:
            self.x_de_linear = nn.Linear(d, ad, bias=False)
            self.x_de_linear_list = nn.ModuleList(
                [nn.Linear(d, ad, bias=False) for _ in range(num - 1)]
            )
        self.y_linear_list = nn.ModuleList(
            [nn.Linear(ad, ad, bias=False) for _ in range(num - 1)]
        )
        self.pad_emb = nn.Parameter(torch.randn(ad))

        pre_p = config.pre_parallel_num
        post_p = config.post_parallel_num
        self.pre_agg_linear = nn.Linear(pre_p * ad, d, bias=False)
        self.post_agg_linear = nn.Linear(post_p * ad, d, bias=False)
        self.agg_linear = (
            self.post_agg_linear if config.parallel_mode else self.pre_agg_linear
        )

        self.attention = Attention(
            RotaryEmb(config.ar_decoder_attn_dim, config.theta_s)
        )

    def parallel_init(self, config: Config):
        params = self.post_agg_linear.weight.data
        c = config.parallel_factor
        params = rearrange(params, "d (c pd) -> d c pd", c=c).sum(1)
        diff = self.pre_agg_linear.weight.data - params
        diff = repeat(diff / c, "d pd -> d (c pd)", c=c)
        with torch.no_grad():
            self.post_agg_linear.weight.data += diff

    def iteration_init(self, config: Config):
        with torch.no_grad():
            nn.init.zeros_(self.y_linear_list[-1].weight)
            self.x_ar_linear_list[-1].weight.data = self.x_ar_linear.weight.data
            if config.ar_decoder_input:
                self.x_de_linear_list[-1].weight.data = self.x_de_linear.weight.data

    def forward(
        self,
        x_input: torch.Tensor,
        x_ar: torch.Tensor,
        doc: torch.Tensor,
        begin: torch.Tensor,
        config: Config,
    ):
        o = config.output_size
        n = config.block_size
        p = config.parallel_num
        w = config.ar_attn_window_len
        ad = config.ar_decoder_dim
        aan = config.ar_decoder_attn_num

        if config.ar_decoder_sole_input:
            m = o - n
            e = o
        else:
            m = o
            e = o + n

        x_old = x_input
        if config.ar_decoder_input:
            x_de = x_input[:, :, -n:]
            if config.ar_decoder_sole_input:
                x_old = x_input[:, :, :-n]

        if config.ar_old_norm:
            x_old = self.ar_old_norm(x_old)
        if config.ar_new_norm:
            x_ar = self.ar_new_norm(x_ar)
            if config.ar_decoder_input:
                x_de = self.ar_new_norm(x_de)

        x_old = self.x_old_linear(x_old)
        x_new = self.x_ar_linear(x_ar)
        if config.ar_decoder_input:
            x_new = x_new + self.x_de_linear(x_de)
        x = torch.cat([x_old, x_new], 2).flatten(1, 2)

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            q_r_idx = q_idx % e
            kv_r_idx = kv_idx % e
            q_new = q_r_idx >= m
            kv_new = kv_r_idx >= m
            q_old = q_r_idx < m
            kv_old = kv_r_idx < m
            same_block = q_b_idx == kv_b_idx
            causal_mask = q_idx >= kv_idx
            kv_b_idx_m = max(kv_b_idx - 1, 0)
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            old_doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx_m]
            window_mask = q_r_idx + q_b_idx * n < kv_r_idx + kv_b_idx * n + w

            new_new_mask = q_new & kv_new & causal_mask & window_mask & doc_mask
            any_old_mask = (q_new & old_doc_mask | q_old) & kv_old & same_block
            return new_new_mask | any_old_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        # TODO: check

        num = config.now_ar_iteration_num
        init_pad = torch.zeros_like(begin)
        for i in range(num):
            for _, layer in enumerate(self.layers):
                x = layer(x, block_mask, self.attention, config, ad, aan)
            y = rearrange(x, "bp (l e) ad -> bp l e ad", e=e)
            y = y[:, :, m:]
            j = i + 1
            if j < num:
                condition = (
                    torch.cat(
                        [begin.expand(-1, -1, j), init_pad.expand(-1, -1, n - j)], 2
                    )
                    .flatten(1, 2)
                    .unsqueeze(-1)
                )
                y = y.flatten(1, 2).roll(1, 1)
                y = torch.where(condition, self.pad_emb, y)
                y = rearrange(y, "bp (l n) ad -> bp l n ad", n=n)
                y = self.ar_new_norm(y)

                x_new = self.x_ar_linear_list[i](x_ar) + self.y_linear_list[i](y)
                if config.ar_decoder_input:
                    x_new = x_new + self.x_de_linear_list[i](x_de)
                x = torch.cat([x_old, x_new], 2).flatten(1, 2)

        y = rearrange(y, "(b p) l n ad -> b l n (p ad)", p=p)
        y = self.agg_linear(y)
        return y.flatten(0, -2)


class LMHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.lm_head = nn.Linear(config.main_dim, config.vocab_size, bias=False)
        self.lm_head_c_vec = nn.Parameter(
            torch.randn(config.main_dim, config.lm_head_c_num, config.lm_head_vec_num)
        )
        self.register_buffer(
            "lm_head_c_labels", torch.zeros(config.vocab_size, dtype=torch.int)
        )

    def init(self, config: Config):
        c = config.lm_head_c_num
        v = config.lm_head_vec_num
        lm_head = self.lm_head.weight
        kmeans = KMeans(n_clusters=c, verbose=0)
        labels = kmeans.fit_predict(lm_head)
        self.lm_head_c_labels = labels
        for i in range(c):
            label_mask = labels == i
            sub_lm_head = lm_head[label_mask]
            sub_kmeans = KMeans(n_clusters=v, verbose=0)
            sub_labels = sub_kmeans.fit_predict(sub_lm_head)
            for j in range(v):
                sub_label_mask = sub_labels == j
                with torch.no_grad():
                    self.lm_head_c_vec.data[:, i, j] = (
                        torch.sum(sub_lm_head * sub_label_mask.unsqueeze(-1), 0)
                        / sub_label_mask.sum()
                    )

    def probs_compute(self, input: torch.Tensor, config: Config):
        probs = (input @ self.lm_head_c_vec).flatten(1, 2).softmax(-1)
        probs = rearrange(probs, "s (c v) -> s c v", c=config.lm_head_c_num)
        return probs.sum(2)

    def update(self, config: Config):
        lm_head = self.lm_head.weight
        vec = lm_head.detach() if config.model_grad_detach else lm_head
        vec_probs = self.probs_compute(vec, config)
        self.lm_head_c_labels = vec_probs.max(1)[1]

        vec_label_loss = 0
        lm_head_aux_loss = 0
        if config.vec_label_loss_factor != 0:
            c_labels = F.one_hot(
                self.lm_head_c_labels, num_classes=config.lm_head_c_num
            )
            vec_label_loss = (
                -torch.sum(c_labels * torch.log(vec_probs + 1e-10), -1).mean()
                * config.vec_label_loss_factor
            )
        if config.lm_head_aux_loss_factor != 0:
            c_probs = vec_probs.sum(0) / vec_probs.sum()
            lm_head_aux_loss = (
                torch.dot(c_probs, c_probs)
                * config.lm_head_c_num
                * config.lm_head_aux_loss_factor
            )
        return vec_label_loss, lm_head_aux_loss

    def forward(
        self,
        y: torch.Tensor,
        tokens: torch.LongTensor,
        config: Config,
    ):
        n = config.block_size
        lm_head = self.lm_head.weight
        labels = tokens[:, n + 1 :].flatten()
        logits = self.lm_head(y)
        mask = tokens[:, n:-1].flatten() != config.pad_idx
        mask_sum = mask.sum()
        main_loss = F.cross_entropy(logits, labels, reduction="none")
        main_loss = (main_loss * mask).sum() / mask_sum

        y_vec_loss = 0
        if config.lm_head_c_mode and config.y_vec_loss_factor != 0:
            y = y.detach() if config.model_grad_detach else y
            y_probs = self.probs_compute(y, config)
            if config.y_label_vec_loss:
                vec = lm_head[labels]
            else:
                if config.vec_topk_num == 1:
                    vec_index = logits.max(-1)[1]
                    vec = lm_head[vec_index]
                else:
                    vec_index = logits.topk(
                        k=config.vec_topk_num, dim=-1, sorted=False
                    )[1]
                    vec = lm_head[vec_index].mean(1)
                    # TODO: mean?
            vec = vec.detach() if config.model_grad_detach else vec
            vec_probs = self.probs_compute(vec, config)
            y_vec_loss = F.mse_loss(vec_probs, y_probs, reduction="none").sum(-1)
            y_vec_loss = (y_vec_loss * mask).sum() / mask_sum * config.y_vec_loss_factor

        return main_loss, y_vec_loss, logits.detach()


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.vocab_emb = nn.Embedding(config.vocab_size, config.main_dim)
        self.codebook = nn.Parameter(torch.randn(config.main_dim, config.codebook_size))
        self.encoder = Encoder(config)
        self.aux_decoder = AuxDecoder(config)
        self.main_model = MainModel(config)
        self.decoder = Decoder(config)
        self.parallel = Parallel(config)
        self.ar_decoder = ARDecoder(config)
        self.lm_head = LMHead(config)
        self.y_latent_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.label_latent_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.final_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.lm_head.weight
        self.init_weights(config)

    def ae_forward(
        self,
        tokens: torch.LongTensor,
        config: Config,
    ):
        n = config.block_size
        tokens = tokens[:, 1:]
        input = rearrange(tokens, "b (l n) -> b l n", n=n)
        eot = (input == config.eot_idx).any(-1)
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        x = self.vocab_emb(input)
        x_latent = self.encoder(x, doc, config)
        if config.aux_decoder_vq:
            probs = (x_latent @ self.codebook).softmax(-1)
            prob, index = torch.max(probs, -1)
            emb = self.codebook.t()[index]
            x_latent = F.normalize(x_latent)
            emb = F.normalize(emb)
            prob = (prob / (1 + prob)).unsqueeze(-1)
            x_latent = x_latent + (emb - x_latent).detach() * prob
        y = self.aux_decoder(x_latent, doc, config)
        logits = self.lm_head.lm_head(y).flatten(0, -2)
        labels = tokens.flatten()
        mask = labels != config.pad_idx
        ae_loss = F.cross_entropy(logits, labels, reduction="none")
        ae_loss = (ae_loss * mask).sum() / mask.sum() * config.ae_loss_factor
        return ae_loss

    def forward(
        self,
        tokens: torch.LongTensor,
        config: Config,
    ):
        b, s = tokens.shape
        n = config.block_size
        ms = config.main_size
        o = config.output_size
        f = config.latent_free_token_num

        assert s == config.seq_len + 1
        input = rearrange(tokens[:, 1:], "b (l n) -> b l n", n=n)
        input_ar = rearrange(tokens[:, :-1], "b (l n) -> b l n", n=n)
        eot = (input == config.eot_idx).any(-1)
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        begin = torch.zeros_like(eot)
        begin[:, 1:] = eot[:, :-1]
        begin[:, 1] = True
        pad_num = (input == config.pad_idx).sum(-1)
        x = self.vocab_emb(input)
        x_ar = self.vocab_emb(input_ar)
        x_latent = self.encoder(x, doc, config)
        y_latent_full = self.main_model(x_latent, doc, config)

        orthog_loss = 0
        if config.orthog_loss:
            x_o = rearrange(x_latent, "b (l n) d -> (b l) n d", n=n)
            x_o = F.normalize(x_o, dim=-1)
            x_ot = x_o.transpose(1, 2)
            eye = torch.eye(n, device=x_o.device).unsqueeze(0)
            score = (torch.bmm(x_o, x_ot) - eye) ** 2
            orthog_loss = score.mean() * config.orthog_loss_factor

        latent_loss, aux_loss, codebook_loss = 0, 0, 0
        if config.latent_loss:
            if f > 0:
                y_latent = rearrange(y_latent_full, "b (l ms) d -> b l ms d", ms=ms)
                y_latent = y_latent[:, :-1, f:].flatten(0, -2)
                label_latent = rearrange(x_latent, "b (l ms) d -> b l ms d", ms=ms)
                label_latent = label_latent[:, 1:, f:].flatten(0, -2)
            else:
                y_latent = y_latent_full[:, :-ms].flatten(0, -2)
                label_latent = x_latent[:, ms:].flatten(0, -2)
            y_latent = self.y_latent_norm(y_latent)
            label_latent = self.label_latent_norm(label_latent)

            y_logit = y_latent @ self.codebook
            label_logit = label_latent @ self.codebook
            label_prob = label_logit.softmax(-1)
            latent_loss = F.cross_entropy(
                y_logit, label_prob.detach(), reduction="none"
            )

            label = torch.max(label_prob, -1)[1]
            codebook_loss = F.cross_entropy(label_logit, label, reduction="none")

            weight = torch.where(eot, 0, (n - pad_num) / n)
            weight = repeat(weight[:, :-1], "b l -> (b l e)", e=ms - f)

            latent_loss = (
                (latent_loss * weight).sum() / weight.sum() * config.latent_loss_factor
            )
            codebook_loss = (
                (codebook_loss * weight).sum()
                / weight.sum()
                * config.codebook_loss_factor
            )

            index = label_prob.max(-1)[1]
            index_freq = torch.bincount(index, minlength=config.codebook_size)
            index_prob = index_freq / index.shape[0]
            prob = label_prob.sum(0) / label_prob.sum()
            aux_loss = (
                torch.dot(index_prob, prob)
                * config.codebook_size
                * config.latent_aux_loss_factor
            )

        x_de = self.decoder(y_latent_full, doc, config)
        p = config.parallel_num
        x_ar = repeat(x_ar, "b l n d -> (b p) l n d", p=p)
        doc = repeat(doc, "b l -> (b p) l", p=p)
        begin = repeat(begin, "b l -> (b p) l 1", p=p)

        x_de = self.parallel(x_de, doc, config)
        x_de = rearrange(x_de, "bp (l o) d -> bp l o d", o=o)
        y = self.ar_decoder(
            x_de[:, :-1], x_ar[:, 1:], doc.roll(-1, 1), begin[:, 1:], config
        )
        y = self.final_norm(y)

        main_loss, y_vec_loss, logits = self.lm_head(y, tokens, config)
        loss = (
            main_loss
            + latent_loss
            + aux_loss
            + codebook_loss
            + y_vec_loss
            + orthog_loss
        )
        real_token_num = b * (s - n) - pad_num[:, 1:].sum()
        info = torch.tensor(
            [
                main_loss,
                latent_loss,
                aux_loss,
                codebook_loss,
                orthog_loss,
                y_vec_loss,
                real_token_num,
            ],
            requires_grad=False,
        )

        y_info = label_info = None
        if config.latent_loss:
            y_info = y_logit
            label_info = label_prob

        return loss, info, logits, y_info.detach(), label_info.detach()

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

    def parallel_init(self, config: Config):
        self.parallel.init(config)
        self.ar_decoder.parallel_init(config)

    def get_optimizer(self, config: Config):
        # TODO: maybe two groups
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

    def params_count(self, config: Config):
        encoder_params = sum(x.numel() for x in self.encoder.parameters())
        aux_decocder_params = sum(x.numel() for x in self.aux_decoder.parameters())
        main_params = sum(x.numel() for x in self.main_model.parameters())
        decoder_params = sum(x.numel() for x in self.decoder.parameters())
        parallel_params = sum(x.numel() for x in self.parallel.parameters())
        ar_decoder_params = sum(x.numel() for x in self.ar_decoder.parameters())

        model_params = (
            encoder_params
            + aux_decocder_params
            + main_params
            + decoder_params
            + parallel_params
            + ar_decoder_params
        )
        total_params = sum(x.numel() for x in self.parameters())
        vocab_params = total_params - model_params

        params_result = {
            "encoder_params": encoder_params,
            "aux_decocder_params": aux_decocder_params,
            "main_params": main_params,
            "decoder_params": decoder_params,
            "parallel_params": parallel_params,
            "ar_decoder_params": ar_decoder_params,
            "model_params": model_params,
            "vocab_params": vocab_params,
            "total_params": total_params,
        }

        n = config.block_size
        m = config.latent_size
        ms = config.main_size
        k = config.decoder_size
        o = config.output_size
        f = config.latent_free_token_num
        d = config.main_dim
        ad = config.ar_decoder_dim
        p = config.parallel_num
        ai = config.now_ar_iteration_num
        v = config.vocab_size
        hn = config.lm_head_c_num
        vn = config.lm_head_vec_num
        lv = config.codebook_size

        if config.encoder_inplace:
            encoder_a_params = encoder_params
        else:
            encoder_a_params = int(encoder_params * (n + m) / n)

        if config.encoder_inplace:
            aux_decoder_a_params = aux_decocder_params
        else:
            aux_decoder_a_params = int(aux_decocder_params * (m + n) / n)

        main_a_params = int(main_params * ms / n)

        if config.decoder_inplace:
            decoder_a_params = int(decoder_params * ms / n)
        else:
            decoder_a_params = int(decoder_params * (ms + k) / n)

        parallel_l_params = sum(x.numel() for x in self.parallel.layers.parameters())
        parallel_a_params = int((parallel_l_params + d * d) * p * o / n)

        ar_decoder_l_params = sum(
            x.numel() for x in self.ar_decoder.layers.parameters()
        )
        e = o if config.ar_decoder_sole_input else o + n
        ar_decoder_la_params = int(ar_decoder_l_params * ai * p * e / n)
        ar_decoder_oa_params = ar_decoder_params - ar_decoder_l_params
        if not config.iteration_mode:
            ar_decoder_oa_params -= (d + ad + config.ar_decoder_input * d) * ad * ai
        pre_p = config.pre_parallel_num
        post_p = config.post_parallel_num
        if config.parallel_mode:
            ar_decoder_oa_params -= pre_p * ad * d
        else:
            ar_decoder_oa_params -= post_p * ad * d
        ar_decoder_a_params = ar_decoder_la_params + ar_decoder_oa_params

        model_a_params = (
            encoder_a_params
            + aux_decoder_a_params
            + main_a_params
            + decoder_a_params
            + parallel_a_params
            + ar_decoder_a_params
        )

        lm_head_a_params = 3 * d * v
        if config.lm_head_c_mode:
            lm_head_a_params += d * hn * vn
        latent_vocab_a_params = 0
        if config.latent_loss:
            latent_vocab_a_params = int(2 * d * lv * (ms - f) / n)
        vocab_a_params = lm_head_a_params + latent_vocab_a_params
        total_a_params = model_a_params + vocab_a_params

        activated_params_result = {
            "encoder_a_params": encoder_a_params,
            "aux_decoder_a_params": aux_decoder_a_params,
            "main_a_params": main_a_params,
            "decoder_a_params": decoder_a_params,
            "parallel_a_params": parallel_a_params,
            "ar_decoder_a_params": ar_decoder_a_params,
            "model_a_params": model_a_params,
            "vocab_a_params": vocab_a_params,
            "total_a_params": total_a_params,
        }
        return params_result, activated_params_result
