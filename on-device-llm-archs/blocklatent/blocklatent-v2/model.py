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


class RotaryEmb2D(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim_w = config.attn_head_dim_w
        dim_h = config.attn_head_dim_h
        self.register_buffer(
            "inv_freq_w", (1 / config.theta_w) ** (torch.arange(0, dim_w, 2) / dim_w)
        )
        self.register_buffer(
            "inv_freq_h", (1 / config.theta_h) ** (torch.arange(0, dim_h, 2) / dim_h)
        )
        self.seq_len_cached = None
        self.freqs_cos_w = None
        self.freqs_cos_h = None
        self.freqs_sin_w = None
        self.freqs_sin_h = None

    def forward(self, x: torch.Tensor, config: Config):
        seq_len = x.shape[2]
        n = config.block_size
        l = seq_len // n
        if seq_len != self.seq_len_cached:
            wt = torch.arange(l, device=x.device)
            wt = repeat(wt, "l -> (l n)", n=n)
            ht = torch.arange(n, device=x.device)
            ht = repeat(ht, "n -> (l n)", l=l)
            freqs_w = torch.outer(wt, self.inv_freq_w)[None, None, :, :]
            freqs_h = torch.outer(ht, self.inv_freq_h)[None, None, :, :]
            self.seq_len_cached = seq_len
            self.freqs_cos_w = freqs_w.cos()
            self.freqs_cos_h = freqs_h.cos()
            self.freqs_sin_w = freqs_w.sin()
            self.freqs_sin_h = freqs_h.sin()

        dim_w = config.attn_head_dim_w
        dim_h = config.attn_head_dim_h
        xw, xh = x.split([dim_w, dim_h], dim=-1)
        xw1, xw2 = xw.chunk(2, dim=-1)
        xh1, xh2 = xh.chunk(2, dim=-1)
        yw1 = xw1 * self.freqs_cos_w + xw2 * self.freqs_sin_w
        yw2 = -xw1 * self.freqs_sin_w + xw2 * self.freqs_cos_w
        yh1 = xh1 * self.freqs_cos_h + xh2 * self.freqs_sin_h
        yh2 = -xh1 * self.freqs_sin_h + xh2 * self.freqs_cos_h
        x_output = torch.cat([yw1, yw2, yh1, yh2], -1).type_as(x)
        return x_output


class Attention(nn.Module):
    def __init__(self, rotary_emb: RotaryEmb | RotaryEmb2D):
        super().__init__()
        self.rotary_emb = rotary_emb

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_mask: BlockMask,
        config: Config,
        rotary_emb_type: str,
        rotary_emb: bool,
    ):
        dim = config.attn_head_dim
        eps = config.norm_eps
        xq = rearrange(xq, "b s (h a) -> b h s a", a=dim)
        xk = rearrange(xk, "b s (h a) -> b h s a", a=dim)
        xv = rearrange(xv, "b s (h a) -> b h s a", a=dim)
        if config.qk_norm:
            xq = F.normalize(xq, dim=-1, eps=eps).type_as(xv)
            xk = F.normalize(xk, dim=-1, eps=eps).type_as(xv)
        if rotary_emb:
            if rotary_emb_type == "1d":
                xq = self.rotary_emb(xq)
                xk = self.rotary_emb(xk)
            if rotary_emb_type == "2d":
                xq = self.rotary_emb(xq, config)
                xk = self.rotary_emb(xk, config)
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
    def __init__(self, config: Config):
        super().__init__()
        self.attn = nn.Linear(config.main_dim, config.main_dim * 3, bias=False)
        self.attn_o = nn.Linear(config.main_dim, config.main_dim, bias=False)
        self.ffn_up = nn.Linear(config.main_dim, config.ffn_hidden_dim * 2, bias=False)
        self.ffn_down = nn.Linear(config.ffn_hidden_dim, config.main_dim, bias=False)
        self.attn_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)

    def forward(
        self,
        x_input: torch.Tensor,
        block_mask: BlockMask,
        attention: Attention,
        config: Config,
        rotary_emb_type: str,
        rotary_emb: bool = True,
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), config.main_dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config, rotary_emb_type, rotary_emb)
        x_attn_o = self.attn_o(x_attn)

        x_ffn_input = x_attn_o + x_input
        x_ffn = self.ffn_norm(x_ffn_input)
        x1, x2 = torch.split(self.ffn_up(x_ffn), config.ffn_hidden_dim, -1)
        x3 = F.silu(x1) * x2
        x_ffn_out = self.ffn_down(x3)
        x_output = x_ffn_out + x_ffn_input
        return x_output


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.encoder_layer_num)]
        )
        md = config.main_dim
        ld = config.latent_head_dim
        pd = config.latent_p_head_dim
        cd = config.latent_c_dim
        m = config.latent_size
        n = config.block_size

        if not config.encoder_only_original_token:
            self.encoder_f1 = nn.Linear(md, m * ld, bias=False)
            self.encoder_f2 = nn.Linear(n * ld, md, bias=False)
            if config.encoder_b:
                self.encoder_bias = nn.Parameter(torch.zeros(1, 1, m, md))
            if config.encoder_positional_w:
                self.encoder_p_in = nn.Linear(md, cd, bias=False)
                self.encoder_p_w1 = nn.Parameter(torch.randn(n, cd, m * pd))
                self.encoder_p_w2 = nn.Parameter(torch.randn(m, n * pd, cd))
                self.encoder_p_out = nn.Linear(cd, md, bias=False)

        if config.encoder_token_pos_emb:
            self.encoder_token_pos_emb = nn.Parameter(torch.zeros(1, 1, n, md))

        if config.encoder_rotary_emb_type == "1d":
            dim = config.attn_head_dim
            theta = config.theta
            self.attention = Attention(RotaryEmb(dim, theta))
        if config.encoder_rotary_emb_type == "2d":
            self.attention = Attention(RotaryEmb2D(config))

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        n = config.block_size
        m = config.latent_size
        b = config.batch_size
        if config.encoder_only_original_token:
            if config.encoder_token_pos_emb:
                x_input = x_input + self.encoder_token_pos_emb
            x = x_input.flatten(1, 2)
            e = n
        else:
            x = rearrange(self.encoder_f1(x_input), "b l n (m ld) -> b l m (n ld)", m=m)
            x = self.encoder_f2(x)
            if config.encoder_b:
                x = x + self.encoder_bias
            if config.encoder_positional_w:
                x_p = self.encoder_p_in(x_input)
                x_p = rearrange(x_p, "b l n cd -> n (b l) cd")
                x_p = torch.bmm(x_p, self.encoder_p_w1)
                x_p = rearrange(x_p, "n bl (m pd) -> m bl (n pd)", m=m)
                x_p = torch.bmm(x_p, self.encoder_p_w2)
                x_p = self.encoder_p_out(x_p)
                x_p = rearrange(x_p, "m (b l) md -> b l m md", b=b)
                x = x + x_p

            if config.encoder_token_pos_emb:
                x_input = x_input + self.encoder_token_pos_emb

            if config.encoder_include_original_token:
                x = torch.cat([x_input, x], 2).flatten(1, 2)
                e = m + n
            else:
                x = x.flatten(1, 2)
                e = m

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.encoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            elif config.encoder_latent_spec_attn:
                q_r_idx = q_idx % e
                kv_r_idx = kv_idx % e
                block_mask = q_b_idx == kv_b_idx
                latent_mask = q_r_idx == kv_r_idx
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return block_mask | (latent_mask & causal_mask & doc_mask)
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        b, s = x.shape[:2]
        rotary_emb_type = config.encoder_rotary_emb_type
        rotary_emb = config.encoder_rotary_emb
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(
                x, block_mask, self.attention, config, rotary_emb_type, rotary_emb
            )
        if config.encoder_include_original_token:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, n:].flatten(1, 2)
        return x


class AuxDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.aux_decoder_layer_num)]
        )
        md = config.main_dim
        ld = config.latent_head_dim
        pd = config.latent_p_head_dim
        cd = config.latent_c_dim
        m = config.latent_size
        n = config.block_size
        ma = config.main_size

        if not config.encoder_only_original_token:
            self.aux_decoder_f1 = nn.Linear(md, n * ld, bias=False)
            self.aux_decoder_f2 = nn.Linear(m * ld, md, bias=False)
            if config.aux_decoder_b:
                self.aux_decoder_bias = nn.Parameter(torch.zeros(1, 1, n, md))
            if config.aux_decoder_positional_w:
                self.aux_decoder_p_in = nn.Linear(md, cd, bias=False)
                self.aux_decoder_p_w1 = nn.Parameter(torch.randn(m, cd, n * pd))
                self.aux_decoder_p_w2 = nn.Parameter(torch.randn(n, m * pd, cd))
                self.aux_decoder_p_out = nn.Linear(cd, md, bias=False)

        if config.aux_decoder_token_pos_emb:
            self.aux_decoder_token_pos_emb = nn.Parameter(torch.zeros(1, 1, ma, md))

        dim = config.attn_head_dim
        theta = config.theta
        self.attention = Attention(RotaryEmb(dim, theta))

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        n = config.block_size
        m = config.latent_size
        ma = config.main_size
        b = config.batch_size
        x_input = rearrange(x_input, "b (l ma) md -> b l ma md", ma=ma)

        if config.encoder_only_original_token:
            if config.aux_decoder_token_pos_emb:
                x_input = x_input + self.aux_decoder_token_pos_emb
            x = x_input.flatten(1, 2)
            e = n
        else:
            x = rearrange(
                self.aux_decoder_f1(x_input), "b l m (n ld) -> b l n (m ld)", n=n
            )
            x = self.aux_decoder_f2(x)
            if config.aux_decoder_b:
                x = x + self.aux_decoder_bias
            if config.aux_decoder_positional_w:
                x_p = self.aux_decoder_p_in(x_input)
                x_p = rearrange(x_p, "b l m cd -> m (b l) cd")
                x_p = torch.bmm(x_p, self.aux_decoder_p_w1)
                x_p = rearrange(x_p, "m bl (n pd) -> n bl (m pd)", n=n)
                x_p = torch.bmm(x_p, self.aux_decoder_p_w2)
                x_p = self.aux_decoder_p_out(x_p)
                x_p = rearrange(x_p, "n (b l) md -> b l n md", b=b)
                x = x + x_p

            if config.aux_decoder_include_original_token:
                if config.aux_decoder_token_pos_emb:
                    x_input = x_input + self.aux_decoder_token_pos_emb
                x = torch.cat([x_input, x], 2).flatten(1, 2)
                e = m + n
            else:
                x = x.flatten(1, 2)
                e = n

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.aux_decoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask
            # TODO: random mask?

        b, s = x.shape[:2]
        rotary_emb = config.aux_decoder_rotary_emb
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, "1d", rotary_emb)
        if config.aux_decoder_include_original_token:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, m:].flatten(1, 2)
        return x


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.main_layer_num)]
        )
        if config.main_rotary_emb_type == "1d":
            dim = config.attn_head_dim
            theta = config.theta
            self.attention = Attention(RotaryEmb(dim, theta))
        if config.main_rotary_emb_type == "2d":
            self.attention = Attention(RotaryEmb2D(config))

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        e = config.main_size

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            causal_mask = q_b_idx >= kv_b_idx
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            if config.main_latent_spec_attn:
                q_r_idx = q_idx % e
                kv_r_idx = kv_idx % e
                block_mask = q_b_idx == kv_b_idx
                latent_mask = q_r_idx == kv_r_idx
                return block_mask | (latent_mask & causal_mask & doc_mask)
            else:
                return causal_mask & doc_mask

        b, s = x.shape[:2]
        rotary_emb_type = config.main_rotary_emb_type
        rotary_emb = config.main_rotary_emb
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(
                x, block_mask, self.attention, config, rotary_emb_type, rotary_emb
            )
        return x


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.decoder_layer_num)]
        )
        md = config.main_dim
        k = config.decoder_output_size
        ma = config.main_size
        ld = config.latent_head_dim
        pd = config.latent_p_head_dim
        cd = config.latent_c_dim

        if not config.decoder_only_original_token:
            self.decoder_f1 = nn.Linear(md, k * ld, bias=False)
            self.decoder_f2 = nn.Linear(ma * ld, md, bias=False)
            if config.decoder_b:
                self.decoder_bias = nn.Parameter(torch.zeros(1, 1, k, md))
            if config.decoder_positional_w:
                self.decoder_p_in = nn.Linear(md, cd, bias=False)
                self.decoder_p_w1 = nn.Parameter(torch.randn(ma, cd, k * pd))
                self.decoder_p_w2 = nn.Parameter(torch.randn(k, ma * pd, cd))
                self.decoder_p_out = nn.Linear(cd, md, bias=False)

        if config.decoder_token_pos_emb:
            self.decoder_token_pos_emb = nn.Parameter(torch.zeros(1, 1, ma, md))

        dim = config.attn_head_dim
        theta = config.theta
        self.attention = Attention(RotaryEmb(dim, theta))

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        m = config.main_size
        k = config.decoder_output_size
        b = config.batch_size
        x_input = rearrange(x_input, "b (l m) md -> b l m md", m=m)

        if config.decoder_only_original_token:
            if config.decoder_token_pos_emb:
                x_input = x_input + self.decoder_token_pos_emb
            x = x_input.flatten(1, 2)
            e = m
        else:
            x = rearrange(self.decoder_f1(x_input), "b l m (k ld) -> b l k (m ld)", k=k)
            x = self.decoder_f2(x)
            if config.decoder_b:
                x = x + self.decoder_bias
            if config.decoder_positional_w:
                x_p = self.decoder_p_in(x_input)
                x_p = rearrange(x_p, "b l m cd -> m (b l) cd")
                x_p = torch.bmm(x_p, self.decoder_p_w1)
                x_p = rearrange(x_p, "m bl (k pd) -> k bl (m pd)", k=k)
                x_p = torch.bmm(x_p, self.decoder_p_w2)
                x_p = self.decoder_p_out(x_p)
                x_p = rearrange(x_p, "k (b l) md -> b l k md", b=b)
                x = x + x_p

            if config.decoder_include_original_token:
                if config.decoder_token_pos_emb:
                    x_input = x_input + self.decoder_token_pos_emb
                x = torch.cat([x_input, x], 2).flatten(1, 2)
                e = m + k
            else:
                x = x.flatten(1, 2)
                e = k

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            if config.decoder_global_attn:
                causal_mask = q_b_idx >= kv_b_idx
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
                return causal_mask & doc_mask
            else:
                block_mask = q_b_idx == kv_b_idx
                return block_mask

        b, s = x.shape[:2]
        rotary_emb = config.decoder_rotary_emb
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, "1d", rotary_emb)
        if config.decoder_include_original_token:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, m:].flatten(1, 2)
        return x


class Parallel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.parallel_layer_num)]
        )
        d = config.main_dim
        k = config.decoder_output_size
        pre_p = config.pre_parallel_num
        post_p = config.post_parallel_num

        self.pre_parallel_linear = nn.Linear(d, pre_p * d, bias=False)
        self.post_parallel_linear = nn.Linear(d, post_p * d, bias=False)
        self.parallel_linear = (
            self.post_parallel_linear
            if config.parallel_mode
            else self.pre_parallel_linear
        )

        if config.parallel_token_pos_emb:
            self.parallel_token_pos_emb = nn.Parameter(torch.zeros(1, k, d))

        dim = config.attn_head_dim
        theta = config.theta_s
        self.attention = Attention(RotaryEmb(dim, theta))

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
        p = config.parallel_num
        e = config.decoder_output_size
        l = config.block_num

        if config.parallel_token_pos_emb:
            token_pos_emb = repeat(
                self.parallel_token_pos_emb, "c k d -> c (l k) d", l=l
            )
            x_input = x_input + token_pos_emb

        x = self.parallel_linear(x_input)
        x = rearrange(x, "b s (p d) -> (b p) s d", p=p)

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            causal_mask = q_b_idx >= kv_b_idx
            window_mask = q_b_idx < kv_b_idx + config.parallel_attn_block_num
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            return causal_mask & window_mask & doc_mask

        b, s = x.shape[:2]
        rotary_emb = config.parallel_rotary_emb
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config, "1d", rotary_emb)
        return x


class ARDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.ar_decoder_layer_num)]
        )
        # TODO: large dim
        d = config.main_dim
        eps = config.norm_eps
        num = config.ar_iteration_num
        if config.decoder_output_norm:
            self.x_de_norm = nn.RMSNorm(d, eps=eps)
        if config.decoder_output_linear:
            self.x_de_linear = nn.Linear(d, d, bias=False)

        self.x_ar_norm = nn.RMSNorm(d, eps=eps)
        self.y_norm_list = nn.ModuleList(
            [nn.RMSNorm(d, eps=eps) for _ in range(num - 1)]
        )
        self.x_ar_linear = nn.Linear(d, d, bias=False)
        self.x_ar_linear_list = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(num - 1)]
        )
        self.y_linear_list = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(num - 1)]
        )
        self.pad_emb = nn.Parameter(torch.randn(d))

        k = config.decoder_output_size
        n = config.block_size
        if config.ar_token_pos_emb:
            self.ar_token_pos_emb = nn.Parameter(torch.zeros(1, 1, k, d))
        if config.ar_b:
            self.ar_bias = nn.Parameter(torch.zeros(1, 1, n, d))

        pre_p = config.pre_parallel_num
        post_p = config.post_parallel_num
        self.pre_agg_linear = nn.Linear(pre_p * d, d, bias=False)
        self.post_agg_linear = nn.Linear(post_p * d, d, bias=False)
        self.agg_linear = (
            self.post_agg_linear if config.parallel_mode else self.pre_agg_linear
        )

        dim = config.attn_head_dim
        theta = config.theta_s
        self.attention = Attention(RotaryEmb(dim, theta))

    def parallel_init(self, config: Config):
        params = self.post_agg_linear.weight.data
        c = config.parallel_factor
        params = rearrange(params, "d (c pd) -> d c pd", c=c).sum(1)
        diff = self.pre_agg_linear.weight.data - params
        diff = repeat(diff / c, "d pd -> d (c pd)", c=c)
        with torch.no_grad():
            self.post_agg_linear.weight.data += diff

    def iteration_init(self):
        with torch.no_grad():
            nn.init.zeros_(self.y_linear_list[-1].weight)
            self.x_ar_linear_list[-1].weight.data = self.x_ar_linear.weight.data

    def forward(
        self,
        x_de: torch.Tensor,
        x_ar: torch.Tensor,
        doc: torch.Tensor,
        begin: torch.Tensor,
        config: Config,
    ):
        k = config.decoder_output_size
        n = config.block_size
        p = config.parallel_num
        a = config.ar_attn_block_num
        e = k + n

        if config.ar_token_pos_emb:
            x_de = x_de + self.ar_token_pos_emb
        if config.ar_b:
            x_ar = x_ar + self.ar_bias

        def mask_mod(b, h, q_idx, kv_idx):
            q_b_idx = q_idx // e
            kv_b_idx = kv_idx // e
            q_r_idx = q_idx % e
            kv_r_idx = kv_idx % e
            q_token = q_r_idx >= k
            kv_token = kv_r_idx >= k
            q_latent = q_r_idx < k
            kv_latent = kv_r_idx < k
            same_block = q_b_idx == kv_b_idx
            causal_mask = q_idx >= kv_idx
            kv_b_idx_m = max(kv_b_idx - 1, 0)
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx]
            latent_doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx_m]
            window_mask = q_b_idx < kv_b_idx + a

            token_token_mask = q_token & kv_token & causal_mask & window_mask & doc_mask
            latent_mask = (
                (q_token & latent_doc_mask | q_latent) & kv_latent & same_block
            )
            return token_token_mask | latent_mask

        if config.decoder_output_norm:
            x_de = self.x_de_norm(x_de)
        if config.decoder_output_linear:
            x_de = self.x_de_linear(x_de)
        x_ar = self.x_ar_norm(x_ar)
        x = torch.cat([x_de, self.x_ar_linear(x_ar)], 2).flatten(1, 2)
        b, s = x.shape[:2]
        rotary_emb = config.ar_rotary_emb
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        # TODO: check

        num = config.ar_iteration_num if config.iteration_mode else 1
        init_pad = torch.zeros_like(begin)
        for i in range(num):
            for _, layer in enumerate(self.layers):
                x = layer(x, block_mask, self.attention, config, "1d", rotary_emb)
            y = rearrange(x, "bp (l e) d -> bp l e d", e=k + n)
            y = y[:, :, k:]
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
                y = rearrange(y, "bp (l n) d -> bp l n d", n=n)
                x = torch.cat(
                    [
                        x_de,
                        self.y_linear_list[i](self.y_norm_list[i](y))
                        + self.x_ar_linear_list[i](x_ar),
                    ],
                    2,
                ).flatten(1, 2)
                # TODO: add decoder output

        y = rearrange(y, "(b p) l n d -> b l n (p d)", p=p)
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
            y_vec_loss = -torch.sum(vec_probs * torch.log(y_probs + 1e-10), -1)
            y_vec_loss = (y_vec_loss * mask).sum() / mask_sum * config.y_vec_loss_factor

        return main_loss, y_vec_loss


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.vocab_emb = nn.Embedding(config.vocab_size, config.main_dim)
        self.encoder = Encoder(config)
        self.aux_decoder = AuxDecoder(config)
        self.main_model = MainModel(config)
        self.decoder = Decoder(config)
        self.parallel = Parallel(config)
        self.ar_decoder = ARDecoder(config)
        self.y_latent_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.label_latent_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.final_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.lm_head = LMHead(config)
        if not config.latent_l2_loss:
            self.latent_head = nn.Linear(
                config.main_dim, config.latent_vocab_size, bias=False
            )
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
        m = config.latent_size
        k = config.decoder_output_size
        f = config.free_latent_token_num

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
        y_latent = self.main_model(x_latent, doc, config)

        latent_loss, aux_loss = 0, 0
        if config.latent_loss:
            if f > 0:
                y_latent_new = rearrange(y_latent, "b (l m) d -> b l m d", m=m)
                y_latent_new = y_latent_new[:, :-1, f:].flatten(0, -2)
                label_latent = rearrange(x_latent, "b (l m) d -> b l m d", m=m)
                label_latent = label_latent[:, 1:, f:].flatten(0, -2)
            else:
                y_latent_new = y_latent[:, :-m].flatten(0, -2)
                label_latent = x_latent[:, m:].flatten(0, -2)
            y_latent_new = self.y_latent_norm(y_latent_new)
            label_latent = self.label_latent_norm(label_latent)
            if config.latent_l2_loss:
                logit_latent = y_latent_new
                target_latent = label_latent
            else:
                logit_latent = self.latent_head(y_latent_new)
                target_latent = self.latent_head(label_latent).softmax(-1)

            weight = torch.where(eot, 0, (n - pad_num) / n)
            weight = repeat(weight[:, :-1], "b l -> (b l e)", e=m - f)
            if config.latent_l2_loss:
                latent_loss = F.mse_loss(logit_latent, target_latent, reduction="none")
            else:
                latent_loss = F.cross_entropy(
                    logit_latent, target_latent, reduction="none"
                )
                # TODO: better loss func
            latent_loss = (
                (latent_loss * weight).sum() / weight.sum() * config.latent_loss_factor
            )

            if not config.latent_l2_loss:
                if config.aux_grad_detach:
                    target_latent_d = self.latent_head(label_latent.detach()).softmax(
                        -1
                    )
                    gate_latent = target_latent_d.sum(0) / target_latent_d.sum()
                else:
                    gate_latent = target_latent.sum(0) / target_latent.sum()
                aux_loss = (
                    torch.dot(gate_latent, gate_latent)
                    * config.latent_vocab_size
                    * config.aux_loss_factor
                )

        x_de = self.decoder(y_latent, doc, config)
        p = config.parallel_num
        x_ar = repeat(x_ar, "b l n d -> (b p) l n d", p=p)
        doc = repeat(doc, "b l -> (b p) l", p=p)
        begin = repeat(begin, "b l -> (b p) l 1", p=p)

        x_de = self.parallel(x_de, doc, config)
        x_de = rearrange(x_de, "bp (l k) d -> bp l k d", k=k)
        y = self.ar_decoder(
            x_de[:, :-1], x_ar[:, 1:], doc.roll(-1, 1), begin[:, 1:], config
        )
        y = self.final_norm(y)

        main_loss, y_vec_loss = self.lm_head(y, tokens, config)
        loss = main_loss + latent_loss + aux_loss + y_vec_loss
        real_token_num = b * (s - n) - pad_num[:, 1:].sum()
        info = torch.tensor(
            [main_loss, latent_loss, aux_loss, y_vec_loss, real_token_num],
            requires_grad=False,
        )
        return loss, info, logit_latent.detach(), target_latent.detach()

    def init_weights(self, config: Config):
        std = config.init_std
        g = torch.Generator()
        g.manual_seed(42)

        for name, param in self.named_parameters():
            if "norm" in name or "bias" in name or "token_pos_emb" in name:
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
        ma = config.main_size
        k = config.decoder_output_size
        f = config.free_latent_token_num
        d = config.main_dim
        ld = config.latent_head_dim
        cd = config.latent_c_dim
        pd = config.latent_p_head_dim
        p = config.parallel_num
        ai = config.ar_iteration_num
        v = config.vocab_size
        hn = config.lm_head_c_num
        vn = config.lm_head_vec_num
        lv = config.latent_vocab_size

        encoder_l_params = sum(x.numel() for x in self.encoder.layers.parameters())
        encoder_oa_params = 0
        if not config.encoder_only_original_token:
            encoder_oa_params = 2 * d * m * ld
            if config.encoder_positional_w:
                encoder_oa_params += 2 * d * cd + 2 * cd * m * pd
        if config.encoder_only_original_token:
            encoder_la_params = encoder_l_params
        elif config.encoder_include_original_token:
            encoder_la_params = encoder_l_params * (m + n) / n
        else:
            encoder_la_params = encoder_l_params * m / n
        encoder_la_params = int(encoder_la_params)
        encoder_a_params = (encoder_la_params + encoder_oa_params) * 2

        aux_decoder_l_params = sum(
            x.numel() for x in self.aux_decoder.layers.parameters()
        )
        aux_decoder_oa_params = 0
        if not config.encoder_only_original_token:
            aux_decoder_oa_params = 2 * d * n * ld * m / n
            if config.aux_decoder_positional_w:
                aux_decoder_oa_params += (2 * d * cd + 2 * cd * n * pd) * m / n
            aux_decoder_oa_params = int(aux_decoder_oa_params)
        if config.aux_decoder_include_original_token:
            aux_decoder_la_params = aux_decoder_l_params * (m + n) / n
        else:
            aux_decoder_la_params = aux_decoder_l_params
        aux_decoder_la_params = int(aux_decoder_la_params)
        aux_decoder_a_params = aux_decoder_la_params + aux_decoder_oa_params

        main_a_params = int(main_params * ma / n)

        decoder_l_params = sum(x.numel() for x in self.decoder.layers.parameters())
        decoder_oa_params = 0
        if not config.decoder_only_original_token:
            decoder_oa_params = 2 * d * k * ld * ma / n
            if config.decoder_positional_w:
                decoder_oa_params += (2 * d * cd + 2 * cd * k * pd) * ma / n
            decoder_oa_params = int(decoder_oa_params)
        if config.decoder_only_original_token:
            decoder_la_params = decoder_l_params * ma / n
        elif config.decoder_include_original_token:
            decoder_la_params = decoder_l_params * (ma + k) / n
        else:
            decoder_la_params = decoder_l_params * k / n
        decoder_la_params = int(decoder_la_params)
        decoder_a_params = decoder_la_params + decoder_oa_params

        parallel_l_params = sum(x.numel() for x in self.parallel.layers.parameters())
        parallel_la_params = int(p * parallel_l_params * k / n)
        parallel_oa_params = int(p * d * d * k / n)
        parallel_a_params = parallel_la_params + parallel_oa_params

        ain = ai if config.iteration_mode else 1
        ar_decoder_l_params = sum(
            x.numel() for x in self.ar_decoder.layers.parameters()
        )
        ar_decoder_la_params = int(ain * p * ar_decoder_l_params * (k + n) / n)
        ar_decoder_oa_params = ar_decoder_params - ar_decoder_l_params
        if not config.iteration_mode:
            ar_decoder_oa_params -= 2 * d * d * ai
        pre_p = config.pre_parallel_num
        post_p = config.post_parallel_num
        if config.parallel_mode:
            ar_decoder_oa_params -= pre_p * d * d
        else:
            ar_decoder_oa_params -= post_p * d * d
        ar_decoder_a_params = ar_decoder_la_params + ar_decoder_oa_params

        model_a_params = (
            encoder_a_params
            + aux_decoder_a_params
            + main_a_params
            + decoder_a_params
            + parallel_a_params
            + ar_decoder_a_params
        )

        lm_head_a_params = 2 * d * v
        if config.lm_head_c_mode:
            lm_head_a_params += d * hn * vn
        latent_count = 0 if config.latent_l2_loss else 2 + config.aux_grad_detach
        latent_vocab_a_params = int(latent_count * d * lv * (m - f) / n)
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
