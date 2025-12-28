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
from config import Config

flex_attention = torch.compile(flex_attention, dynamic=False)


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
    ):
        x = self.attn_norm(x_input)
        xq, xk, xv = torch.split(self.attn(x), config.main_dim, -1)
        x_attn = attention(xq, xk, xv, block_mask, config)
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
        if not config.encoder_only_original_token:
            self.latent_w1 = nn.Linear(
                config.main_dim, config.latent_size * config.latent_head_dim, bias=False
            )
            self.latent_w2 = nn.Linear(
                config.block_size * config.latent_head_dim, config.main_dim, bias=False
            )
        self.attention = Attention(RotaryEmb(config))

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        n = config.block_size
        m = config.latent_size
        if config.encoder_only_original_token:
            x = x_input.flatten(1, 2)
            e = n
        else:
            x_latent = rearrange(
                self.latent_w1(x_input), "b l n (m v) -> b l m (n v)", m=m
            )
            x_latent = self.latent_w2(x_latent)
            if config.encoder_include_original_token:
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
        if config.encoder_include_original_token:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, n:].flatten(1, 2)
        return x


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.main_layer_num)]
        )
        self.attention = Attention(RotaryEmb(config))

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
            return causal_mask & doc_mask

        b, s = x.shape[:2]
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        return x


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.decoder_layer_num)]
        )
        self.attention = Attention(RotaryEmb(config))
        if not config.decoder_only_original_token:
            self.decoder_w1 = nn.Linear(
                config.main_dim,
                config.decoder_output_size * config.decoder_head_dim,
                bias=False,
            )
            self.decoder_w2 = nn.Linear(
                config.main_size * config.decoder_head_dim, config.main_dim, bias=False
            )
        if config.decoder_output_linear:
            self.decoder_output_linear = nn.Linear(config.main_dim, config.main_dim)

    def forward(
        self,
        x_input: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        m = config.main_size
        k = config.decoder_output_size
        if config.decoder_only_original_token:
            e = m
        else:
            x = rearrange(
                self.decoder_w1(x_input), "b (l m) (k v) -> b l k (m v)", m=m, k=k
            )
            x = self.decoder_w2(x)
            if config.decoder_include_original_token:
                x_input = rearrange(x_input, "b (l m) d -> b l m d", m=m)
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
        if config.decoder_include_original_token:
            x = rearrange(x, "b (l e) d -> b l e d", e=e)
            x = x[:, :, m:].flatten(1, 2)
        if config.decoder_output_linear:
            x = self.decoder_output_linear(x)
        return x


class ARDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Layer(config) for _ in range(config.ar_decoder_layer_num)]
        )
        self.attention = Attention(RotaryEmb(config))

    def forward(
        self,
        x: torch.Tensor,
        doc: torch.Tensor,
        config: Config,
    ):
        k = config.decoder_output_size
        n = config.block_size
        e = k + n
        if config.ar_only_in_block:
            x = x.flatten(0, 1)
        else:
            x = x.flatten(1, 2)

        def mask_mod_1(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            latent_mask = (q_idx < k) & (kv_idx < k)
            return causal_mask | latent_mask

        def mask_mod_2(b, h, q_idx, kv_idx):
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
            doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx_m]
            intra_block_token_mask = q_token & same_block & causal_mask & doc_mask
            inter_block_token_mask = False
            if config.include_previous_block:
                previous_block = q_b_idx == kv_b_idx + 1
                doc_mask = doc[b, q_b_idx] == doc[b, kv_b_idx_m]
                inter_block_token_mask = q_token & kv_token & previous_block & doc_mask
            intra_block_latent_mask = q_latent & kv_latent & same_block
            return (
                intra_block_token_mask
                | inter_block_token_mask
                | intra_block_latent_mask
            )

        b, s = x.shape[:2]
        mask_mod = mask_mod_1 if config.ar_only_in_block else mask_mod_2
        block_mask = create_block_mask(mask_mod, b, None, s, s, _compile=True)
        for _, layer in enumerate(self.layers):
            x = layer(x, block_mask, self.attention, config)
        if config.ar_only_in_block:
            y = rearrange(x, "(b l) e d -> b l e d", b=config.batch_size)
        else:
            y = rearrange(x, "b (l e) d -> b l e d", e=k + n)
        y = y[:, :, k:].flatten(0, -2)
        return y


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.vocab_emb = nn.Embedding(config.vocab_size, config.main_dim)
        self.encoder = Encoder(config)
        self.main_model = MainModel(config)
        self.decoder = Decoder(config)
        self.ar_decoder = ARDecoder(config)
        self.y_latent_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.label_latent_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        self.final_norm = nn.RMSNorm(config.main_dim, eps=config.norm_eps)
        if not config.latent_l2_loss:
            self.latent_head = nn.Linear(
                config.main_dim, config.latent_vocab_size, bias=False
            )
        self.lm_head = nn.Linear(config.main_dim, config.vocab_size, bias=False)
        if config.tied_vocab_emb:
            self.vocab_emb.weight = self.lm_head.weight
        if config.token_pos_emb:
            self.token_pos_emb = nn.Parameter(
                torch.randn(1, 1, config.block_size, config.main_dim)
            )
        self.init_weights(config)

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

        assert (s - 1) % config.block_size == 0
        input = rearrange(tokens[:, 1:], "b (l n) -> b l n", n=n)
        eot = (input == config.eot_idx).any(-1)
        doc = torch.zeros_like(eot)
        doc[:, 1:] = eot.cumsum(-1)[:, :-1]
        pad_num = (input == config.pad_idx).sum(-1)
        input_ar = rearrange(tokens[:, :-1], "b (l n) -> b l n", n=n)
        x = self.vocab_emb(input)
        x_ar = self.vocab_emb(input_ar)
        if config.token_pos_emb:
            x = x + self.token_pos_emb
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
                y_latent_new = y_latent[:, :-m].flatten(0, 1)
                label_latent = x_latent[:, m:].flatten(0, 1)
            y_latent_new = self.y_latent_norm(y_latent_new)
            label_latent = self.label_latent_norm(label_latent)
            if config.latent_l2_loss:
                logit_latent = y_latent_new
                target_latent = label_latent
            else:
                logit_latent = self.latent_head(y_latent_new)
                target_latent = self.latent_head(label_latent).softmax(-1)
            if config.latent_target_grad_detach:
                target_latent = target_latent.detach()
            weight = torch.where(eot, 0, (n - pad_num) / n)
            weight = repeat(weight[:, :-1], "b l -> (b l e)", e=m - f)
            if config.latent_l2_loss:
                latent_loss = F.mse_loss(logit_latent, target_latent, reduction="none")
            else:
                latent_loss = F.cross_entropy(
                    logit_latent, target_latent, reduction="none"
                )
            latent_loss = (
                (latent_loss * weight).sum() / weight.sum() * config.latent_loss_factor
            )

            if not config.latent_l2_loss:
                gate_latent = target_latent
                gate_latent = gate_latent.sum(0) / gate_latent.sum()
                aux_loss = (
                    torch.dot(gate_latent, gate_latent)
                    * config.latent_vocab_size
                    * config.aux_loss_factor
                )

        y_decoder = self.decoder(y_latent, doc, config)
        y_decoder = rearrange(y_decoder, "b (l k) d -> b l k d", k=k)
        ar_decoder_input = torch.cat([y_decoder[:, :-1], x_ar[:, 1:]], 2)
        y = self.ar_decoder(ar_decoder_input, doc, config)
        y = self.final_norm(y)
        logits = self.lm_head(y)
        mask = tokens[:, n:-1].flatten() != config.pad_idx
        main_loss = F.cross_entropy(
            logits, tokens[:, n + 1 :].flatten(), reduction="none"
        )
        main_loss = (main_loss * mask).sum() / mask.sum()
        loss = main_loss + latent_loss + aux_loss
        real_token_num = b * (s - n) - pad_num[:, 1:].sum()
        info = torch.tensor([main_loss, latent_loss, aux_loss, real_token_num])
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
        encoder_params = sum(x.numel() for x in self.encoder.parameters())
        main_params = sum(x.numel() for x in self.main_model.parameters())
        decoder_params = sum(x.numel() for x in self.decoder.parameters())
        ar_decoder_params = sum(x.numel() for x in self.ar_decoder.parameters())
        model_params = encoder_params + main_params + decoder_params + ar_decoder_params
        total_params = sum(x.numel() for x in self.parameters())
        vocab_params = total_params - model_params

        params_result = {
            "encoder_params": encoder_params,
            "main_params": main_params,
            "decoder_params": decoder_params,
            "ar_decoder_params": ar_decoder_params,
            "model_params": model_params,
            "vocab_params": vocab_params,
            "total_params": total_params,
        }

        n = config.block_size
        m = config.latent_size
        k = config.decoder_output_size
        f = config.free_latent_token_num

        if config.encoder_only_original_token:
            encoder_a_params = encoder_params
        elif config.encoder_include_original_token:
            encoder_a_params = encoder_params * (m + n) / n
        else:
            encoder_a_params = encoder_params * m / n
        encoder_a_params = int(encoder_a_params)
        main_a_params = int(main_params * m / n)
        if config.decoder_only_original_token:
            decoder_a_params = decoder_params * m / n
        elif config.decoder_include_original_token:
            decoder_a_params = decoder_params * (m + k) / n
        else:
            decoder_a_params = decoder_params * k / n
        decoder_a_params = int(decoder_a_params)
        ar_decoder_a_params = int(ar_decoder_params * (k + n) / n)
        model_a_params = (
            encoder_a_params + main_a_params + decoder_a_params + ar_decoder_a_params
        )
        vocab_a_params = self.vocab_emb.weight.numel()
        latent_vocab_a_params = (
            0
            if config.latent_l2_loss
            else int(2 * self.latent_head.weight.numel() * (m - f) / n)
        )
        vocab_a_params += latent_vocab_a_params
        total_a_params = model_a_params + vocab_a_params

        activated_params_result = {
            "encoder_a_params": encoder_a_params,
            "main_a_params": main_a_params,
            "decoder_a_params": decoder_a_params,
            "ar_decoder_a_params": ar_decoder_a_params,
            "model_a_params": model_a_params,
            "vocab_a_params": vocab_a_params,
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
