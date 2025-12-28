import yaml
import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict

@dataclass
class Config:
    block_size: int
    latent_size: int
    decoder_output_size: int
    latent_vocab_size: int

    encoder_layer_num: int
    main_layer_num: int
    decoder_layer_num: int
    ar_decoder_layer_num: int

    main_dim: int
    seq_len: int
    vocab_size: int
    attn_head_dim: int
    attn_head_num: int
    latent_head_dim: int
    decoder_head_dim: int
    ffn_hidden_dim: int
    eot_idx: int
    pad_idx: int

    init_std: float
    max_lr: float
    latent_loss_factor: float
    aux_loss_factor: float
    theta: float
    weight_decay: float
    adam_betas: Tuple[float, float]
    grad_clip: float
    norm_eps: float
    adam_eps: float
    dtype: str
    kernel_options: Dict[str, int]

    tied_vocab_emb: bool
    qk_norm: bool
    flex_attn_spec_kernel: bool
    zero_optimizer: bool

    token_pos_emb: bool
    encoder_only_original_token: bool
    encoder_include_original_token: bool
    encoder_global_attn: bool
    decoder_only_original_token: bool
    decoder_include_original_token: bool
    decoder_global_original_attn: bool
    decoder_global_attn: bool
    decoder_output_linear: bool
    latent_target_grad_detach: bool
    include_previous_block: bool
    latent_loss: bool
    latent_l2_loss: bool
    ar_only_in_block: bool
    free_latent_token_num: int

    train_bin: str
    val_bin: str
    batch_size: int
    accumulated_steps: int
    num_steps: int
    warmup_steps: int
    cooldown_steps: int
    val_every: int
    val_steps: int
    lr_scheduler: str  # "wsd", "cos"
    min_lr_ratio: float
    save: bool
    save_step: int
    save_every: int
    save_before_cooldown: bool
    keep_latest_k: int
    overfit: bool
    mixed_precision: bool
    scaler: bool
    model_compile: bool
    wandb_mode: str  # "online", "offline", "disabled"
    load_from_checkpoint: bool
    exclude_optimizer: bool
    profiler: bool

    main_size: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        self.main_size = self.block_size if self.encoder_only_original_token else self.latent_size
        if self.decoder_global_original_attn:
            assert self.decoder_include_original_token


def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)