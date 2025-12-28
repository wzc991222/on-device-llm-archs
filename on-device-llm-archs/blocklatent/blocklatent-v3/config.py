import yaml
import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class Config:
    main_dim: int
    main_attn_num: int
    ffn_factor: float
    ar_decoder_dim: int
    ar_decoder_attn_num: int

    block_size: int
    latent_size: int
    decoder_size: int
    codebook_size: int
    vocab_size: int
    eot_idx: int
    pad_idx: int

    encoder_layer_num: int
    aux_decoder_layer_num: int
    main_layer_num: int
    decoder_layer_num: int
    parallel_layer_num: int
    ar_decoder_layer_num: int

    ae_loss: bool
    latent_loss: bool
    orthog_loss: bool
    ae_loss_factor: float
    latent_loss_factor: float
    codebook_loss_factor: float
    latent_aux_loss_factor: float
    orthog_loss_factor: float
    latent_free_token_num: int

    seq_len: int
    init_std: float
    max_lr: float
    theta: float
    theta_s: float
    weight_decay: float
    adam_betas: Tuple[float, float]
    grad_clip: float
    norm_eps: float
    adam_eps: float
    dtype: str
    tied_vocab_emb: bool
    qk_norm: bool
    flex_attn_spec_kernel: bool
    zero_optimizer: bool
    kernel_options: Dict[str, int]

    encoder_inplace: bool
    encoder_global_attn: bool
    encoder_old_full_attn: bool
    encoder_new_full_attn: bool
    encoder_token_pos_emb: bool
    aux_decoder_vq: bool
    aux_decoder_global_attn: bool
    aux_decoder_old_full_attn: bool
    aux_decoder_new_full_attn: bool
    aux_decoder_token_pos_emb: bool
    main_token_pos_emb: bool
    decoder_inplace: bool
    decoder_global_attn: bool
    decoder_old_full_attn: bool
    decoder_new_full_attn: bool
    decoder_token_pos_emb: bool

    parallel_init: bool
    parallel_mode: bool
    pre_parallel_num: int
    post_parallel_num: int
    ar_old_norm: bool
    ar_new_norm: bool
    ar_decoder_input: bool
    ar_decoder_sole_input: bool
    iteration_init: bool
    iteration_mode: bool
    ar_iteration_num: int
    ar_attn_window_len: int

    lm_head_c_init: bool
    lm_head_c_mode: bool
    lm_head_c_num: int
    lm_head_vec_num: int
    y_vec_loss_factor: float
    vec_label_loss_factor: float
    lm_head_aux_loss_factor: float
    model_grad_detach: bool
    vec_topk_num: int
    y_label_vec_loss: bool

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
    beginning_save_step: int
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
    profiler_step: int
    memory_profiler: bool
    memory_profiler_start_step: int
    memory_profiler_end_step: int

    main_size: int = field(init=False)
    output_size: int = field(init=False)
    parallel_factor: int = field(init=False)
    block_num: int = field(init=False)
    parallel_num: int = field(init=False)
    main_attn_dim: int = field(init=False)
    ar_decoder_attn_dim: int = field(init=False)
    now_ar_iteration_num: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        self.main_size = (
            self.block_size if self.encoder_inplace else self.latent_size
        )
        self.output_size = self.main_size if self.decoder_inplace else self.decoder_size
        assert self.post_parallel_num % self.pre_parallel_num == 0
        self.parallel_factor = self.post_parallel_num // self.pre_parallel_num
        assert self.seq_len % self.block_size == 0
        self.block_num = self.seq_len // self.block_size
        self.parallel_num = self.post_parallel_num if self.parallel_mode else self.pre_parallel_num
        assert self.main_dim % self.main_attn_num == 0
        self.main_attn_dim = self.main_dim // self.main_attn_num
        assert self.ar_decoder_dim % self.ar_decoder_attn_num == 0
        self.ar_decoder_attn_dim = self.ar_decoder_dim // self.ar_decoder_attn_num
        if self.ar_decoder_sole_input:
            assert self.ar_decoder_input and self.output_size > self.block_size
        if self.iteration_mode:
            self.now_ar_iteration_num = self.ar_iteration_num
        else:
            self.now_ar_iteration_num = 1

def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
