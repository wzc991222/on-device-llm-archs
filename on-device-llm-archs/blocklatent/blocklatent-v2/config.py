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
    vocab_size: int

    encoder_layer_num: int
    aux_decoder_layer_num: int
    main_layer_num: int
    decoder_layer_num: int
    parallel_layer_num: int
    ar_decoder_layer_num: int

    main_dim: int
    attn_head_dim: int
    attn_head_dim_w: int
    attn_head_dim_h: int
    attn_head_num: int
    latent_head_dim: int
    latent_p_head_dim: int
    latent_c_dim: int
    ffn_hidden_dim: int
    eot_idx: int
    pad_idx: int

    seq_len: int
    init_std: float
    max_lr: float
    ae_loss_factor: float
    latent_loss_factor: float
    aux_loss_factor: float
    theta: float
    theta_w: float
    theta_h: float
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

    encoder_only_original_token: bool
    encoder_include_original_token: bool
    encoder_global_attn: bool
    encoder_token_pos_emb: bool
    encoder_b: bool
    encoder_positional_w: bool
    encoder_rotary_emb: bool
    encoder_rotary_emb_type: str
    encoder_latent_spec_attn: bool

    aux_decoder_include_original_token: bool
    aux_decoder_global_attn: bool
    aux_decoder_token_pos_emb: bool
    aux_decoder_b: bool
    aux_decoder_positional_w: bool
    aux_decoder_rotary_emb: bool

    main_token_pos_emb: bool
    main_b: bool
    main_rotary_emb: bool
    main_rotary_emb_type: str
    main_latent_spec_attn: bool

    decoder_only_original_token: bool
    decoder_include_original_token: bool
    decoder_global_attn: bool
    decoder_token_pos_emb: bool
    decoder_b: bool
    decoder_positional_w: bool
    decoder_rotary_emb: bool

    parallel_token_pos_emb: bool
    parallel_rotary_emb: bool
    ar_token_pos_emb: bool
    ar_b: bool
    ar_rotary_emb: bool
    latent_loss: bool
    latent_l2_loss: bool
    free_latent_token_num: int
    aux_grad_detach: bool

    parallel_init: bool
    parallel_mode: bool
    pre_parallel_num: int
    post_parallel_num: int
    parallel_attn_block_num: int
    decoder_output_norm: bool
    decoder_output_linear: bool
    iteration_init: bool
    iteration_mode: bool
    ar_iteration_num: int
    ar_attn_block_num: int

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
    parallel_factor: int = field(init=False)
    block_num: int = field(init=False)
    parallel_num: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        self.main_size = (
            self.block_size if self.encoder_only_original_token else self.latent_size
        )
        assert self.attn_head_dim_w + self.attn_head_dim_h == self.attn_head_dim
        assert self.attn_head_dim_w % 2 == 0
        assert self.attn_head_dim_h % 2 == 0
        assert self.post_parallel_num % self.pre_parallel_num == 0
        self.parallel_factor = self.post_parallel_num // self.pre_parallel_num
        assert self.seq_len % self.block_size == 0
        self.block_num = self.seq_len // self.block_size
        self.parallel_num = self.post_parallel_num if self.parallel_mode else self.pre_parallel_num

def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
