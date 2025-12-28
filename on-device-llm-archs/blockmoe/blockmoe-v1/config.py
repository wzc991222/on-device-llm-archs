import yaml
import torch
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class Config:
    aux_dim: int
    aux_ffn_hidden_dim: int
    aux_encoder_layer_num: int
    aux_block_layer_num: int
    aux_decoder_layer_num: int
    aux_router_layer_num: int
    routed_expert_num: int
    expert_num_per_token: int
    expert_dim: int
    aux_router_head_num: int
    aux_key_head_num: int
    main_key_head_num: int
    reuse_aux_block: bool
    aux_layer_wise_router: bool
    aux_expert_wise_router: bool

    block_size: int
    latent_size: int
    latent_vocab_size: int
    main_layer_num: int
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
    latent_loss_factor: Tuple[float, float]
    aux_loss_factor: float
    theta: float
    weight_decay: float
    adam_betas: Tuple[float, float]
    grad_clip: float
    norm_eps: float
    adam_eps: float
    dtype: str
    router_dtype: str
    grouped_gemm_option: str  # "gmm", "for_loop"
    grouped_gemm_func: bool
    grouped_gemm_checkpoint: bool
    kernel_options: Dict[str, int]
    tied_vocab_emb: bool
    qk_norm: bool
    flex_attn_spec_kernel: bool
    zero_optimizer: bool

    token_pos_emb: bool
    encoder_only_original_token: bool
    encoder_include_original_token: bool
    encoder_global_attn: bool
    decoder_include_original_token: bool
    decoder_global_original_attn: bool
    decoder_global_attn: bool
    latent_target_grad_detach: bool
    latent_l2_loss: bool
    free_latent_token_num: int
    router_global_attn: bool
    adjacent_latent_loss: bool
    decoder_pos_emb: bool
    router_pos_emb: bool

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

    router_size: int = field(init=False)
    router_no_amp: bool = field(init=False)
    predictor_size: int = field(init=False)
    aux_head_num: int = field(init=False)
    aux_router_repeat_num: int = field(init=False)
    aux_key_repeat_num: int = field(init=False)
    main_router_repeat_num: int = field(init=False)
    main_key_repeat_num: int = field(init=False)
    aux_router_expert_num: int = field(init=False)
    total_layer_expert_num: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        self.router_dtype = getattr(torch, self.router_dtype)
        self.router_no_amp = (
            self.dtype == torch.bfloat16 and self.router_dtype == torch.float32
        )
        self.predictor_size = (
            self.block_size if self.encoder_only_original_token else self.latent_size
        )
        self.router_size = self.aux_router_head_num
        if self.aux_layer_wise_router:
            self.router_size *= self.main_layer_num
        if self.aux_expert_wise_router:
            self.router_size *= self.expert_num_per_token
        arh = self.aux_router_head_num
        akh = self.aux_key_head_num
        mkh = self.main_key_head_num
        self.aux_head_num = max(arh, akh)
        ah = self.aux_head_num 
        self.main_head_num = max(self.aux_head_num, mkh)
        assert self.aux_head_num % arh == 0 and self.aux_head_num % akh == 0
        assert self.main_head_num % ah == 0 and self.main_head_num % mkh == 0
        self.aux_router_repeat_num = self.aux_head_num // arh
        self.aux_key_repeat_num = self.aux_head_num // akh
        self.aux_repeat_num = self.main_head_num // ah
        self.main_key_repeat_num = self.main_head_num // mkh
        if not self.aux_layer_wise_router:
            assert self.aux_router_layer_num == self.main_layer_num
        self.aux_router_expert_num = (
            self.expert_num_per_token if self.aux_expert_wise_router else 1
        )
        self.total_layer_expert_num = self.routed_expert_num + self.expert_num_per_token


def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
