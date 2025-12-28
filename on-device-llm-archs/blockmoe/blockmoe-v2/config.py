import yaml
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    main_dim: int
    main_attn_num: int
    moe_layer_num: int
    aux_dim: int
    aux_router_dim: int
    aux_gate_dim: int
    aux_attn_num: int
    aux_encoder_layer_num: list[int, int]
    aux_router_layer_num: list[int, int]
    aux_pred: bool
    aux_pred_loss_func: str  # "cross_entropy", "mse"
    aux_pred_loss_factor: list[float, float]
    aux_router_head_num: int
    aux_expert_wise_router: bool
    balance_loss_free: bool
    balance_loss_factor: float
    balance_decay_factor: float
    balance_update_rate: float
    routed_expert_num: int
    expert_num_per_token: int
    expert_dim: int
    gate_func: str  # "softmax", "sigmoid", "sigmoid_norm"
    routed_scaling_factor: float
    first_k_layer_dense: int
    shared_expert: bool
    shared_expert_dim: int
    ffn_factor: float
    block_size: int
    vocab_size: int
    eot_idx: int
    pad_idx: int

    seq_len: int
    init_std: float
    max_lr: float
    theta: float
    theta_s: float
    weight_decay: float
    adam_betas: list[float, float]
    grad_clip: float
    norm_eps: float
    adam_eps: float
    dtype: str
    tied_vocab_emb: bool
    qk_norm: bool
    flex_attn_spec_kernel: bool
    zero_optimizer: bool
    grouped_gemm_option: str  # "gmm", "for_loop"
    grouped_gemm_func: bool
    grouped_gemm_checkpoint: bool
    kernel_options: dict[str, int]

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
    checkpoint_path: str
    load_model: bool
    load_data: bool
    load_optimizer: bool
    profiler: bool
    profiler_step: int
    memory_profiler: bool
    memory_profiler_start_step: int
    memory_profiler_end_step: int

    aux_router_expert_num: int = field(init=False)
    router_size: int = field(init=False)
    total_layer_expert_num: int = field(init=False)
    aux_output_dim: int = field(init=False)
    block_len: int = field(init=False)
    main_attn_dim: int = field(init=False)
    aux_attn_dim: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        self.aux_router_expert_num = (
            self.expert_num_per_token if self.aux_expert_wise_router else 1
        )
        self.total_layer_expert_num = self.routed_expert_num + self.expert_num_per_token
        self.router_size = (
            self.moe_layer_num * self.aux_router_expert_num * self.aux_router_head_num
        )
        self.aux_output_dim = self.aux_router_dim + self.aux_gate_dim
        self.block_len = self.seq_len // self.block_size
        self.main_attn_dim = self.main_dim // self.main_attn_num
        self.aux_attn_dim = self.aux_dim // self.aux_attn_num


def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
