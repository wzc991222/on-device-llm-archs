import yaml
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    main_dim: int
    main_attn_num: int
    moe_layer_num: int
    aux_dim: int
    aux_attn_num: int
    aux_layer_num: int
    aux_global_attn: bool
    routed_expert_num: int
    expert_num_per_token: int
    router_group_num: int
    gate_func: str  # "softmax", "sigmoid", "sigmoid_norm"
    balance_loss_factor: float
    balance_loss_free: bool
    balance_loss_sign: bool
    balance_update_rate: float
    routed_scaling_factor: float
    expert_dim: int
    shared_expert: bool
    shared_expert_dim: int
    first_k_layer_dense: int
    last_k_layer_dense: int
    ffn_factor: float
    block_size: int
    vocab_size: int
    eot_idx: int
    indices_sum: bool
    layer_wise_router: bool
    group_wise_router: bool
    token_keys: bool
    expert_bias: bool
    aux_mode: bool
    aux_short_start: int
    aux_short_steps: int
    aux_long_start: int
    aux_long_steps: int
    stat_o_factor: float

    seq_len: int
    init_std: float
    max_lr: float
    theta: float
    weight_decay: float
    adam_betas: list[float, float]
    grad_clip: float
    norm_eps: float
    adam_eps: float
    dtype: str
    tied_vocab_emb: bool
    flex_attn_spec_kernel: bool
    zero_optimizer: bool
    grouped_gemm_option: str  # "gmm", "for_loop"
    grouped_gemm_func: bool
    grouped_gemm_checkpoint: bool
    kernel_options: dict[str, int]

    train_bin: str
    val_bin: str
    batch_size: int
    sum_steps: int
    num_steps: int
    warmup_steps: int
    cooldown_steps: int
    val_every: int
    val_steps: int
    lr_scheduler: str  # "wsd", "cos"
    min_lr_ratio: float
    save: bool
    cooldown_save: bool
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

    group_expert_num: int = field(init=False)
    group_expert_num_per_token: int = field(init=False)
    layer_expert_num: int = field(init=False)
    seq_block_num: int = field(init=False)
    main_attn_dim: int = field(init=False)
    aux_attn_dim: int = field(init=False)
    router_num: int = field(init=False)
    router_size: int = field(init=False)
    router_repeated_num: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        assert self.routed_expert_num % self.router_group_num == 0
        assert self.expert_num_per_token % self.router_group_num == 0
        self.group_expert_num = self.routed_expert_num // self.router_group_num
        self.group_expert_num_per_token = (
            self.expert_num_per_token // self.router_group_num
        )
        self.layer_expert_num = self.routed_expert_num + self.expert_num_per_token
        self.seq_block_num = self.seq_len // self.block_size
        self.main_attn_dim = self.main_dim // self.main_attn_num
        self.aux_attn_dim = self.aux_dim // self.aux_attn_num
        self.router_num = self.moe_layer_num if self.layer_wise_router else 1
        self.router_size = (
            self.router_num * self.router_group_num
            if self.group_wise_router
            else self.router_num
        )
        self.router_repeated_num = (
            1 if self.group_wise_router else self.router_group_num
        )

def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
