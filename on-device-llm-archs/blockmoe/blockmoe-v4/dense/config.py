import yaml
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    main_dim: int
    main_attn_num: int
    layer_num: int
    ffn_factor: float
    vocab_size: int
    eot_idx: int
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

    main_attn_dim: int = field(init=False)

    def __post_init__(self):
        self.dtype = getattr(torch, self.dtype)
        self.main_attn_dim = self.main_dim // self.main_attn_num


def load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
