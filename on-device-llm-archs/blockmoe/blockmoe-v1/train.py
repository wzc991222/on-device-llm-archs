import os
import math
import glob
import time
from contextlib import nullcontext
import numpy as np
import subprocess
import logging
from dataclasses import asdict
import wandb

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity

from model import Model
from config import load_config

config_file: str = ""
checkpoint_path: str = ""
config = load_config(config_file)

batch_size = config.batch_size
seq_len = config.seq_len
num_steps = config.num_steps
warmup_steps = config.warmup_steps
cooldown_steps = config.cooldown_steps
save_steps = num_steps - cooldown_steps
min_lr_ratio = config.min_lr_ratio
scheduler = config.lr_scheduler
val_every = config.val_every
val_steps = config.val_steps
save_every = config.save_every
accumulated_steps = config.accumulated_steps
load_from_checkpoint = config.load_from_checkpoint
profiler = config.profiler
block_size = config.block_size
eot_idx = config.eot_idx
pad_idx = config.pad_idx


def load_data_shard(filename: str, peek: bool = False):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        num_token = header[2]
        if peek:
            return num_token
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(tokens) == num_token
        return tokens


class DistributedDataLoader:
    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_len: int,
        process_rank: int,
        num_processes: int,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_token = batch_size * seq_len
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0

        num_token_total = 0
        for fname in self.files:
            shard_num_token = load_data_shard(fname, peek=True)
            assert shard_num_token >= num_processes * self.num_token + 1
            num_token_total += int(shard_num_token)
        self.num_token_total = num_token_total
        self.reset()

    def state(self):
        return self.current_shard, self.current_pos

    def load_state(self, current_shard: int, current_pos: int):
        self.current_shard = current_shard
        self.current_pos = current_pos + self.process_rank * self.num_token
        self.tokens = load_data_shard(self.files[self.current_shard])

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_pos = self.process_rank * self.num_token
        self.tokens = load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        b, s = self.batch_size, self.seq_len
        total_seq_len = self.num_token * self.num_processes
        buffer = self.tokens[self.current_pos : self.current_pos + self.num_token]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)

        tokens = torch.full((b, s + 1), pad_idx, dtype=torch.long)
        eot = torch.argwhere(buffer == eot_idx).squeeze(-1)
        eot_num = eot.shape[0]
        b_pos_left, b_pos_right, eot_pos = 0, 0, 0

        for row in range(b):
            pos = 0
            while pos <= s:
                rest_len = s + 1 - pos

                if eot_pos >= eot_num or (
                    eot_pos < eot_num and eot[eot_pos] - b_pos_left + 1 > rest_len
                ):
                    b_pos_right = b_pos_left + rest_len
                    tokens[row, pos:] = buffer[b_pos_left:b_pos_right]
                    pos = s + 1
                    b_pos_left = b_pos_right
                else:
                    b_pos_right = eot[eot_pos] + 1
                    s_len = b_pos_right - b_pos_left
                    tokens[row, pos : pos + s_len] = buffer[b_pos_left:b_pos_right]
                    end_pad = (block_size - (pos + s_len) % block_size) % block_size
                    pos = pos + s_len + end_pad
                    b_pos_left = b_pos_right
                    eot_pos += 1

        self.current_pos += total_seq_len
        border_pos = (
            self.current_pos + (self.num_processes - self.process_rank) * self.num_token
        )
        if border_pos > len(self.tokens) - 1:
            self.advance()
        return tokens


def setup_logger(log_dir=dir):
    if not master_process:
        logger.disabled = True
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    return logger


def get_lr_wsd(step: int):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step < num_steps - cooldown_steps:
        return 1.0
    else:
        decay_ratio = (num_steps - step) / cooldown_steps
        return max(decay_ratio, min_lr_ratio)


def get_lr_cos(step: int):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step < num_steps:
        decay_ratio = (step - warmup_steps) / (num_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)
    else:
        return min_lr_ratio


assert torch.cuda.is_available()
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = f"cuda:{torch.cuda.current_device()}"
    master_process = True

dir_name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
dir = f"logs/{dir_name}"

logger = setup_logger()
wandb_logger = wandb.init(dir=dir, mode=config.wandb_mode)
logger.info(f"config_file: {config_file}")
logger.info(f"checkpoint_path: {checkpoint_path}")
logger.info(f"config:\n{asdict(config)}")
logger.info(f"pytorch version: {torch.__version__}")
logger.info(f"cuda version: {torch.version.cuda}")
nvidia_smi_result = subprocess.run(
    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
logger.info(f"nvidia-smi:\n{nvidia_smi_result.stdout}")

train_loader = DistributedDataLoader(
    config.input_bin, batch_size, seq_len, ddp_rank, ddp_world_size
)
val_loader = DistributedDataLoader(
    config.input_val_bin, batch_size, seq_len, ddp_rank, ddp_world_size
)
step_tokens = ddp_world_size * batch_size * seq_len
accumulated_step_tokens = step_tokens * accumulated_steps
logger.info(
    f"step_tokens: {step_tokens} accumulated_step_tokens: {accumulated_step_tokens}"
)
logger.info("=" * 91)

if scheduler == "wsd":
    get_lr = get_lr_wsd
if scheduler == "cos":
    get_lr = get_lr_cos
    assert cooldown_steps == 0

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True

model = Model(config)
if load_from_checkpoint:
    model.load_state_dict(
        torch.load(
            checkpoint_path["model_state_dict"], map_location="cpu", weights_only=True
        )
    )
    train_loader.load_state(*checkpoint_path["train_loader_state"])
model = model.to(device)
if config.model_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
optimizer = raw_model.get_optimizer()
if load_from_checkpoint and not config.exclude_optimizer:
    optimizer.load_state_dict(
        torch.load(checkpoint_path["optimizer_state_dict"], weights_only=True)
    )
    if ddp:
        dist.barrier()
original_lr = []
for param_group in optimizer.param_groups:
    original_lr.append(param_group["lr"])
amp_ctx = (
    torch.amp.autocast(device_type=device, dtype=config.dtype)
    if config.mixed_precision
    else nullcontext()
)
scaler = torch.amp.GradScaler()

torch.cuda.synchronize()
total_time_ms = 0
real_token_num = 0
last_time = time.time()


for step in range(num_steps + 1):
    last_step = step == num_steps
    if last_step or (step > 0 and val_every > 0 and step % val_every == 0):
        model.eval()
        val_loader.reset()
        val_loss = 0
        with torch.no_grad():
            with amp_ctx:
                for _ in range(val_steps):
                    token_val = val_loader.next_batch().to(device)
                    _, output_info = model(token_val, config)
                    val_loss += output_info[0] / val_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        if master_process:
            logger.log(f"step:{step}/{num_steps} val_loss:{val_loss:.4f}")
            wandb_logger.log({"val_loss": val_loss}, step=step)

        model.train()
        torch.cuda.synchronize()
        last_time = time.time()

    if (
        last_step
        or (config.save_before_cooldown and step == save_steps)
        or (step > 0 and save_every and step % save_every == 0 and step < save_steps)
    ):
        if ddp:
            dist.barrier()
        if ddp and config.zero_optimizer:
            optimizer_state_dict = optimizer.consolidate_state_dict()
        else:
            optimizer_state_dict = optimizer.state_dict()
        if master_process:
            save_path = f"{dir}/checkpoint_step_{step}.pth"
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer_state_dict,
                    "train_loader_state": train_loader.state(),
                },
                save_path,
            )
        if ddp:
            dist.barrier()
        logger.log(f"checkpoint at step {step} saved")
        saved_checkpoints = sorted(
            glob.glob(os.path.join(dir, "*.pth")), key=os.path.getmtime, reverse=True
        )
        for old_checkpoint in saved_checkpoints[config.keep_latest_k :]:
            os.remove(old_checkpoint)
            print(f"deleted old checkpoint: {old_checkpoint}")
        torch.cuda.synchronize()
        last_time = time.time()

    if last_step:
        break

    if config.overfit:
        train_loader.reset()

    if step == 6 and profiler:
        torch.cuda.memory._record_memory_history()

    profile_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if (step == 5 and profiler)
        else nullcontext()
    )

    output_info = 0
    with profile_ctx as prof:
        for micro_step in range(accumulated_steps):
            token = train_loader.next_batch().to(device)
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < accumulated_steps - 1
                else nullcontext()
            )

            with sync_ctx:
                with amp_ctx:
                    loss, step_output_info = model(token, config)
                    loss = loss / accumulated_steps
                    output_info += step_output_info
                    if config.scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

        output_info /= accumulated_steps
        lr = get_lr(step)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr * original_lr[i]

        if config.scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ddp:
            dist.all_reduce(output_info, op=dist.ReduceOp.AVG)

    if step == 5 and profiler:
        prof.export_chrome_trace(f"{dir}/trace.json")
        logger.log("profiler generated")

    if step == 8 and profiler:
        torch.cuda.memory._dump_snapshot(f"{dir}/snapshot.pickle")
        logger.log("memory profiler generated")

    if master_process:
        torch.cuda.synchronize()
        time_ms = 1000 * (time.time() - last_time)
        total_time_ms += time_ms
        tokens_per_second = accumulated_step_tokens / time_ms * 1000
        main_loss, aux_loss, latent_loss, latent_aux_loss, step_real_token_num = output_info
        real_token_num += step_real_token_num * accumulated_steps
        token_rate = real_token_num / ((step + 1) * accumulated_step_tokens)
        logger.log(
            f"step: {step}/{num_steps} "
            f"main_loss: {main_loss:.4f} "
            f"aux_loss: {aux_loss:.3f} "
            f"latent_loss: {latent_loss:.3f} "
            f"latent_aux_loss: {latent_aux_loss:.3f} "
            f"total_time: {total_time_ms / 1000:.0f}s "
            f"step_time: {time_ms:.1f}ms "
            f"tps: {tokens_per_second:.0f} t/s "
            f"gard_norm: {grad_norm:.3f} "
            f"token_rate: {token_rate * 100:.2f}%"
        )
        wandb_logger.log(
            {
                "main_loss": main_loss,
                "aux_loss": aux_loss,
                "latent_loss": latent_loss,
                "latent_aux_loss": latent_aux_loss,
                "step_time": time_ms,
                "tokens_per_second": tokens_per_second,
                "grad_norm": grad_norm,
                "token_rate": token_rate,
            },
            step=step,
        )

    torch.cuda.synchronize()
    last_time = time.time()

wandb_logger.finish()
if ddp:
    dist.destroy_process_group()
