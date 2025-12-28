import os
import math
import sys
import pandas as pd
import glob
import time
from contextlib import nullcontext
import numpy as np
import argparse
import subprocess

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity
from model import Config, Model


def load_data_shard(filename: str, peek: bool = False):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            print0("magic number mismatch in the data .bin file", log_mode=False)
            exit(1)
        assert header[1] == 1, "unsupported version"
        num_token = header[2]
        if peek:
            return num_token
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(tokens) == num_token, "number of tokens read does not match header"
        return tokens


class DistributedDataLoader:
    def __init__(
        self, filename_pattern: str, seq_len: int, process_rank: int, num_processes: int
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_len = seq_len
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        num_token_total = 0
        for fname in self.files:
            shard_num_token = load_data_shard(fname, peek=True)
            assert shard_num_token >= num_processes * seq_len + 1
            num_token_total += int(shard_num_token)
        self.num_token_total = num_token_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_pos = self.process_rank * self.seq_len
        self.tokens = load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        total_seq_len = self.seq_len * self.num_processes
        buffer = self.tokens[self.current_pos : self.current_pos + self.seq_len + 1]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        x = buffer[:-1]
        y = buffer[1:]
        self.current_pos += total_seq_len

        if self.current_pos + total_seq_len >= len(self.tokens):
            self.advance()

        return x, y


config = Config()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_bin",
    type=str,
    default="/root/autodl-tmp/llm.c/dev/data/fineweb10B/fineweb_train_*.bin",
)
parser.add_argument(
    "--input_val_bin",
    type=str,
    default="/root/autodl-tmp/llm.c/dev/data/fineweb10B/fineweb_val_*.bin",
)
parser.add_argument("--accumulated_steps", type=int, default=1)
parser.add_argument("--scheduler", type=str, default="wsd")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--cooldown_steps", type=int, default=100)
parser.add_argument("--val_every", type=int, default=10)
parser.add_argument("--val_steps", type=int, default=20)
parser.add_argument("--min_lr_ratio", type=float, default=0.01)
parser.add_argument("--overfit", type=bool, default=False)
parser.add_argument("--save_every", type=int, default=0)
parser.add_argument("--save_before_cooldown", type=bool, default=False)
parser.add_argument("--amp", type=bool, default=True)
parser.add_argument("--zero_optimizer", type=bool, default=False)
parser.add_argument("--scaler", type=bool, default=True)
parser.add_argument("--model_compile", type=bool, default=True)
parser.add_argument("--scheduler", type=str, default="wsd")  # "wsd" or "cos"
parser.add_argument("--log_freq", type=int, default=10)

args = parser.parse_args()
for arg, value in vars(args).items():
    setattr(config, arg, value)

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


def print0(msg, print_mode: bool = True, log_mode: bool = True):
    if master_process:
        if print_mode:
            print(msg)
        if log_mode:
            with open(log_file, "a") as f:
                f.write(msg + "\n")


with open(sys.argv[0]) as f:
    code = f.read()

run_id = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
if master_process:
    log_dir = f"logs/{run_id}/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"logs/{run_id}.txt"

    with open(log_file, "w") as f:
        f.write(code)
        f.write("=" * 100 + "\n")

print0(
    f"running pytorch {torch.version.__version__}"
    f"compiled for cuda {torch.version.cuda}\nnvidia-smi:"
)
result = subprocess.run(
    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
print0(f"{result.stdout}")
print0("=" * 100)

train_loader = DistributedDataLoader(
    config.input_bin, config.seq_len, ddp_rank, ddp_world_size
)
val_loader = DistributedDataLoader(
    config.input_val_bin, config.seq_len, ddp_rank, ddp_world_size
)

step_tokens = ddp_world_size * config.seq_len
accumulated_step_tokens = step_tokens * config.accumulated_steps
assert (
    val_loader.num_token_total >= config.val_steps * step_tokens
), "validation tokens not enough"

print0(
    f"step_tokens: {step_tokens} \
    accumulated_step_tokens: {accumulated_step_tokens}"
)
print0(
    f"training dataloader: total number of tokens: \
    {train_loader.num_token_total} across {len(train_loader.files)} files"
)
print0(
    f"validation dataloader: total number of tokens: \
    {val_loader.num_token_total} across {len(val_loader.files)} files"
)
print0("=" * 100)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = Model(config).to(device)
model.train()
amp_ctx = (
    torch.amp.autocast(device_type=device, dtype=config.dtype)
    if config.amp
    else nullcontext
)
scaler = torch.amp.GradScaler()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if config.model_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
optimizer = raw_model.optimizer()
original_lr = []
for param_group in optimizer.param_groups:
    original_lr.append(param_group["lr"])

num_steps = config.num_steps
warmup_steps = config.warmup_steps
cooldown_steps = config.cooldown_steps
min_lr_ratio = config.min_lr_ratio


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


if config.scheduler == "wsd":
    get_lr = get_lr_wsd
if config.scheduler == "cos":
    get_lr = get_lr_cos

torch.cuda.synchronize()

total_time_ms = 0.0
time_ms = 0.0
running_avg_time = 0.0
tokens_per_second = 0.0
running_avg_tokens = 0.0
grad_norm = 0.0
train_log = []
train_log_file = f"logs/{run_id}/train_log.csv"
t0 = time.time()

for step in range(num_steps + 1):
    last_step = step == num_steps
    if last_step or (
        step > 0 and config.val_every > 0 and step % config.val_every == 0
    ):
        torch.cuda.synchronize()
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        with torch.no_grad():
            with amp_ctx:
                for _ in range(config.val_steps):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    _, train_loss, _ = model(x_val, y_val)
                    val_loss += train_loss / config.val_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{num_steps} val_loss:{val_loss.item():.4f}")

        model.train()
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    if config.overfit:
        train_loader.reset()

    if step == 6:
        torch.cuda.memory._record_memory_history()

    profile_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if step == 5
        else nullcontext()
    )

    train_loss = 0.0
    aux_loss = 0.0

    with profile_ctx as prof:
        for micro_step in range(config.accumulated_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < config.accumulated_steps - 1
                else nullcontext()
            )

            with sync_ctx:
                with amp_ctx:
                    loss, step_train_loss, step_aux_loss = model(x, y)
                    loss = loss / config.accumulated_steps
                    train_loss += step_train_loss / config.accumulated_steps
                    aux_loss += step_aux_loss / config.accumulated_steps
                    if config.scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

        lr = get_lr(step)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr * original_lr[i]

        if config.scaler:
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.grad_clip != 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(aux_loss, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()

        if step == 5:
            prof.export_chrome_trace(f"logs/{run_id}/trace.json")

        if step == 10:
            torch.cuda.memory._dump_snapshot(f"logs/{run_id}/snapshot.pickle")

    if master_process:
        time_ms = 1000 * (time.time() - t0)
        running_avg_time = (
            time_ms if step < 10 else 0.8 * running_avg_time + 0.2 * time_ms
        )
        total_time_ms += time_ms
        tokens_per_second = accumulated_step_tokens / time_ms * 1000
        running_avg_tokens = (
            tokens_per_second
            if step < 10
            else 0.8 * running_avg_tokens + 0.2 * tokens_per_second
        )

        print0(
            f"step: {step}/{num_steps} "
            f"train_loss: {train_loss.item():.4f} "
            f"time: {total_time_ms / 1000:.0f}s "
            f"avg_time: {running_avg_time:.1f}ms "
            f"avg_speed: {running_avg_tokens:.0f} t/s  "
            f"norm: {grad_norm.item():.3f} "
            f"aux_loss: ({aux_loss.item():.3f}) "
        )

        train_log.append(
            [
                step,
                train_loss.item(),
                total_time_ms / 1000,
                running_avg_time,
                running_avg_tokens,
                grad_norm.item(),
                aux_loss.item(),
            ]
        )

        if step % config.log_freq == 0:
            df = pd.DataFrame(
                train_log,
                columns=[
                    "step",
                    "train_loss",
                    "total_time(s)",
                    "step_time(ms)",
                    "token_speed(t/s)",
                    "grad_norm",
                    "aux_loss",
                ],
            )

            if step == 0:
                df.to_csv(train_log_file, index=False)
            else:
                df.to_csv(train_log_file, mode="a", header=False, index=False)
            train_log = []

    torch.cuda.synchronize()
    t0 = time.time()

if ddp:
    dist.destroy_process_group()
