import os
import gc
import glob
import time
from contextlib import nullcontext
import numpy as np
import subprocess
import logging
import random
from dataclasses import asdict
import wandb

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity

from model import Model
from config import load_config

config_file: str = "./config.yaml"
config = load_config(config_file)

block_size = config.block_size
seq_len = config.seq_len
router_batch_size = config.router_batch_size
router_sum_steps = config.router_sum_steps
router_num_steps = config.router_num_steps
aux_batch_size = config.aux_batch_size
aux_sum_steps = config.aux_sum_steps
batch_size = config.batch_size
sum_steps = config.sum_steps
num_steps = config.num_steps
router_warmup_steps = config.router_warmup_steps
router_cooldown_steps = config.router_cooldown_steps
warmup_steps = config.warmup_steps
cooldown_steps = config.cooldown_steps
cooldown_save_steps = num_steps - cooldown_steps
min_lr_ratio = config.min_lr_ratio
val_every = config.val_every
val_steps = config.val_steps
load_from_checkpoint = config.load_from_checkpoint
profiler = config.profiler
memory_profiler = config.memory_profiler
eot_idx = config.eot_idx
checkpoint_path = config.checkpoint_path


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
        aux_batch_size: int,
        seq_len: int,
        block_size: int,
        process_rank: int,
        num_processes: int,
    ):
        self.process_rank = process_rank
        self.inv_rank = num_processes - process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.aux_batch_size = aux_batch_size
        self.seq_len = seq_len
        self.num_token = batch_size * seq_len
        self.block_size = block_size
        self.seq_block_num = seq_len // block_size
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
        buffer = self.tokens[self.current_pos : self.current_pos + self.num_token + 1]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        x = buffer[:-1].reshape(b, s)
        y = buffer[1:].reshape(b, s)

        self.current_pos += total_seq_len
        border_pos = self.current_pos + self.inv_rank * self.num_token
        if border_pos > len(self.tokens) - 1:
            self.advance()
        return x, y

    def next_aux_batch(self):
        b, l, n, s = (
            self.aux_batch_size,
            self.seq_block_num,
            self.block_size,
            self.seq_len,
        )
        tokens_len = len(self.tokens) - n - 1
        indices = random.sample(range(tokens_len), b * l)
        offsets = np.arange(n)
        indices = np.array(indices)[:, None] + offsets[None, :]
        buffer = self.tokens[indices]
        buffer = torch.tensor(buffer.astype(np.int32), dtype=torch.long)
        aux_x = buffer.reshape(b, s)
        return aux_x


def setup_logger(log_dir=dir):
    if not master_process:
        logger = logging.getLogger()
        logger.disabled = True
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train.log")

    def filter(msg):
        return not msg.getMessage().startswith("config")

    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(filter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), stream_handler],
    )
    logger = logging.getLogger()
    return logger


def get_lr(step: int):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step < num_steps - cooldown_steps:
        return 1.0
    else:
        decay_ratio = (num_steps - step) / cooldown_steps
        return max(decay_ratio, min_lr_ratio)


def router_get_lr(step: int):
    if step < router_warmup_steps:
        return (step + 1) / router_warmup_steps
    elif step < router_num_steps - router_cooldown_steps:
        return 1.0
    else:
        decay_ratio = (router_num_steps - step) / router_cooldown_steps
        return max(decay_ratio, min_lr_ratio)


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

logger = setup_logger(dir)
wandb_logger = wandb.init(dir=dir, mode=config.wandb_mode)
logger.info(f"config_file: {config_file}")
logger.info(f"checkpoint_path: {checkpoint_path}")
logger.info(f"config:\n{asdict(config)}")
logger.info(f"\npytorch version: {torch.__version__}")
logger.info(f"cuda version: {torch.version.cuda}")
nvidia_smi_result = subprocess.run(
    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
logger.info(f"nvidia-smi:\n{nvidia_smi_result.stdout}")

train_loader = DistributedDataLoader(
    config.train_bin,
    batch_size,
    aux_batch_size,
    seq_len,
    block_size,
    ddp_rank,
    ddp_world_size,
)
val_loader = DistributedDataLoader(
    config.val_bin,
    batch_size,
    aux_batch_size,
    seq_len,
    block_size,
    ddp_rank,
    ddp_world_size,
)

router_train_loader = DistributedDataLoader(
    config.train_bin,
    router_batch_size,
    aux_batch_size,
    seq_len,
    block_size,
    ddp_rank,
    ddp_world_size,
)
router_val_loader = DistributedDataLoader(
    config.val_bin,
    router_batch_size,
    aux_batch_size,
    seq_len,
    block_size,
    ddp_rank,
    ddp_world_size,
)

router_step_tokens = ddp_world_size * router_batch_size * seq_len
router_sum_step_tokens = router_step_tokens * router_sum_steps
total_router_tokens = router_sum_step_tokens * router_num_steps
logger.info(
    f"router_step_tokens: {router_step_tokens} "
    f"router_sum_step_tokens: {router_sum_step_tokens} "
    f"total_router_tokens: {total_router_tokens} "
)

step_tokens = ddp_world_size * batch_size * seq_len
sum_step_tokens = step_tokens * sum_steps
total_tokens = sum_step_tokens * num_steps
logger.info(
    f"step_tokens: {step_tokens} "
    f"sum_step_tokens: {sum_step_tokens} "
    f"total_tokens: {total_tokens} "
)
logger.info("=" * 91)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 32
torch.set_printoptions(precision=3, sci_mode=False)

model = Model(config)
init_step = -1
if load_from_checkpoint:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if config.load_model:
        model.load_state_dict(checkpoint["model_state_dict"])
    if config.load_data:
        train_loader.load_state(*checkpoint["train_loader_state"])
        init_step = checkpoint["step"]
    del checkpoint

model = model.to(device)
if config.model_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
optimizer = raw_model.get_optimizer(config)
if load_from_checkpoint and config.load_optimizer:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    del checkpoint
    torch.cuda.empty_cache()
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
last_time = time.time()


for step in range(init_step + 1, router_num_steps + 1):
    last_step = step == router_num_steps
    if last_step or (step > 0 and val_every > 0 and step % val_every == 0):
        model.eval()
        router_val_loader.reset()
        router_val_loss = 0
        with torch.no_grad():
            with amp_ctx:
                for _ in range(val_steps):
                    x, y = router_val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # aux_mode, router_mode, peek_mode, pre_reweight_mode, reweight_mode, stat_mode
                    cond = False, True, False, False, False, False
                    step_val_loss = model(x, y, config, *cond)[1]
                    router_val_loss += step_val_loss / val_steps
        if ddp:
            dist.all_reduce(router_val_loss, op=dist.ReduceOp.AVG)

        if master_process:
            logger.log(
                logging.INFO,
                f"step:{step}/{router_num_steps} router_val_loss:{router_val_loss:.4f}",
            )
            wandb_logger.log({"router_val_loss": router_val_loss}, step=step)

        model.train()
        torch.cuda.synchronize()
        last_time = time.time()

    if last_step:
        break

    if config.overfit:
        router_train_loader.reset()

    if memory_profiler and step == config.memory_profiler_start_step:
        torch.cuda.memory._record_memory_history()

    profile_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if (profiler and step == config.profiler_step)
        else nullcontext()
    )

    router_main_loss = 0
    router_main_b_loss = 0
    router_aux_b_loss = 0
    with profile_ctx as prof:
        x_list = []
        y_list = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for micro_step in range(router_sum_steps):
            x, y = router_train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            x_list.append(x)
            y_list.append(y)

        with torch.no_grad():
            with amp_ctx:
                for i in range(router_sum_steps):
                    x, y = x_list[i], y_list[i]
                    # aux_mode, router_mode, peek_mode, pre_reweight_mode, reweight_mode, stat_mode
                    cond = False, True, True, True, False, True
                    model(x, y, config, *cond)

        for i in range(router_sum_steps):
            x, y = x_list[i], y_list[i]
            sync_ctx = (
                model.no_sync() if ddp and i < router_sum_steps - 1 else nullcontext()
            )
            with sync_ctx:
                with amp_ctx:
                    # aux_mode, router_mode, peek_mode, pre_reweight_mode, reweight_mode, stat_mode
                    cond = False, True, False, False, True, False
                    loss, step_main_loss, step_main_b_loss = model(x, y, config, *cond)
                    loss = loss / sum_steps
                    router_main_loss += step_main_loss / sum_steps
                    router_main_b_loss += step_main_b_loss / sum_steps
                    if config.scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

        aux_model = raw_model.aux_model
        avg_stat = aux_model.stat / aux_model.count
        avg_logits_max = aux_model.logits_max / aux_model.count
        avg_logits_mean = aux_model.logits_mean / aux_model.count
        logger.log(logging.INFO, f"main indices: {avg_stat}")
        logger.log(logging.INFO, f"main indices max: {avg_stat.max()}")
        logger.log(logging.INFO, f"main logits max: {avg_logits_max}")
        logger.log(logging.INFO, f"main logits mean: {avg_logits_mean}")
        aux_model.reset()

        for micro_step in range(aux_sum_steps):
            aux_x = router_train_loader.next_aux_batch()
            aux_x = aux_x.to(device)
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < aux_sum_steps - 1
                else nullcontext()
            )
            with sync_ctx:
                with amp_ctx:
                    # aux_mode, router_mode, peek_mode, pre_reweight_mode, reweight_mode, stat_mode
                    cond = True, True, False, False, False, True
                    loss = model(aux_x, None, config, *cond)
                    router_aux_b_loss += loss.detach() / aux_sum_steps
                    if config.scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

        if aux_sum_steps > 0:
            aux_model = raw_model.aux_model
            avg_stat = aux_model.stat / aux_model.count
            avg_logits_max = aux_model.logits_max / aux_model.count
            avg_logits_mean = aux_model.logits_mean / aux_model.count
            logger.log(logging.INFO, f"aux indices: {avg_stat}")
            logger.log(logging.INFO, f"aux indices max: {avg_stat.max()}")
            logger.log(logging.INFO, f"aux logits max: {avg_logits_max}")
            logger.log(logging.INFO, f"aux logits mean: {avg_logits_mean}")
            aux_model.reset()

        lr = router_get_lr(step)
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
            dist.all_reduce(router_main_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(router_main_b_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(router_aux_b_loss, op=dist.ReduceOp.AVG)

    if step == config.profiler_step and profiler:
        prof.export_chrome_trace(f"{dir}/trace.json")
        logger.log(logging.INFO, "profiler generated")

    if step == config.memory_profiler_end_step and memory_profiler:
        torch.cuda.memory._dump_snapshot(f"{dir}/snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=False)
        logger.log(logging.INFO, "memory profiler generated")

    if master_process:
        torch.cuda.synchronize()
        time_ms = 1000 * (time.time() - last_time)
        total_time_ms += time_ms
        tokens_per_second = router_sum_step_tokens / time_ms * 1000

        msg = (
            f"\nstep: {step}/{router_num_steps} "
            f"router_main_loss: {router_main_loss:.4f} "
            f"router_main_b_loss: {router_main_b_loss:.4f} "
            f"router_aux_b_loss: {router_aux_b_loss:.4f} "
            f"step_time: {time_ms:.1f}ms "
            f"tps: {tokens_per_second:.0f} t/s "
            f"grad_norm: {grad_norm:.3f} "
        )
        logger.log(logging.INFO, msg)

        wandb_stat = {
            "router_main_loss": router_main_loss,
            "router_main_b_loss": router_main_b_loss,
            "router_aux_b_loss": router_aux_b_loss,
            "step_time": time_ms,
            "tokens_per_second": tokens_per_second,
            "grad_norm": grad_norm,
        }
        wandb_logger.log(wandb_stat, step=step)

    torch.cuda.synchronize()
    last_time = time.time()

# aux_mode, router_mode, peek_mode, pre_reweight_mode, reweight_mode, stat_mode

del raw_model.router_main_model
torch.cuda.empty_cache()

for step in range(init_step + 1, num_steps + 1):
    last_step = step == num_steps
    if last_step or (step > 0 and val_every > 0 and step % val_every == 0):
        model.eval()
        val_loader.reset()
        val_loss = 0
        with torch.no_grad():
            with amp_ctx:
                for _ in range(val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    cond = False, False, False, False, False, False
                    step_val_loss = model(x, y, config, *cond)[1]
                    val_loss += step_val_loss / val_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        if master_process:
            logger.log(logging.INFO, f"step:{step}/{num_steps} val_loss:{val_loss:.4f}")
            wandb_logger.log({"val_loss": val_loss}, step=step)

        model.train()
        torch.cuda.synchronize()
        last_time = time.time()

    cooldown_save = config.cooldown_save and step == cooldown_save_steps
    if config.save and (last_step or cooldown_save):
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
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer_state_dict,
                    "train_loader_state": train_loader.state(),
                    "step": step,
                },
                save_path,
            )
        if ddp:
            dist.barrier()
        logger.log(logging.INFO, f"checkpoint at step {step} saved")
        torch.cuda.synchronize()
        last_time = time.time()

    if last_step:
        break

    if config.overfit:
        train_loader.reset()

    if memory_profiler and step == config.memory_profiler_start_step:
        torch.cuda.memory._record_memory_history()

    profile_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if (profiler and step == config.profiler_step)
        else nullcontext()
    )

    main_loss = 0
    with profile_ctx as prof:
        for micro_step in range(sum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            sync_ctx = (
                model.no_sync() if ddp and micro_step < sum_steps - 1 else nullcontext()
            )
            with sync_ctx:
                with amp_ctx:
                    cond = False, False, False, False, False, False
                    loss, step_main_loss, _ = model(x, y, config, *cond)
                    loss = loss / sum_steps
                    main_loss += step_main_loss / sum_steps
                    if config.scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

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
            dist.all_reduce(main_loss, op=dist.ReduceOp.AVG)

    if step == config.profiler_step and profiler:
        prof.export_chrome_trace(f"{dir}/trace.json")
        logger.log(logging.INFO, "profiler generated")

    if step == config.memory_profiler_end_step and memory_profiler:
        torch.cuda.memory._dump_snapshot(f"{dir}/snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=False)
        logger.log(logging.INFO, "memory profiler generated")

    if master_process:
        torch.cuda.synchronize()
        time_ms = 1000 * (time.time() - last_time)
        total_time_ms += time_ms
        tokens_per_second = sum_step_tokens / time_ms * 1000

        msg = (
            f"\nstep: {step}/{num_steps} "
            f"main_loss: {main_loss:.4f} "
            f"step_time: {time_ms:.1f}ms "
            f"tps: {tokens_per_second:.0f} t/s "
            f"grad_norm: {grad_norm:.3f} "
        )
        logger.log(logging.INFO, msg)

        wandb_stat = {
            "main_loss": main_loss,
            "step_time": time_ms,
            "tokens_per_second": tokens_per_second,
            "grad_norm": grad_norm,
        }
        wandb_logger.log(wandb_stat, step=step)

    torch.cuda.synchronize()
    last_time = time.time()

wandb_logger.finish()
if ddp:
    dist.destroy_process_group()
