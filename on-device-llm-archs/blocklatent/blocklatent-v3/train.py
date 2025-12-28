import os
import math
import glob
import time
import random
from contextlib import nullcontext
import numpy as np
import subprocess
import logging
from dataclasses import asdict
import wandb

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity

from model import Model
from config import load_config

config_file: str = "./config.yaml"
checkpoint_path: str = ""
config = load_config(config_file)

batch_size = config.batch_size
seq_len = config.seq_len
num_steps = config.num_steps
warmup_steps = config.warmup_steps
cooldown_steps = config.cooldown_steps
wsd_save_steps = num_steps - cooldown_steps
min_lr_ratio = config.min_lr_ratio
scheduler = config.lr_scheduler
val_every = config.val_every
val_steps = config.val_steps
save_every = config.save_every
accumulated_steps = config.accumulated_steps
load_from_checkpoint = config.load_from_checkpoint
profiler = config.profiler
memory_profiler = config.memory_profiler
block_size = config.block_size
eot_idx = config.eot_idx
pad_idx = config.pad_idx
chunk_size = config.main_size - config.latent_free_token_num
n = config.block_size


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
            pos = 1
            while pos <= s:
                pad_num = random.randint(0, block_size - 1)
                pos += pad_num
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
                    end_pad = (block_size - (pos + s_len - 1) % block_size) % block_size
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
    config.train_bin, batch_size, seq_len, ddp_rank, ddp_world_size
)
val_loader = DistributedDataLoader(
    config.val_bin, batch_size, seq_len, ddp_rank, ddp_world_size
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
torch._dynamo.config.cache_size_limit = 32

model = Model(config)
init_step = -1
if load_from_checkpoint:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_loader.load_state(*checkpoint["train_loader_state"])
    init_step = checkpoint["step"]
    del checkpoint

if config.parallel_init:
    model.parallel_init(config)
if config.iteration_init:
    model.ar_decoder.iteration_init(config)
if config.lm_head_c_init:
    model.lm_head.init(config)

model = model.to(device)
if config.model_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
optimizer = raw_model.get_optimizer(config)
if load_from_checkpoint and not config.exclude_optimizer:
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
real_token_num = 0
last_time = time.time()


for step in range(init_step + 1, num_steps + 1):
    last_step = step == num_steps
    if last_step or (step > 0 and val_every > 0 and step % val_every == 0):
        model.eval()
        val_loader.reset()
        val_loss = 0
        with torch.no_grad():
            with amp_ctx:
                for _ in range(val_steps):
                    token_val = val_loader.next_batch().to(device)
                    _, output_info, _, _, _ = model(token_val, config)
                    val_loss += output_info[0] / val_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        if master_process:
            logger.log(logging.INFO, f"step:{step}/{num_steps} val_loss:{val_loss:.4f}")
            wandb_logger.log({"val_loss": val_loss}, step=step)

        model.train()
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

    output_info = 0
    ae_loss_info = 0
    with profile_ctx as prof:
        if config.lm_head_c_mode:
            vec_label_loss, lm_head_aux_loss = model.lm_head.update(config)
            lm_head_loss = vec_label_loss + lm_head_aux_loss
            if config.scaler:
                scaler.scale(lm_head_loss).backward()
            else:
                lm_head_loss.backward()

        if step % 3 == 0:
            tokens = train_loader.next_batch().to(device)

        for micro_step in range(accumulated_steps):
            #
            sync_ctx = (
                model.no_sync()
                if ddp and micro_step < accumulated_steps - 1
                else nullcontext()
            )

            with sync_ctx:
                with amp_ctx:
                    if config.ae_loss:
                        ae_loss = model.ae_forward(tokens, config)
                        ae_loss = ae_loss / accumulated_steps
                        ae_loss_info += ae_loss.detach()
                        if config.scaler:
                            scaler.scale(ae_loss).backward()
                        else:
                            ae_loss.backward()

                    loss, step_output_info, logits, y_info, label_info = model(
                        tokens, config
                    )
                    loss = loss / accumulated_steps
                    output_info += step_output_info
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

        output_info /= accumulated_steps
        if ddp:
            dist.all_reduce(output_info, op=dist.ReduceOp.AVG)
            dist.all_reduce(ae_loss_info, op=dist.ReduceOp.AVG)

        if step % 60 < 3 and config.latent_loss:
            sample_size = 64 // chunk_size * chunk_size
            token_sample = tokens[0, n + 1 : sample_size + n + 1]
            token_onehot = F.one_hot(token_sample, config.vocab_size)
            # print(f"token_onehot: {token_onehot.shape}")
            main_probs = logits.softmax(-1)
            main_probs_sample = main_probs[:sample_size]
            # print(f"main_probs_sample: {main_probs_sample.shape}")
            main_diff = main_probs_sample - token_onehot
            main_diff = main_diff.abs().sum(-1).reshape(-1, chunk_size)
            token_sample = token_sample.reshape(-1, chunk_size)

            y_info = y_info.softmax(-1)
            y_sample = y_info[:sample_size].max(-1)[1].reshape(-1, chunk_size)
            label_sample = label_info[:sample_size].max(-1)[1].reshape(-1, chunk_size)
            token_msg = f"token: {token_sample}"
            y_msg = f"y: {y_sample}"
            label_msg = f"label: {label_sample}"
            y_label_msg = f"y_label: {y_sample == label_sample}"
            prob_diff = y_info[:sample_size] - label_info[:sample_size]
            prob_diff = prob_diff.abs().sum(-1).reshape(-1, chunk_size)
            label_msg_1 = f"{label_info.topk(10, -1)[0].mean(0)}"
            label_msg_2 = f"{label_info.mean(0).topk(10)[0]}"
            label_msg_3 = f"{main_probs.topk(10, -1)[0].mean(0)}"
            label_msg_4 = f"{main_probs.mean(0).topk(10)[0]}"

            logger.log(logging.INFO, token_msg)
            logger.log(logging.INFO, y_msg)
            logger.log(logging.INFO, label_msg)
            logger.log(logging.INFO, y_label_msg)
            logger.log(logging.INFO, prob_diff)
            logger.log(logging.INFO, main_diff)
            logger.log(logging.INFO, label_msg_1)
            logger.log(logging.INFO, label_msg_2)
            logger.log(logging.INFO, label_msg_3)
            logger.log(logging.INFO, label_msg_4)

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
        tokens_per_second = accumulated_step_tokens / time_ms * 1000
        (
            main_loss,
            latent_loss,
            aux_loss,
            codebook_loss,
            orthog_loss,
            y_vec_loss,
            step_real_token_num,
        ) = output_info
        real_token_num += step_real_token_num * accumulated_steps
        token_rate = real_token_num / ((step - init_step) * accumulated_step_tokens)
        msg = (
            f"\nstep: {step}/{num_steps} "
            f"main_loss: {main_loss:.4f} "
            f"latent_loss: {latent_loss:.3f} "
            f"ae_loss: {ae_loss_info:.3f} "
            f"aux_loss: {aux_loss:.3f} "
            f"codebook_loss: {codebook_loss:.3f} "
            f"orthog_loss: {orthog_loss:.3f} "
            f"total_time: {total_time_ms / 1000:.0f}s "
            f"step_time: {time_ms:.1f}ms "
            f"tps: {tokens_per_second:.0f} t/s "
            f"grad_norm: {grad_norm:.3f} "
            f"token_rate: {token_rate * 100:.2f}% "
        )
        if config.lm_head_c_mode:
            lm_head_msg = (
                f"y_vec_loss: {y_vec_loss:.3f} "
                f"vec_label_loss: {vec_label_loss:.3f} "
                f"lm_head_aux_loss: {lm_head_aux_loss:.3f} "
            )
            msg += lm_head_msg
        logger.log(logging.INFO, msg)
        wandb_stat = {
            "main_loss": main_loss,
            "latent_loss": latent_loss,
            "ae_loss": ae_loss_info,
            "aux_loss": aux_loss,
            "codebook_loss": codebook_loss,
            "orthog_loss": orthog_loss,
            "step_time": time_ms,
            "tokens_per_second": tokens_per_second,
            "grad_norm": grad_norm,
            "token_rate": token_rate,
        }
        if config.lm_head_c_mode:
            lm_head_wandb_stat = {
                "y_vec_loss": y_vec_loss,
                "vec_label_loss": vec_label_loss,
                "lm_head_aux_loss": lm_head_aux_loss,
            }
            wandb_stat.update(lm_head_wandb_stat)
        wandb_logger.log(wandb_stat, step=step)

    if config.save and (
        last_step
        or (config.save_before_cooldown and step == wsd_save_steps)
        or (
            step > 0 and save_every and step % save_every == 0 and step < wsd_save_steps
        )
        or (config.beginning_save_step > 0 and step == config.beginning_save_step)
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
        saved_checkpoints = sorted(
            glob.glob(os.path.join(dir, "*.pth")), key=os.path.getmtime, reverse=True
        )
        for old_checkpoint in saved_checkpoints[config.keep_latest_k :]:
            os.remove(old_checkpoint)
            print(f"deleted old checkpoint: {old_checkpoint}")
        torch.cuda.synchronize()
        last_time = time.time()

    torch.cuda.synchronize()
    last_time = time.time()

wandb_logger.finish()
if ddp:
    dist.destroy_process_group()
