"""
DSP-LM fast-iteration research harness. Single-GPU, fixed time budget.

Methodology borrowed from Karpathy's `autoresearch` project: a fixed 5-minute
wall-clock training budget, a fixed reference dataset/tokenizer/eval metric
(val_bpb, vocab-size independent) in `prepare_data.py`, and a single file here
that's free to iterate on -- architecture, optimizer, hyperparameters, batch
size. Unlike autoresearch, this harness uses DSP-LM's *actual* model code
(ResonatorSSM + DendriticMLP) imported directly from the project's source of
truth (`colab_trainable_dendritic_lm.py`) instead of Karpathy's GPT+attention
baseline, so the point of this harness is fast architecture/hyperparameter
search for DSP-LM itself, not a from-scratch reimplementation.

Usage: uv run research_harness/train_harness.py
Run from anywhere -- paths are resolved relative to this file.

Log results to research_harness/results.tsv after each run:
    commit\tval_bpb\tmemory_gb\tstatus\tdescription
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Make the DSP-LM project root importable so we can pull the real model
# classes from colab_trainable_dendritic_lm.py -- this harness is a component
# of the DSP-LM project (self-contained within this one repo), not a separate
# project reaching into another codebase.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from colab_trainable_dendritic_lm import DendriticResonatorBlock  # noqa: E402
from prepare_data import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb  # noqa: E402

# ---------------------------------------------------------------------------
# DSP-LM model, wired to the harness's fixed contract:
#   forward(idx, targets=None, reduction='mean') -> loss (scalar or per-token)
#   num_scaling_params() -> dict with 'total'
#   estimate_flops()     -> approx FLOPs/token (diagnostic only, not scored)
#   setup_optimizer(...) -> torch.optim.Optimizer
# ---------------------------------------------------------------------------


@dataclass
class DSPLMConfig:
    sequence_len: int = 256
    vocab_size: int = 8192
    d_model: int = 256
    depth: int = 4
    n_states: int = 32
    num_branches: int = 8
    branch_dim: int = 128
    use_checkpoint: bool = False


class DSPLMHarness(nn.Module):
    """DendriticResonatorBlock stack + tied embedding/head, for the harness."""

    def __init__(self, config: DSPLMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                DendriticResonatorBlock(
                    config.d_model,
                    n_states=config.n_states,
                    num_branches=config.num_branches,
                    branch_dim=config.branch_dim,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self.apply(self._init_weights)
        # Scale residual output projections by 1/sqrt(2*depth) (GPT-2 / nanoGPT
        # trick) so the residual stream doesn't grow with depth.
        residual_std = 0.02 / math.sqrt(2 * config.depth)
        for pname, p in self.named_parameters():
            if pname.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        blocks = sum(p.numel() for p in self.blocks.parameters())
        norm_out = sum(p.numel() for p in self.norm_out.parameters())
        total = sum(p.numel() for p in self.parameters())  # tied lm_head not double-counted
        return {"embedding": embedding, "blocks": blocks, "norm_out": norm_out, "total": total}

    def estimate_flops(self):
        """Rough per-token FLOPs estimate (forward+backward). Diagnostic only
        -- used for the printed mfu_percent, NOT the scored metric (that's
        val_bpb via evaluate_bpb). The dense projections dominate at this
        scale; the SSM's FFT convolution is O(N log T) per channel rather than
        a matmul, approximated with a small constant-factor term so mfu%
        stays in a sane ballpark rather than being exactly right.
        """
        nparams = sum(p.numel() for p in self.parameters())
        embed_params = self.embedding.weight.numel()
        dense_flops = 6 * (nparams - embed_params)
        t = self.config.sequence_len
        half_states = self.config.n_states // 2
        fft_flops_per_token = sum(
            10 * self.config.d_model * half_states * max(1, math.log2(2 * t)) for _ in self.blocks
        )
        return dense_flops + fft_flops_per_token

    def setup_optimizer(self, lr, weight_decay, betas):
        """Weight-decay only 2-D projection matrices. Exclude biases,
        LayerNorm gains (ndim < 2), and the SSM kernel poles (log_A_real,
        A_imag, log_dt, C) + skip D -- decaying resonator parameters is
        harmful (same convention as the main DSP-LM training script)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or ".kernel." in name or name.endswith(".D"):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            dict(kind="adamw", params=decay, lr=lr, betas=betas, weight_decay=weight_decay),
            dict(kind="adamw", params=no_decay, lr=lr, betas=betas, weight_decay=0.0),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        x = self.embedding(idx)
        for block in self.blocks:
            if self.config.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm_out(x)
        logits = self.lm_head(x).float()

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
D_MODEL = 256
DEPTH = 4
N_STATES = 32  # SSM states per channel (n_states/2 conjugate pole pairs)
NUM_BRANCHES = 8
BRANCH_DIM = 128  # d_ff = NUM_BRANCHES * BRANCH_DIM
# exp2 (batch 128) and exp3 (n_states=16, branch_dim=64) were both worse than
# this config -- see results.tsv. Reverted back to exp1's settings, the best
# of the three tried so far.
USE_CHECKPOINT = False  # small model, VRAM isn't the bottleneck at this scale

# Optimization
TOTAL_BATCH_SIZE = 2**14  # ~16K tokens per optimizer step -- kept equal to exp1-6 for a clean comparison
DEVICE_BATCH_SIZE = 16  # exp7: seq_len 256->1024 (4x), batch 64->16 (1/4x) -- same tokens/step, arranged
                        # as fewer/longer sequences instead of more/shorter ones
LR = 4e-3  # exp6: exp5's 2e-3 only gave a small further gain over exp4 -- probing for the ceiling
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.3  # exp4: shortened from 0.5 -- spend more time at peak LR
FINAL_LR_FRAC = 0.0

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12  # reference point only; this runs on an RTX 3060

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = DSPLMConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    d_model=D_MODEL,
    depth=DEPTH,
    n_states=N_STATES,
    num_branches=NUM_BRANCHES,
    branch_dim=BRANCH_DIM,
    use_checkpoint=USE_CHECKPOINT,
)
print(f"Model config: {asdict(config)}")

model = DSPLMHarness(config).to(device)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
        f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | "
        f"remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
