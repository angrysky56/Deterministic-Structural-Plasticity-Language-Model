r"""
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
from dataclasses import asdict, dataclass
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

from prepare_data import (  # noqa: E402
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_bpb,
    make_dataloader,
)

from colab_trainable_dendritic_lm import (  # noqa: E402
    DendriticResonatorBlock,
    dendrite_diagnostics,
    print_dendrite_diagnostics,
)

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
    d_model_base: int = 256  # muP reference width -- see DSPLMHarness docstring
    # Dendrite mechanism under test -- see DendriticMLP's docstring in the
    # source of truth. Override per run with
    #   DENDRITE=tree uv run research_harness/train_harness.py
    # Nested ablation: baseline -> nmda -> compart -> tree, each adding one
    # factor at matched parameter count.
    dendrite_variant: str = "baseline"


class DSPLMHarness(nn.Module):
    """DendriticResonatorBlock stack + tied embedding/head, muP-parametrized.

    muP (Yang & Hu et al., "Tensor Programs V") for AdamW: "hidden" matrix
    layers -- weights whose fan_in scales with model width, i.e. every
    Linear in the SSM/dendrite sublayers except the embedding/tied head --
    get init std proportional to 1/sqrt(fan_in) (needs no calibration, it's
    a property of each layer's own shape) and AdamW LR proportional to
    1/fan_in, i.e. 1/d_model here since fan_in scales with d_model across
    our presets. The LR rule needs one anchor point, `d_model_base`: a base
    LR tuned at that reference width transfers to any other width via
    lr * (d_model_base / d_model) instead of being re-guessed at each scale.
    This is the correct Adam exponent per Tensor Programs V -- note it is
    NOT the same as the ad hoc 1/sqrt(width) heuristic used in some GPT
    baselines (e.g. Karpathy's autoresearch `dmodel_lr_scale`); that
    heuristic is a coarser approximation, not the derived muP rule.

    Embedding (tied to lm_head) and "structural" parameters (LayerNorm,
    biases, and DSP-LM's own SSM pole parameters log_A_real/A_imag/log_dt/C)
    keep width-independent LR/init -- the standard muP treatment for
    input/output layers and non-matrix parameters. The SSM poles and
    dendritic branch-gate are novel components with no published muP
    analysis, so they simply stay on the pre-existing no-decay,
    width-independent-LR treatment rather than a rule that hasn't been
    derived for them.

    Caveat, stated plainly: muP formally guarantees transfer across WIDTH at
    FIXED DEPTH. Our own preset progression changes depth alongside d_model,
    so applying this transfer rule across those configs is width-transfer
    under a simultaneous depth change, not the textbook-clean case. Treat a
    transferred LR as a much better-grounded starting point than an
    unguided guess or a borrowed number, not as a proven optimum.
    """

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
                    dendrite_variant=config.dendrite_variant,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self._init_weights_mup()

    def _init_weights_mup(self) -> None:
        cfg = self.config
        # Gain calibrated so a hidden layer with fan_in == d_model_base gets
        # std == 0.02 -- the fixed value the pre-muP harness used and that
        # the local exp1-8 sweep validated at d_model=256. Other widths then
        # get the *correct* fan_in-scaled std automatically, not a guess.
        gain = 0.02 * math.sqrt(cfg.d_model_base)
        embed_id = id(self.embedding.weight)  # lm_head.weight is the same tensor (tied)

        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module.weight) != embed_id:
                fan_in = module.weight.shape[1]
                nn.init.normal_(module.weight, mean=0.0, std=gain / math.sqrt(fan_in))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Embedding/tied head: width-independent "input layer" init.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # GPT-2/nanoGPT residual-scaling trick, folded into the muP fan_in
        # std (multiplicatively) rather than overwriting it outright, so
        # out_proj layers keep their width scaling AND their depth scaling.
        for pname, p in self.named_parameters():
            if pname.endswith("out_proj.weight"):
                fan_in = p.shape[1]
                std = (gain / math.sqrt(fan_in)) / math.sqrt(2 * cfg.depth)
                nn.init.normal_(p, mean=0.0, std=std)

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        blocks = sum(p.numel() for p in self.blocks.parameters())
        norm_out = sum(p.numel() for p in self.norm_out.parameters())
        total = sum(
            p.numel() for p in self.parameters()
        )  # tied lm_head not double-counted
        return {
            "embedding": embedding,
            "blocks": blocks,
            "norm_out": norm_out,
            "total": total,
        }

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
            10 * self.config.d_model * half_states * max(1, math.log2(2 * t))
            for _ in self.blocks
        )
        return dense_flops + fft_flops_per_token

    def _classify_params(self):
        """Split params into the 3 muP-aware groups shared by both optimizer
        paths below: embedding (tied to lm_head), hidden matrices (SSM
        out_proj/gate_proj, dendrite value_proj/branch_gate/out_proj -- the
        only params that are both ndim>=2 AND not name-excluded), and
        structural/no_decay (LayerNorm, biases, and -- by NAME, not ndim --
        the SSM kernel poles log_A_real/A_imag/log_dt/C + skip D. This
        matters for Muon: log_A_real and A_imag are themselves 2D tensors
        (d_model, n_states/2), so an ndim-only filter would incorrectly feed
        them to Muon's orthogonalization, which has no clear meaning for a
        per-channel pole array. The explicit ".kernel." name check is what
        keeps them out regardless of which optimizer consumes this split.
        """
        embed_id = id(self.embedding.weight)
        embedding_params, hidden_params, no_decay_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) == embed_id:
                embedding_params.append(p)
            elif p.ndim < 2 or ".kernel." in name or name.endswith(".D"):
                no_decay_params.append(p)
            else:
                hidden_params.append(p)
        return embedding_params, hidden_params, no_decay_params

    def setup_optimizer(self, lr, weight_decay, betas):
        """3 muP-aware param groups, all AdamW (see class docstring for the
        rule): embedding gets width-independent LR == `lr`, no weight decay;
        hidden matrices get LR == `lr` * (d_model_base / d_model) -- the muP
        AdamW transfer rule -- with weight decay applied; structural/no_decay
        params get width-independent LR == `lr`, no weight decay.

        `lr` is meant to be the base LR *tuned at config.d_model_base*, not
        re-guessed at whatever width this config actually is.
        """
        cfg = self.config
        mup_lr = lr * (cfg.d_model_base / cfg.d_model)
        embedding_params, hidden_params, no_decay_params = self._classify_params()

        param_groups = [
            dict(
                kind="adamw",
                params=embedding_params,
                lr=lr,
                betas=betas,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=hidden_params,
                lr=mup_lr,
                betas=betas,
                weight_decay=weight_decay,
            ),
            dict(
                kind="adamw",
                params=no_decay_params,
                lr=lr,
                betas=betas,
                weight_decay=0.0,
            ),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def setup_optimizer_muon(
        self, embed_lr, muon_lr, weight_decay, betas, muon_momentum=0.95
    ):
        """Same 3-group split as setup_optimizer, but the hidden-matrix group
        runs Muon instead of AdamW; embedding and no_decay/structural groups
        stay on AdamW exactly as before (Muon's own README: "other
        parameters, such as embeddings, classifier heads, and hidden
        gains/biases should be optimized using standard AdamW").

        `muon_lr` is a DIFFERENT hyperparameter than the AdamW `lr` used
        elsewhere in this file -- Muon's update is unit-normalized by its
        Newton-Schulz orthogonalization step (spectral norm ~1 per update),
        so muon_lr is "learning rate in units of spectral norm per update"
        per upstream's docstring, not directly comparable to an AdamW LR.
        Muon's README also states its LR needs no explicit muP width
        rescaling ("built-in muP scaling... shouldn't need to retune it"),
        so unlike setup_optimizer's hidden group, muon_lr is passed through
        unscaled -- no d_model_base/d_model factor. This is a claim from
        Muon's own docs, not independently re-derived here; the local
        ablation this method exists for is exactly how we'd notice if it
        doesn't hold for DSP-LM's architecture.
        """
        from muon import SingleDeviceMuonWithAuxAdam

        embedding_params, hidden_params, no_decay_params = self._classify_params()

        param_groups = [
            dict(
                use_muon=False,
                params=embedding_params,
                lr=embed_lr,
                betas=betas,
                weight_decay=0.0,
            ),
            dict(
                use_muon=True,
                params=hidden_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
            ),
            dict(
                use_muon=False,
                params=no_decay_params,
                lr=embed_lr,
                betas=betas,
                weight_decay=0.0,
            ),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
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
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction
            )
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
# depth=6, n_states=64 chosen to match the A100 target's ratios (push1/2/3
# all use n_states=64; d_model_base=256 makes THIS config the muP reference
# width itself, so its swept LR needs no transfer to be used here).
D_MODEL = 256
DEPTH = 6
N_STATES = 64  # SSM states per channel (n_states/2 conjugate pole pairs)
NUM_BRANCHES = 8
BRANCH_DIM = 128  # d_ff = NUM_BRANCHES * BRANCH_DIM
USE_CHECKPOINT = False  # small model, VRAM isn't the bottleneck at this scale
D_MODEL_BASE = 256  # muP reference width -- this run IS the base-width sweep
# Dendrite mechanism -- override with DENDRITE=<baseline|nmda|compart|tree>.
# Default stays 'baseline' so every prior number in results.tsv reproduces.
DENDRITE_VARIANT = os.environ.get("DENDRITE", "baseline")

# Optimization
TOTAL_BATCH_SIZE = (
    2**14
)  # ~16K tokens per optimizer step -- kept equal to exp1-6 for a clean comparison
DEVICE_BATCH_SIZE = (
    16  # exp7's setting (best tonight) -- exp8 tried seq_len=2048/batch=8, was worse
)
# Base LR for the muP sweep at d_model_base -- override per run with
# SWEEP_LR=<value> uv run research_harness/train_harness.py
LR = float(
    os.environ.get("SWEEP_LR", 8e-3)
)  # grounded value from the muP sweep (mup_sweep_results.tsv)
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.3  # exp4: shortened from 0.5 -- spend more time at peak LR
FINAL_LR_FRAC = 0.0

# Optimizer selection -- OPTIMIZER=adamw (default) or OPTIMIZER=muon.
# MUON_LR is a SEPARATE hyperparameter from LR/SWEEP_LR above -- Muon's
# update is unit-normalized (spectral norm ~1/step), not comparable to an
# AdamW LR. See DSPLMHarness.setup_optimizer_muon's docstring. Embedding and
# no_decay groups still use AdamW at LR (SWEEP_LR) even when OPTIMIZER=muon.
OPTIMIZER = os.environ.get("OPTIMIZER", "adamw")
MUON_LR = float(os.environ.get("MUON_LR", 0.02))  # Muon upstream default
MUON_MOMENTUM = 0.95

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
# Seed is overridable so an ablation can estimate run-to-run noise before
# claiming a val_bpb difference is real. Default 42 reproduces every earlier
# entry in results.tsv.
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
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
    d_model_base=D_MODEL_BASE,
    dendrite_variant=DENDRITE_VARIANT,
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

if OPTIMIZER == "muon":
    optimizer = model.setup_optimizer_muon(
        embed_lr=LR,
        muon_lr=MUON_LR,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
        muon_momentum=MUON_MOMENTUM,
    )
    print(
        f"Optimizer: Muon (hidden matrices) + AdamW (embed/no_decay) -- muon_lr={MUON_LR}, embed/no_decay lr={LR}"
    )
elif OPTIMIZER == "adamw":
    optimizer = model.setup_optimizer(
        lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS
    )
    print(f"Optimizer: AdamW (all groups) -- base_lr={LR}")
else:
    raise ValueError(f"Unknown OPTIMIZER={OPTIMIZER!r}, expected 'adamw' or 'muon'")

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
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / H100_BF16_PEAK_FLOPS
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
print(f"lr:               {LR}")
print(f"dendrite_variant: {DENDRITE_VARIANT}")
print(f"seed:             {SEED}")

# What each layer LEARNED about its own dendrite mechanism. This is the point
# of the ablation: gate_k rising means the supralinear NMDA transition is being
# used; lambda falling means the layer chose compartmentalisation. Either can
# come out flat, which is a real answer and gets reported as one.
if DENDRITE_VARIANT != "baseline":
    print_dendrite_diagnostics(model)
    _rows = dendrite_diagnostics(model)
    _k = [r["gate_k_mean"] for r in _rows]
    print(f"gate_k_mean:      {sum(_k) / len(_k):.4f}")
    if "lambda" in _rows[0]:
        _l = [r["lambda"] for r in _rows]
        print(f"lambda_mean:      {sum(_l) / len(_l):.4f}")
