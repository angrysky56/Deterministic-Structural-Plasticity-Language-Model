# -*- coding: utf-8 -*-
"""DSP-LM: Deterministic Structural Plasticity Language Model.

Colab-trainable dendritic language model with an SSM temporal mixer.

This script is the source of truth; convert to a notebook for Colab:
    jupytext --to notebook colab_trainable_dendritic_lm.py

Quick local sanity check (no downloads, tiny synthetic data):
    DSP_SMOKE=1 python colab_trainable_dendritic_lm.py

Architecture (v2):
    Each block is TWO residual sublayers, cleanly separated:

      1. ResonatorSSM  -- the temporal/sequence mixer.
         A diagonal, damped-complex-pole state-space model (S4D-style).
         Each channel is literally a damped resonator: pole = -exp(a) + i*w
         (negative real part = damping, imaginary part = oscillation).
         Unbounded receptive field, O(N log N) via FFT convolution.
         THIS is the "causal resonant field mixing" the project describes,
         and it is what makes long sequence length actually work -- the old
         dilated Conv1d only saw ~100 tokens no matter how long the input.

      2. DendriticMLP  -- the per-token nonlinear compute (attention-free
         FFN replacement). Input is fanned out into many independent
         "branches", each of which locally solves nonlinear logic before an
         asymmetric soma gate integrates them (Type-II-error avoidance:
         structurally unresolved states are pushed toward zero).

Separating time-mixing (SSM) from channel-mixing (dendrite) is what lets the
model scale to long context without the quadratic cost of self-attention and
without the finite window of a plain convolution.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

# ==========================================================================
# 1. TEMPORAL MIXER  --  Diagonal damped-resonator SSM (S4D-style)
# ==========================================================================


class ResonatorSSMKernel(nn.Module):
    """Computes the SSM convolution kernel from diagonal damped-complex poles.

    Follows the S4D parameterization (Gu et al., NeurIPS 2022). Each of the
    ``d_model`` channels owns ``n_states`` complex conjugate poles. A pole is
    ``A = -exp(log_A_real) + i * A_imag`` so the real part is always negative
    (a stable, damped resonator) and the imaginary part sets its resonant
    frequency. The kernel is materialised only when needed and consumed by an
    FFT convolution, giving an effectively infinite, causal receptive field.
    """

    def __init__(
        self,
        d_model: int,
        n_states: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        # Store N/2 conjugate pairs; the kernel takes 2*Re(...) to recover the
        # full real response.
        half = n_states // 2

        # Per-channel timestep (discretisation step of the continuous system).
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Output/readout weights C (complex), stored as real view for autograd.
        c = torch.randn(d_model, half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c))

        # Pole parameterisation. log_A_real -> damping; A_imag -> frequency.
        # S4D-Lin init: real part 1/2, frequencies pi * n.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, half)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(half).repeat(d_model, 1).float())

    def forward(self, length: int) -> torch.Tensor:
        """Return the real causal kernel of shape ``(d_model, length)``."""
        # Kernel math runs in float32 (complex ops are unsupported in bf16).
        dt = torch.exp(self.log_dt)                          # (H,)
        c = torch.view_as_complex(self.C)                    # (H, N/2)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag   # (H, N/2) stable

        # Discretise: dtA = dt * A, folding (e^{dtA}-1)/A into C (B ~ ones).
        dt_a = a * dt.unsqueeze(-1)                           # (H, N/2)
        c = c * (torch.exp(dt_a) - 1.0) / a                  # (H, N/2)

        # Vandermonde: powers of the discrete pole along the sequence axis.
        arange = torch.arange(length, device=a.device)
        powers = torch.exp(dt_a.unsqueeze(-1) * arange)      # (H, N/2, L)
        kernel = 2.0 * torch.einsum("hn,hnl->hl", c, powers).real  # (H, L)
        return kernel


class ResonatorSSM(nn.Module):
    """Causal SSM sequence mixer using FFT convolution with a skip term."""

    def __init__(self, d_model: int, n_states: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel = ResonatorSSMKernel(d_model, n_states=n_states)
        self.D = nn.Parameter(torch.randn(d_model))  # direct feed-through skip
        # Gated output projection (GLU) — standard in modern SSM blocks.
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> operate along time.
        length = x.size(1)
        u = x.transpose(1, 2)  # (B, H, T)

        # Kernel + FFT convolution in float32 for numerical stability.
        kernel = self.kernel(length).to(torch.float32)     # (H, T)
        u32 = u.to(torch.float32)
        n_fft = 2 * length
        k_f = torch.fft.rfft(kernel, n=n_fft)              # (H, T_f)
        u_f = torch.fft.rfft(u32, n=n_fft)                 # (B, H, T_f)
        y = torch.fft.irfft(u_f * k_f, n=n_fft)[..., :length]  # (B, H, T)
        y = y + u32 * self.D.unsqueeze(-1)
        y = y.transpose(1, 2).to(x.dtype)                  # (B, T, H)

        # Gated readout.
        return self.out_proj(y) * F.silu(self.gate_proj(x))


# ==========================================================================
# 2. CHANNEL MIXER  --  Dendritic branch logic (per-token, attention-free)
# ==========================================================================


class DendriticMLP(nn.Module):
    """Per-token nonlinear compute via independent dendritic branches.

    The layer fans the token into ``num_branches`` branches, each of width
    ``branch_dim``. Within its own sub-space each branch locally solves
    nonlinear logic (SiLU), then an asymmetric soma gate decides whether the
    branch has "structurally resolved" — a steep sigmoid that pushes unresolved
    branches toward zero (Type-II-error avoidance). The soma (a down
    projection) integrates the surviving branch *vectors* back to d_model.

    Fix vs v1 (the "collapse"): v1 reduced each branch to a single SCALAR
    before integrating, so a huge synaptic projection
    (d_model * num_branches * branch_hidden) fed a d_model-wide scalar
    bottleneck and threw away almost all of its own capacity. Here the gate is
    applied to the full branch VECTOR and the entire branch_dim signal flows
    into the soma — a gated-GLU dendrite. Non-lossy, fewer params, and the
    hidden width ``num_branches * branch_dim`` is a clean FFN-style expansion.
    """

    def __init__(
        self,
        d_model: int,
        num_branches: int = 8,
        branch_dim: int = 256,
        threshold: float = 0.1,
        gate_steepness: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        self.branch_dim = branch_dim
        self.d_ff = num_branches * branch_dim  # total hidden width
        self.threshold = threshold
        self.gate_steepness = gate_steepness

        self.value_proj = nn.Linear(d_model, self.d_ff)   # branch value vectors
        self.branch_gate = nn.Linear(d_model, num_branches)  # per-branch logic
        self.out_proj = nn.Linear(self.d_ff, d_model)     # soma integration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        # Each branch computes a local nonlinear value vector.
        value = F.silu(self.value_proj(x)).view(b, t, self.num_branches, self.branch_dim)

        # Asymmetric soma gate: a branch fires only once structurally resolved.
        logit = self.branch_gate(x)                       # (B, T, num_branches)
        gate = torch.sigmoid(self.gate_steepness * (torch.sigmoid(logit) - self.threshold))

        # Gate the full branch vectors (not a scalar), then integrate.
        gated = value * gate.unsqueeze(-1)                # (B, T, num_branches, branch_dim)
        return self.out_proj(gated.reshape(b, t, self.d_ff))


# ==========================================================================
# 3. BLOCK + FULL MODEL
# ==========================================================================


class DendriticResonatorBlock(nn.Module):
    """One block: pre-norm SSM time-mix + pre-norm dendritic channel-mix."""

    def __init__(
        self,
        d_model: int,
        n_states: int = 64,
        num_branches: int = 8,
        branch_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = ResonatorSSM(d_model, n_states=n_states)
        self.norm2 = nn.LayerNorm(d_model)
        self.dendrite = DendriticMLP(
            d_model, num_branches=num_branches, branch_dim=branch_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.dendrite(self.norm2(x))
        return x


class VectorizedDendriticLM(nn.Module):
    """DSP-LM: dendritic branches over a diagonal-SSM temporal backbone.

    No positional embeddings are needed: the SSM is inherently sequential and
    causal, so position is encoded by the recurrence itself.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        depth: int = 6,
        n_states: int = 64,
        num_branches: int = 8,
        branch_dim: int = 256,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                DendriticResonatorBlock(
                    d_model,
                    n_states=n_states,
                    num_branches=num_branches,
                    branch_dim=branch_dim,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.lm_head(self.norm_out(x))

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        idx = start_tokens
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        if was_training:
            self.train()
        return idx


# ==========================================================================
# 4. CONFIG
# ==========================================================================


@dataclass
class Config:
    # Model (A100-scale defaults).
    d_model: int = 512
    depth: int = 6
    n_states: int = 64        # SSM states per channel (N/2 conjugate pole pairs)
    num_branches: int = 8     # dendritic branches per token
    branch_dim: int = 256     # width of each branch (d_ff = num_branches*branch_dim)
    use_checkpoint: bool = True

    # Data / optimisation.
    seq_len: int = 2048          # SSM gives real long context (was 256)
    batch_size: int = 8
    grad_accum: int = 8          # effective batch 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1    # cosine floor as a fraction of lr

    # Curriculum.
    steps_per_substep: int = 500
    log_every: int = 100

    output_dir: str = "./checkpoints/DSP_LM"
    resume: bool = True          # resume from latest checkpoint if present

    # Dataset slate: a balanced, non-gated, all-streamable "curriculum of
    # courses". Each entry is (repo, config_or_None). All four share the spirit
    # of the curriculum — formal reasoning, then course/textbook material.
    #
    #   logic      reasoning-core/procedural-pretraining-pile  ~7.3GB  formal, correct-by-design
    #   math       cosmopedia/khanacademy                      ~108MB  Khan Academy course prose
    #   physics    cosmopedia/openstax                         ~668MB  OpenStax textbooks (incl. Univ. Physics)
    #   philosophy sayhan/strix-philosophy-qa                  ~391MB  134k philosophy Q&A
    #
    # Streaming means pool size barely matters — you only pull what
    # steps*batch*seq_len demands. Scale-up swaps (bigger, still non-gated):
    # math -> ("HuggingFaceTB/cosmopedia", "auto_math_text") ~8.8GB;
    # humanities -> ("HuggingFaceTB/cosmopedia", "stanford") ~6.3GB.
    repos: dict = field(
        default_factory=lambda: {
            "logic": ("reasoning-core/procedural-pretraining-pile", None),
            "math": ("HuggingFaceTB/cosmopedia", "khanacademy"),
            "physics": ("HuggingFaceTB/cosmopedia", "openstax"),
            "philosophy": ("sayhan/strix-philosophy-qa", None),
        }
    )


# ==========================================================================
# 5. DATA  --  schema-aware extraction, robust loading, sequence packing
# ==========================================================================


def make_extractors():
    """Per-dataset text extractors. The datasets have very different schemas;
    a generic 'first string field' guess trains on labels/titles, not content.
    """

    def logic(item: dict) -> str:  # reasoning-core: prompt / cot / answer
        parts = [item.get("prompt", ""), item.get("cot", ""), item.get("answer", "")]
        return "\n".join(p for p in parts if isinstance(p, str) and p)

    def course_text(item: dict) -> str:  # cosmopedia (khanacademy/openstax): 'text'
        t = item.get("text", "")
        return t if isinstance(t, str) else ""

    def philosophy(item: dict) -> str:  # strix: question / answer (csv columns)
        q, a = item.get("question", ""), item.get("answer", "")
        if q or a:
            return "\n".join(p for p in (q, a) if isinstance(p, str) and p)
        for key in ("text", "content", "output"):
            if isinstance(item.get(key), str):
                return item[key]
        return ""

    # math and physics both come from Cosmopedia's uniform 'text' schema.
    return {
        "logic": logic,
        "math": course_text,
        "physics": course_text,
        "philosophy": philosophy,
    }


class PackedTokenStream:
    """Packs tokenized documents into contiguous (seq_len+1) blocks.

    Far more efficient than padding every document to max_length: no padding
    waste, every position contributes a real next-token loss, and long-context
    windows are actually filled. Documents are separated by the EOS token.
    """

    def __init__(self, ds_iter, extractors, phase_datasets, tokenizer, seq_len):
        self.ds_iter = ds_iter
        self.extractors = extractors
        self.phase_datasets = phase_datasets
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos = tokenizer.eos_token_id
        self.buffer: list[int] = []

    def _next_text(self):
        # interleave_datasets loses which sub-dataset a row came from, so we
        # detect schema by trying each phase extractor and taking the longest.
        item = next(self.ds_iter)
        best = ""
        for name in self.phase_datasets:
            try:
                t = self.extractors[name](item)
            except Exception:
                t = ""
            if len(t) > len(best):
                best = t
        return best

    def get_block(self, device):
        """Return one (1, seq_len) x / y pair of packed tokens."""
        need = self.seq_len + 1
        while len(self.buffer) < need:
            try:
                text = self._next_text()
            except StopIteration:
                if len(self.buffer) < need:
                    return None, None
                break
            if text and len(text.strip()) > 20:
                self.buffer.extend(self.tokenizer.encode(text))
                self.buffer.append(self.eos)
        block = self.buffer[:need]
        self.buffer = self.buffer[need:]
        ids = torch.tensor(block, dtype=torch.long, device=device)
        return ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)


def get_packed_batch(streams, batch_size, device):
    """Assemble a batch by drawing one block from a round-robin of streams."""
    xs, ys = [], []
    for _ in range(batch_size):
        stream = streams[len(xs) % len(streams)]
        x, y = stream.get_block(device)
        if x is None:
            break
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return torch.cat(xs, 0), torch.cat(ys, 0)


# ==========================================================================
# 6. CHECKPOINTING
# ==========================================================================


def save_checkpoint(path, model, optimizer, scheduler, step, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("step", 0)


# ==========================================================================
# 7. TRAINING
# ==========================================================================


def build_model_and_optim(cfg: Config, vocab_size: int, device: str):
    model = VectorizedDendriticLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        depth=cfg.depth,
        n_states=cfg.n_states,
        num_branches=cfg.num_branches,
        branch_dim=cfg.branch_dim,
        use_checkpoint=cfg.use_checkpoint,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    return model, optimizer


def make_scheduler(cfg: Config, optimizer, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def smoke_test(cfg: Config, device: str) -> None:
    """Validate the whole train step on synthetic data — no downloads."""
    print("=== SMOKE TEST (synthetic data, tiny model) ===")
    cfg.d_model, cfg.depth, cfg.seq_len = 64, 2, 128
    cfg.batch_size, cfg.grad_accum, cfg.n_states = 2, 2, 16
    cfg.num_branches, cfg.branch_dim = 4, 32
    vocab_size = 512
    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    scheduler = make_scheduler(cfg, optimizer, total_steps=10)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    for step in range(6):
        x = torch.randint(0, vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        y = torch.randint(0, vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        (loss / cfg.grad_accum).backward()
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), "NaN/Inf loss"
        print(f"  step {step} loss {loss.item():.4f}")
    gen = model.generate(x[:, :4], max_new_tokens=8, top_k=10)
    assert gen.shape == (cfg.batch_size, 12)
    print(f"generate OK -> {tuple(gen.shape)}\n=== SMOKE TEST PASSED ===")


def main() -> None:
    from datasets import interleave_datasets, load_dataset
    from transformers import AutoTokenizer

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if os.environ.get("DSP_SMOKE") == "1":
        smoke_test(cfg, device)
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size} tokens (GPT-2 BPE)")

    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Robust dataset loading: drop anything that fails (e.g. gated physics).
    extractors = make_extractors()
    loaded = {}
    for name, (repo, cfg_name) in cfg.repos.items():
        try:
            loaded[name] = load_dataset(repo, cfg_name, split="train", streaming=True)
            tag = f"{repo}:{cfg_name}" if cfg_name else repo
            print(f"  loaded {name:<10} <- {tag}")
        except Exception as exc:  # gated / offline / renamed
            print(f"  SKIP  {name:<10} <- {repo}  ({type(exc).__name__}: {exc})")

    phases = [
        {"name": "Phase_1_Logic_Dominant", "desc": "Foundational Logic (90/10)",
         "datasets": ["logic", "math"], "mixtures": [[0.90, 0.10]]},
        {"name": "Phase_2_Math_Introduction", "desc": "Interpolating Logic and Math",
         "datasets": ["logic", "math"], "mixtures": [[0.30, 0.70], [0.50, 0.50]]},
        {"name": "Phase_3_Physics_Application", "desc": "Physics & Derivations",
         "datasets": ["logic", "math", "physics"],
         "mixtures": [[0.10, 0.20, 0.70], [0.15, 0.25, 0.60], [0.20, 0.30, 0.50]]},
        {"name": "Phase_4_Philosophy_Meta", "desc": "Philosophy & Conceptual Analysis",
         "datasets": ["logic", "math", "physics", "philosophy"],
         "mixtures": [[0.25, 0.25, 0.25, 0.25]]},
    ]

    total_substeps = sum(len(p["mixtures"]) for p in phases)
    total_steps = total_substeps * cfg.steps_per_substep // cfg.grad_accum
    scheduler = make_scheduler(cfg, optimizer, total_steps)

    os.makedirs(cfg.output_dir, exist_ok=True)
    latest = os.path.join(cfg.output_dir, "latest.pt")
    global_step = 0
    if cfg.resume and os.path.exists(latest):
        global_step = load_checkpoint(latest, model, optimizer, scheduler, device)
        print(f"Resumed from {latest} at optimizer step {global_step}")

    print("\nStarting curriculum training...")
    for phase in phases:
        # Keep only datasets that actually loaded; renormalise the mixture.
        avail = [d for d in phase["datasets"] if d in loaded]
        if not avail:
            print(f"Skipping {phase['name']} — no datasets available.")
            continue
        print(f"\n{'=' * 60}\n{phase['name']}\n{phase['desc']}\n{'=' * 60}")

        for substep_idx, probs in enumerate(phase["mixtures"]):
            kept = [(d, p) for d, p in zip(phase["datasets"], probs) if d in loaded]
            names = [d for d, _ in kept]
            weights = [p for _, p in kept]
            weights = [w / sum(weights) for w in weights]  # renormalise
            print(f"\n  Substep {substep_idx + 1}/{len(phase['mixtures'])} - "
                  f"{dict(zip(names, [round(w, 3) for w in weights]))}")

            merged = interleave_datasets(
                [loaded[n] for n in names],
                probabilities=weights,
                seed=3407 + substep_idx,
                stopping_strategy="all_exhausted",
            )
            stream = PackedTokenStream(iter(merged), extractors, names, tokenizer, cfg.seq_len)
            streams = [stream]  # single packed stream; batch draws multiple blocks

            optimizer.zero_grad(set_to_none=True)
            for step in range(cfg.steps_per_substep):
                model.train()
                x, y = get_packed_batch(streams, cfg.batch_size, device)
                if x is None:
                    print("  Stream exhausted early.")
                    break

                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

                (loss / cfg.grad_accum).backward()

                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if step % cfg.log_every == 0 or step == cfg.steps_per_substep - 1:
                    ppl = math.exp(min(20, loss.item()))
                    print(f"  [{phase['name']} | sub {substep_idx + 1}] step {step:04d} "
                          f"| loss {loss.item():.4f} | ppl {ppl:8.1f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e}")

            # Flush trailing accumulated gradients, then checkpoint.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            out_dir = os.path.join(cfg.output_dir, phase["name"], f"substep_{substep_idx + 1}")
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(os.path.join(out_dir, "checkpoint.pt"),
                            model, optimizer, scheduler, global_step, cfg)
            save_checkpoint(latest, model, optimizer, scheduler, global_step, cfg)
            tokenizer.save_pretrained(out_dir)
            print(f"  Saved checkpoint -> {out_dir}")

    print("\nTraining complete.")

    print("\n--- Generation test ---")
    seed_text = "The physical principles governing the universe state that"
    seed_idx = tokenizer.encode(seed_text, return_tensors="pt").to(device)
    generated = model.generate(seed_idx, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f"Seed: {seed_text!r}")
    print("Generated:\n" + tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
