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
import random
import time
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
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(
            dt_min
        )
        self.log_dt = nn.Parameter(log_dt)

        # Output/readout weights C (complex), stored as real view for autograd.
        c = torch.randn(d_model, half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c))

        # Pole parameterisation. log_A_real -> damping; A_imag -> frequency.
        # S4D-Lin init: real part 1/2, frequencies pi * n.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, half)))
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(half).repeat(d_model, 1).float()
        )

    def _discretise(self):
        """Shared discretisation used by both convolutional and recurrent paths.

        Returns ``(A_bar, B_bar, C)`` where:
        - ``A_bar = exp(dt * A)`` — discrete pole (damped complex), (H, N/2)
        - ``B_bar = (A_bar - 1) / A`` — ZOH-discretised input matrix, (H, N/2)
        - ``C`` — readout weights (unmodified), (H, N/2)
        """
        dt = torch.exp(self.log_dt)  # (H,)
        c = torch.view_as_complex(self.C)  # (H, N/2)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        dt_a = a * dt.unsqueeze(-1)  # (H, N/2)
        a_bar = torch.exp(dt_a)  # (H, N/2)
        b_bar = (a_bar - 1.0) / a  # (H, N/2)
        return a_bar, b_bar, c

    def forward(self, length: int) -> torch.Tensor:
        """Return the real causal kernel of shape ``(d_model, length)``."""
        # Kernel math runs in float32 (complex ops are unsupported in bf16).
        a_bar, b_bar, c = self._discretise()
        c_mod = c * b_bar  # fold B into C for the convolutional form

        # Vandermonde: powers of the discrete pole along the sequence axis.
        dt_a = torch.log(a_bar)  # recover dt*A for exponentiation
        arange = torch.arange(length, device=a_bar.device)
        powers = torch.exp(dt_a.unsqueeze(-1) * arange)  # (H, N/2, L)
        kernel = 2.0 * torch.einsum("hn,hnl->hl", c_mod, powers).real  # (H, L)
        return kernel

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised hidden state: ``(B, H, N/2)`` complex float32."""
        half = self.log_A_real.shape[1]
        h = self.log_A_real.shape[0]
        return torch.zeros(batch_size, h, half, dtype=torch.cfloat, device=device)

    def step(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step (the dual of the FFT convolutional form).

        Args:
            x_t: Input token embedding, ``(B, H)`` real.
            h:   Hidden state from the previous step, ``(B, H, N/2)`` complex.

        Returns:
            ``(y_t, h_new)`` where ``y_t`` is ``(B, H)`` real and ``h_new``
            has the same shape as ``h``.
        """
        a_bar, b_bar, c = self._discretise()  # (H, N/2) each
        # State update: h_new = A_bar * h + B_bar * x_t
        h_new = a_bar * h + b_bar * x_t.to(torch.cfloat).unsqueeze(-1)
        # Readout: y_t = 2 * Re(C · h_new)  (conjugate-pair reconstruction)
        y_t = 2.0 * torch.einsum("hn,bhn->bh", c, h_new).real
        return y_t, h_new


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
        """Convolutional (training) path: process a full sequence via FFT."""
        # x: (B, T, H) -> operate along time.
        length = x.size(1)
        u = x.transpose(1, 2)  # (B, H, T)

        # Kernel + FFT convolution in float32 for numerical stability.
        kernel = self.kernel(length).to(torch.float32)  # (H, T)
        u32 = u.to(torch.float32)
        n_fft = 2 * length
        k_f = torch.fft.rfft(kernel, n=n_fft)  # (H, T_f)
        u_f = torch.fft.rfft(u32, n=n_fft)  # (B, H, T_f)
        y = torch.fft.irfft(u_f * k_f, n=n_fft)[..., :length]  # (B, H, T)
        y = y + u32 * self.D.unsqueeze(-1)
        y = y.transpose(1, 2).to(x.dtype)  # (B, T, H)

        # Gated readout.
        return self.out_proj(y) * F.silu(self.gate_proj(x))

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised SSM hidden state."""
        return self.kernel.initial_state(batch_size, device)

    def step(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent (generation) path: process ONE token, O(1) per step.

        Args:
            x_t: Pre-norm'd token embedding, ``(B, H)``.
            h:   SSM hidden state from prior step, ``(B, H, N/2)`` complex.

        Returns:
            ``(out, h_new)``.
        """
        x32 = x_t.to(torch.float32)
        y_raw, h_new = self.kernel.step(x32, h)
        y_raw = y_raw + x32 * self.D  # skip connection
        y_raw = y_raw.to(x_t.dtype)
        # Gated readout (same gate as forward, but for a single token).
        return self.out_proj(y_raw) * F.silu(self.gate_proj(x_t)), h_new


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

        self.value_proj = nn.Linear(d_model, self.d_ff)  # branch value vectors
        self.branch_gate = nn.Linear(d_model, num_branches)  # per-branch logic
        self.out_proj = nn.Linear(self.d_ff, d_model)  # soma integration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        # Each branch computes a local nonlinear value vector.
        value = F.silu(self.value_proj(x)).view(
            b, t, self.num_branches, self.branch_dim
        )

        # Asymmetric soma gate: a branch fires only once structurally resolved.
        logit = self.branch_gate(x)  # (B, T, num_branches)
        gate = torch.sigmoid(
            self.gate_steepness * (torch.sigmoid(logit) - self.threshold)
        )

        # Gate the full branch vectors (not a scalar), then integrate.
        gated = value * gate.unsqueeze(-1)  # (B, T, num_branches, branch_dim)
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

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised per-block SSM hidden state."""
        return self.ssm.initial_state(batch_size, device)

    def step(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent single-token step through this block.

        Args:
            x_t: Token representation, ``(B, H)``.
            h:   SSM hidden state, ``(B, H, N/2)`` complex.

        Returns:
            ``(x_out, h_new)``.
        """
        ssm_out, h_new = self.ssm.step(self.norm1(x_t), h)
        x_t = x_t + ssm_out
        # Dendrite is per-token — add a T=1 dim, apply, squeeze back.
        x_t = x_t + self.dendrite(self.norm2(x_t).unsqueeze(1)).squeeze(1)
        return x_t, h_new


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

        # Weight init (GPT-2 style). Without this, nn.Embedding defaults to
        # N(0,1); tied to the output head that yields logits ~sqrt(d_model) in
        # scale, so initial loss is ~10x above ln(vocab) and training stalls.
        self.apply(self._init_weights)
        # Scale residual output projections by 1/sqrt(2*depth) so the residual
        # stream doesn't grow with depth (GPT-2 / nanoGPT trick).
        residual_std = 0.02 / math.sqrt(2 * depth)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.lm_head(self.norm_out(x))

    # -- Recurrent (generation) interface --------------------------------

    def initial_states(
        self, batch_size: int, device: torch.device
    ) -> list[torch.Tensor]:
        """Zero-initialised hidden states for every block."""
        return [block.initial_state(batch_size, device) for block in self.blocks]

    def step(
        self, token_ids: torch.Tensor, states: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process a single token through all blocks using the recurrent SSM.

        Args:
            token_ids: ``(B,)`` — one token id per sequence in the batch.
            states:    Per-block hidden states (from ``initial_states`` or a
                       previous ``step`` call).

        Returns:
            ``(logits, new_states)`` where logits is ``(B, vocab_size)``.
        """
        x = self.embedding(token_ids)  # (B, H)
        new_states: list[torch.Tensor] = []
        for block, h in zip(self.blocks, states, strict=False):
            x, h_new = block.step(x, h)
            new_states.append(h_new)
        logits = self.lm_head(self.norm_out(x))  # (B, vocab)
        return logits, new_states

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Autoregressive generation using the O(1)-per-step recurrent form.

        The prompt is processed token-by-token through the recurrent SSM to
        build up hidden state (compressed context), then each new token is
        generated with a single ``step`` call — no FFT re-convolution over
        the growing sequence.
        """
        was_training = self.training
        self.eval()

        batch_size = start_tokens.size(0)
        device = start_tokens.device
        states = self.initial_states(batch_size, device)

        # Prefill: step through the prompt to build up hidden states.
        for t in range(start_tokens.size(1)):
            logits, states = self.step(start_tokens[:, t], states)

        # Generate: each new token is O(1) — just one recurrent step.
        idx = start_tokens
        for _ in range(max_new_tokens):
            scaled = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[:, [-1]]] = -float("inf")
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            logits, states = self.step(next_token.squeeze(1), states)

        if was_training:
            self.train()
        return idx


# ==========================================================================
# 4. CONFIG
# ==========================================================================


# Model-size presets. Each sets architecture (d_model/depth/branch_dim, keeping
# d_ff = 4*d_model with num_branches=8) plus batch/grad_accum/lr. Effective batch
# is held at ~96 across presets. Token needs are ~20 tokens/param (Chinchilla).
#   42m  ~0.8B tokens  fastest, proof-of-concept
#   110m ~2.2B tokens  strong capability-per-hour
#   500m ~6-10B tokens RECOMMENDED max for a single A100 (DEFAULT)
#   1b   ~20B tokens   fits an 80GB A100 but needs ~10+ A100-days — will be badly
#                      undertrained on one GPU; included so you can experiment.
#
# batch_size is tuned for an **80GB** A100 with gradient checkpointing on; watch
# the GPU-RAM gauge and push batch_size higher to fill ~60-70GB (raise grad_accum
# to keep the same effective batch). On a 40GB A100 or a 12GB card, halve/quarter
# batch_size. To trade the freed VRAM for ~30% more speed instead, you can also
# set use_checkpoint=False (works at smaller batch; it OOMs with a big batch).
MODEL_PRESETS = {
    "42m":  {"d_model": 512,  "depth": 6,  "branch_dim": 256,  "batch_size": 48, "grad_accum": 2, "lr": 3e-4},
    "110m": {"d_model": 768,  "depth": 12, "branch_dim": 384,  "batch_size": 48, "grad_accum": 2, "lr": 3e-4},
    "500m": {"d_model": 1536, "depth": 18, "branch_dim": 768,  "batch_size": 32, "grad_accum": 3, "lr": 2e-4},
    "1b":   {"d_model": 2048, "depth": 22, "branch_dim": 1024, "batch_size": 16, "grad_accum": 6, "lr": 1.5e-4},
}


@dataclass
class Config:
    # Pick model scale here; override any field below to customise.
    preset: str = "110m"

    # Architecture — left None to inherit from the preset.
    d_model: int | None = None
    depth: int | None = None
    branch_dim: int | None = None  # width of each branch (d_ff = num_branches*branch_dim)
    n_states: int = 64  # SSM states per channel (N/2 conjugate pole pairs)
    num_branches: int = 8  # dendritic branches per token
    use_checkpoint: bool = True

    # Data / optimisation (batch_size/grad_accum/lr inherit from the preset if None).
    seq_len: int = 2048  # SSM gives real long context (was 256)
    batch_size: int | None = None
    grad_accum: int | None = None
    lr: float | None = None
    # 0.1 is standard (Chinchilla/Llama). Data-constrained scaling work (Lovelace
    # et al. 2026) finds strong decay ~1.0 cuts repetition-overfitting ~70%; go
    # higher (up to ~1.0) if you lean heavily on small/repeated domains.
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1  # LR floor at the end of the final decay
    # WSD schedule: warmup -> stable plateau at full LR -> decay only over the
    # final fraction of training. Keeps every curriculum phase at full LR
    # (cosine-over-whole-run otherwise starves the later physics/philosophy
    # phases). decay_frac is the tail portion spent decaying.
    decay_frac: float = 0.2

    # Curriculum.
    steps_per_substep: int = 3000  # try 500+ for a trial run (see token math note)
    log_every: int = 100
    save_every: int = 3000  # also checkpoint every N optimizer steps (crash safety)

    # Held-out evaluation: hold out the first eval_docs rows of each dataset
    # (train skips them), pack a few blocks each, and report eval loss at every
    # checkpoint to distinguish generalisation from memorisation.
    eval_docs: int = 200
    eval_blocks_per: int = 3
    # Chat template is always applied (teaches User/Assistant turn structure).
    # Prompt-masking is OFF by default: for a from-scratch base model you want
    # every token to teach, and long formal prompts otherwise leave whole packed
    # blocks fully masked (wasted compute, risk of all-ignored batches). Turn it
    # ON for a later dedicated SFT/instruction-polish phase.
    mask_prompt_loss: bool = False

    output_dir: str = "./checkpoints/DSP_LM"
    resume: bool = True  # resume from latest checkpoint (skips finished substeps)

    # Hugging Face Hub: set this to auto-sync checkpoints between Colab and
    # your local machine. Format: "username/repo-name". Created as a PRIVATE
    # repo on first push. Pull locally with: python colab_trainable_dendritic_lm.py pull
    hf_repo: str = "angrysky/dendritic-lm"  # e.g. "your_hf_name/dendritic-lm"

    # Dataset slate: balanced, non-gated, all-streamable. Each entry is
    # (repo, config_or_None). "language" (fineweb-edu) is the general-English
    # backbone — it teaches grammar, vocabulary and world knowledge, present in
    # every phase so fluency is always reinforced (the formal/technical corpora
    # alone would leave the model stilted).
    #
    #   language   HuggingFaceFW/fineweb-edu:sample-10BT      10B tokens          grammar + fluency + knowledge
    #   logic      reasoning-core/procedural-pretraining-pile ~7.3GB / 3.1M rows  formal, correct-by-design
    #   math       open-web-math/open-web-math                14.7B tokens        never runs out
    #   physics    millawell/wikipedia_field_of_science       ~9.6GB science wiki broad science, never runs out
    #   philosophy sayhan/strix-philosophy-qa                 ~391MB / 134k       philosophy Q&A (final phase)
    #
    # Streaming reads only the consumed slice, so pool size is free; these pools
    # are large enough that nothing cycles at these step counts. Q&A/instruction
    # (alpaca, orca-math) is intentionally deferred to a later SFT phase, not the
    # base run. Swap options (all non-gated):
    #   grammar bootstrap -> ("roneneldan/TinyStories", None)     simple stories, teaches basic syntax
    #   logic scale-up    -> ("reasoning-core/basic-procedural", None)   7.6M rows
    #   humanities scale  -> ("HuggingFaceTB/cosmopedia", "stanford")    6.3GB (broader than philosophy)
    repos: dict = field(
        default_factory=lambda: {
            "grammar": ("grammarly/coedit", None),          # explicit grammar correction
            "language": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
            "logic": ("reasoning-core/procedural-pretraining-pile", None),
            "math": ("open-web-math/open-web-math", None),
            "physics": ("millawell/wikipedia_field_of_science", None),
            "philosophy": ("sayhan/strix-philosophy-qa", None),
            "humanities": ("HuggingFaceTB/cosmopedia", "stanford"),  # ~6.3GB academic prose
        }
    )

    def __post_init__(self):
        # Fill any architecture / batch field left as None from the chosen preset.
        if self.preset not in MODEL_PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose {list(MODEL_PRESETS)}")
        for key, value in MODEL_PRESETS[self.preset].items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        # Key checkpoints by preset so different sizes never overwrite each other
        # (local dir and HF Hub subfolder both become e.g. .../DSP_LM/110m).
        self.output_dir = os.path.join(self.output_dir, self.preset)


# ==========================================================================
# 5. DATA  --  schema-aware extraction, robust loading, sequence packing
# ==========================================================================


CHAT_TEMPLATE = "User: {prompt}\nAssistant:"  # response follows after this


def make_formatters():
    """Structured per-dataset formatters -> (is_instruction, prompt, response).

    QA/instruction datasets become User/Assistant turns; during training the
    prompt tokens are masked out of the loss so the model is only supervised to
    produce the response (standard SFT). Prose corpora (textbooks) are plain
    continuation with full supervision. Returns None when the schema doesn't
    match (used to detect provenance after interleave loses it).
    """

    def logic(item):  # reasoning-core: prompt -> (chain-of-thought + answer)
        p = item.get("prompt", "")
        if not (isinstance(p, str) and p):
            return None
        c, a = item.get("cot", ""), item.get("answer", "")
        response = "\n".join(s for s in (c, a) if isinstance(s, str) and s)
        return (True, p, response) if response else None

    def qa_pair(item):  # orca-math / strix: question -> answer
        q, a = item.get("question", ""), item.get("answer", "")
        if isinstance(q, str) and q and isinstance(a, str) and a:
            return (True, q, a)
        return None

    def alpaca(item):  # alpaca-cleaned: instruction (+input) -> output
        instr, inp, out = (
            item.get("instruction", ""),
            item.get("input", ""),
            item.get("output", ""),
        )
        if not (isinstance(instr, str) and instr and isinstance(out, str) and out):
            return None
        prompt = f"{instr}\n{inp}" if isinstance(inp, str) and inp else instr
        return (True, prompt, out)

    def coedit(item):  # CoEdIT: src (instruction+text) -> tgt (corrected)
        s, t = item.get("src", ""), item.get("tgt", "")
        if isinstance(s, str) and s and isinstance(t, str) and t:
            return (True, s, t)
        return None

    def prose(item):  # open-web-math / science-wiki: plain continuation ('text')
        t = item.get("text", "")
        if isinstance(t, str) and t:
            return (False, "", t)
        # robustness: fall back to any other long string field
        for v in item.values():
            if isinstance(v, str) and len(v) > 200:
                return (False, "", v)
        return None

    # language/math/physics are large prose corpora; philosophy is Q&A.
    # qa_pair/alpaca are kept defined for a future SFT pass (orca-math, alpaca).
    return {
        "grammar": coedit,
        "language": prose,
        "logic": logic,
        "math": prose,
        "physics": prose,
        "philosophy": qa_pair,
        "humanities": prose,
        "qa": alpaca,
    }


def encode_example(name, item, formatters, tokenizer, mask_prompt):
    """Tokenise one (dataset-name, row) into (ids, loss_mask).

    loss_mask[i] == 1 -> token i is supervised; 0 -> ignored (prompt tokens).
    Provenance is known (the multiplexer tags each row with its dataset), so we
    apply exactly that dataset's formatter — no schema guessing.
    """
    r = formatters[name](item)
    if r is None:
        return [], []

    is_instruction, prompt, response = r
    eos = tokenizer.eos_token_id
    if is_instruction:
        pre = tokenizer.encode(CHAT_TEMPLATE.format(prompt=prompt))
        ans = tokenizer.encode(f" {response}") + [eos]
        ids = pre + ans
        mask = ([0] * len(pre) + [1] * len(ans)) if mask_prompt else [1] * len(ids)
    else:
        ids = tokenizer.encode(response) + [eos]
        mask = [1] * len(ids)
    return ids, mask


class WeightedMultiplex:
    """Weighted round-robin over several raw streaming iterators.

    Replaces datasets.interleave_datasets, which fails when sibling datasets
    have differently-typed columns of the same name (e.g. reasoning-core's
    ``prompt`` is Arrow large_string while Cosmopedia's is string). We sample
    in plain Python, so no cross-dataset schema alignment is attempted, and
    each yielded row keeps its dataset name for correct formatting.

    Exhausted sources are **cycled** (restarted from the top) so the training
    mix ratio stays stable for the entire substep. A warning is printed the
    first time each source restarts.
    """

    def __init__(self, iterables, weights, names, seed=3407):
        self._iterables = list(iterables)  # keep originals for recycling
        self.iters = [iter(it) for it in self._iterables]
        self.weights = list(weights)
        self.names = list(names)
        self._cycled: set[int] = set()  # indices that have been restarted
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iters:
            raise StopIteration
        i = self.rng.choices(range(len(self.iters)), weights=self.weights)[0]
        try:
            return self.names[i], next(self.iters[i])
        except StopIteration:
            # Cycle: restart from the beginning instead of dropping.
            if i not in self._cycled:
                print(
                    f"    [WeightedMultiplex] cycling exhausted source '{self.names[i]}'"
                )
                self._cycled.add(i)
            self.iters[i] = iter(self._iterables[i])
            return self.names[i], next(self.iters[i])


class PackedTokenStream:
    """Packs tokenised examples into (seq_len+1) blocks with an aligned loss mask.

    No padding waste; every block is full. Prompt tokens carry mask 0 (ignored
    by the loss), response / prose tokens carry mask 1. Examples are separated
    by EOS (added inside encode_example).
    """

    def __init__(self, multiplex, formatters, tokenizer, seq_len, mask_prompt=True):
        self.mux = iter(multiplex)
        self.formatters = formatters
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_prompt = mask_prompt
        self.ids_buf: list[int] = []
        self.mask_buf: list[int] = []

    def get_block(self, device):
        need = self.seq_len + 1
        while len(self.ids_buf) < need:
            try:
                name, item = next(self.mux)
            except StopIteration:
                if len(self.ids_buf) < need:
                    return None, None
                break
            ids, mask = encode_example(
                name, item, self.formatters, self.tokenizer, self.mask_prompt
            )
            if len(ids) > 5:
                self.ids_buf.extend(ids)
                self.mask_buf.extend(mask)
        ids = torch.tensor(self.ids_buf[:need], dtype=torch.long, device=device)
        mask = torch.tensor(self.mask_buf[:need], dtype=torch.long, device=device)
        self.ids_buf = self.ids_buf[need:]
        self.mask_buf = self.mask_buf[need:]
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].clone()
        y[mask[1:] == 0] = -100  # ignore prompt targets (cross_entropy default)
        return x, y.unsqueeze(0)


def get_packed_batch(streams, batch_size, device):
    """Assemble a batch by drawing packed blocks from the stream(s)."""
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


def save_checkpoint(path, model, optimizer, scheduler, step, completed_substeps, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "completed_substeps": completed_substeps,
            "config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("step", 0), ckpt.get("completed_substeps", 0)


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

    # Weight-decay only the 2-D projection/embedding matrices. Exclude biases,
    # LayerNorm gains (ndim < 2) and the SSM kernel poles (log_A_real, A_imag,
    # log_dt, C) + skip D — decaying resonator parameters is harmful.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or ".kernel." in name or name.endswith(".D"):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    return model, optimizer


def make_scheduler(cfg: Config, optimizer, total_steps: int):
    """WSD: linear warmup -> stable plateau at full LR -> final decay to floor."""
    decay_steps = int(total_steps * cfg.decay_frac)
    stable_end = max(cfg.warmup_steps, total_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if step < stable_end:
            return 1.0  # full LR for every curriculum phase
        progress = (step - stable_end) / max(1, total_steps - stable_end)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_eval_blocks(eval_loaded, formatters, tokenizer, seq_len, blocks_per):
    """Pack a fixed set of held-out blocks (CPU tensors) **per dataset**.

    Returns ``dict[str, list[tuple[Tensor, Tensor]]]`` so callers can evaluate
    on individual domains or a filtered subset.
    """
    blocks_by_name: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, ds in eval_loaded.items():
        mux = WeightedMultiplex([ds], [1.0], [name], seed=0)
        stream = PackedTokenStream(
            mux, formatters, tokenizer, seq_len, mask_prompt=False
        )
        ds_blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(blocks_per):
            x, y = stream.get_block("cpu")
            if x is None:
                break
            ds_blocks.append((x, y))
        if ds_blocks:
            blocks_by_name[name] = ds_blocks
    return blocks_by_name


@torch.no_grad()
def evaluate(model, blocks, vocab_size, device):
    """Mean cross-entropy over a flat list of held-out blocks."""
    if not blocks:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    for x, y in blocks:
        x, y = x.to(device), y.to(device)
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def evaluate_per_dataset(
    model, eval_blocks_by_name, vocab_size, device, active_datasets=None
):
    """Eval loss per dataset and an average over the active datasets only.

    ``active_datasets`` limits which datasets contribute to the reported
    average; if *None* all datasets are included.
    """
    per_ds: dict[str, float] = {}
    for name, blocks in eval_blocks_by_name.items():
        per_ds[name] = evaluate(model, blocks, vocab_size, device)

    if active_datasets is not None:
        active_vals = [per_ds[n] for n in active_datasets if n in per_ds]
    else:
        active_vals = list(per_ds.values())
    avg = sum(active_vals) / max(1, len(active_vals)) if active_vals else float("nan")
    return per_ds, avg


@torch.no_grad()
def ssm_diagnostics(model) -> dict:
    """Interpretability readout of the resonator poles (à la ResonatorLM's
    physics diagnostics): the timescales the SSM has learned to occupy.

    Per channel/state: effective per-step damping alpha = exp(log_A_real)*dt, so
    half-life = ln2 / alpha (in tokens); frequency = |A_imag| * dt (rad/step).
    A healthy model spreads half-lives across short and long timescales rather
    than collapsing to one regime.
    """
    half_lives, freqs = [], []
    for blk in model.blocks:
        k = blk.ssm.kernel
        dt = torch.exp(k.log_dt).unsqueeze(-1)            # (H, 1)
        alpha = (torch.exp(k.log_A_real) * dt).clamp_min(1e-8)  # (H, N/2)
        half_lives.append((math.log(2) / alpha).flatten().float())
        freqs.append((k.A_imag.abs() * dt).flatten().float())
    hl = torch.cat(half_lives)
    fr = torch.cat(freqs)
    qs = torch.quantile(hl, torch.tensor([0.05, 0.5, 0.95], device=hl.device))
    return {
        "channels": hl.numel(),
        "half_life_tokens": {
            "p5": round(qs[0].item(), 1),
            "median": round(qs[1].item(), 1),
            "p95": round(qs[2].item(), 1),
            "max": round(hl.max().item(), 1),
        },
        "freq_rad_per_step": {
            "min": round(fr.min().item(), 4),
            "max": round(fr.max().item(), 4),
        },
        "frac_longmemory_gt_1024tok": round((hl > 1024).float().mean().item(), 3),
    }


def print_ssm_diagnostics(model) -> None:
    d = ssm_diagnostics(model)
    hl = d["half_life_tokens"]
    print(
        f"SSM resonators: {d['channels']} modes | half-life tokens "
        f"p5={hl['p5']} median={hl['median']} p95={hl['p95']} max={hl['max']} | "
        f"{d['frac_longmemory_gt_1024tok']:.0%} long-memory (>1024 tok)"
    )


def smoke_test(cfg: Config, device: str) -> None:
    """Validate the whole train step on a LEARNABLE synthetic task.

    The task is next = (token + 1) mod vocab: a deterministic pattern the model
    must actually learn, so a healthy run shows loss falling well below the
    random-guess baseline ln(vocab). (If x and y are independent random noise,
    loss can never drop below ln(vocab) — that tests plumbing, not learning.)
    """
    import math as _math

    print("=== SMOKE TEST (learnable synthetic task, tiny model) ===")
    cfg.d_model, cfg.depth, cfg.seq_len = 64, 2, 64
    cfg.batch_size, cfg.grad_accum, cfg.n_states = 8, 1, 16
    cfg.num_branches, cfg.branch_dim = 4, 32
    cfg.warmup_steps, cfg.lr = 5, 3e-3
    vocab_size = 256
    n_steps = 60
    baseline = _math.log(vocab_size)

    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    scheduler = make_scheduler(cfg, optimizer, total_steps=n_steps)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    print(
        f"random-guess loss (ln vocab) = {baseline:.3f}; a healthy run drops below it"
    )

    first_loss = None
    last_loss = None
    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        y = (x + 1) % vocab_size
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), "NaN/Inf loss"
        first_loss = first_loss or loss.item()
        last_loss = loss.item()
        if step % 10 == 0 or step == n_steps - 1:
            print(
                f"  step {step:02d} | loss {loss.item():.4f} | ppl {_math.exp(min(20, loss.item())):.1f}"
            )

    gen = model.generate(x[:, :4], max_new_tokens=8, top_k=10)
    assert gen.shape == (cfg.batch_size, 12)
    print(f"\ninitial loss {first_loss:.3f} -> final loss {last_loss:.3f}")
    assert (
        first_loss < baseline * 2
    ), f"initial loss {first_loss:.1f} >> ln(vocab) {baseline:.1f} — logits mis-scaled (init bug)"
    assert last_loss < first_loss * 0.8, "loss did not fall — model is not learning"
    print(
        f"generate OK -> {tuple(gen.shape)}\n=== SMOKE TEST PASSED (model learns) ==="
    )


# ==========================================================================
# 8. CHECKPOINT SYNC  --  Hugging Face Hub (Colab <-> local machine)
# ==========================================================================


def _is_colab() -> bool:
    """Detect whether we're running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def hf_push_checkpoint(hf_repo: str, output_dir: str, subdir: str = "") -> None:
    """Push the checkpoint directory to a private Hugging Face Hub repo.

    ``subdir`` (the model-size preset) keeps each size in its own folder in the
    repo so sizes never overwrite each other. Requires ``huggingface-cli login``
    or an ``HF_TOKEN`` env var.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo,
        folder_path=output_dir,
        path_in_repo=subdir or ".",
        commit_message=f"checkpoint update ({subdir or 'root'})",
    )
    print(f"  Pushed checkpoint to https://huggingface.co/{hf_repo}/{subdir}")


def hf_pull_checkpoint(hf_repo: str, output_dir: str, subdir: str = "") -> bool:
    """Download this size's checkpoints from the Hub into output_dir.

    Returns True if the pull succeeded. Only the ``subdir`` (preset) folder is
    fetched, and it lands directly in output_dir.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        # output_dir ends in the preset; download repo/<preset>/* into its parent
        # so files land back in output_dir.
        parent = os.path.dirname(output_dir.rstrip("/")) or "."
        snapshot_download(
            repo_id=hf_repo,
            local_dir=parent if subdir else output_dir,
            allow_patterns=[f"{subdir}/*"] if subdir else None,
            local_dir_use_symlinks=False,
        )
        print(f"  Pulled checkpoint from https://huggingface.co/{hf_repo}/{subdir}")
        return True
    except RepositoryNotFoundError:
        print(f"  No remote checkpoint found at {hf_repo}")
        return False
    except Exception as exc:
        print(f"  Could not pull checkpoint: {type(exc).__name__}: {exc}")
        return False


def hf_clean(hf_repo: str, output_dir: str, subdir: str = "") -> None:
    """Delete this size's local checkpoints and its remote Hub folder.

    Only the current preset is removed; other sizes in the repo are left alone.
    """
    import shutil

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"  Deleted local checkpoints: {output_dir}")
    else:
        print("  No local checkpoints to delete.")

    if hf_repo and subdir:
        from huggingface_hub import HfApi

        try:
            HfApi().delete_folder(path_in_repo=subdir, repo_id=hf_repo)
            print(f"  Deleted remote folder: {hf_repo}/{subdir}")
        except Exception as exc:
            print(f"  Could not delete remote folder ({type(exc).__name__}: {exc})")


def main(
    resume_override: bool | None = None,
    continue_stage: bool = False,
    preset: str | None = None,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = Config(preset=preset) if preset else Config()
    print(f"Preset: {cfg.preset} ({cfg.d_model}d x {cfg.depth}L) -> {cfg.output_dir}")
    if resume_override is not None:
        cfg.resume = resume_override  # CLI 'resume' / 'overwrite' switch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if continue_stage:
        mode = "CONTINUE (load weights, fresh schedule, new data stage)"
    elif cfg.resume:
        mode = "RESUME from latest checkpoint"
    else:
        mode = "OVERWRITE (fresh start)"
    print(f"Mode: {mode}")

    if os.environ.get("DSP_SMOKE") == "1":
        smoke_test(cfg, device)
        return

    # On Colab with hf_repo configured, pull latest checkpoint for resume/continue.
    if _is_colab() and cfg.hf_repo and (cfg.resume or continue_stage):
        print("Checking HF Hub for existing checkpoint...")
        hf_pull_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)

    print(f"Checkpoint directory: {cfg.output_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # GPT-2's tokenizer carries a legacy 1024 cap (its positional-embedding
    # limit). This model has NO such limit — token embeddings + SSM handle any
    # length — so raise it to silence a spurious "sequence too long" warning.
    tokenizer.model_max_length = int(1e12)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size} tokens (GPT-2 BPE)")

    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print_ssm_diagnostics(model)  # timescale coverage at init

    # Robust dataset loading: drop anything that fails. Each dataset is split
    # into a held-out eval set (first eval_docs rows) and a train set (the rest)
    # so eval never overlaps training.
    formatters = make_formatters()
    loaded, eval_loaded = {}, {}
    for name, (repo, cfg_name) in cfg.repos.items():
        try:
            base = load_dataset(repo, cfg_name, split="train", streaming=True)
            eval_loaded[name] = base.take(cfg.eval_docs)
            loaded[name] = base.skip(cfg.eval_docs)
            tag = f"{repo}:{cfg_name}" if cfg_name else repo
            print(f"  loaded {name:<10} <- {tag}")
        except Exception as exc:  # gated / offline / renamed
            print(f"  SKIP  {name:<10} <- {repo}  ({type(exc).__name__}: {exc})")

    print("Building held-out eval blocks...")
    eval_blocks_by_name = build_eval_blocks(
        eval_loaded, formatters, tokenizer, cfg.seq_len, cfg.eval_blocks_per
    )
    total_blocks = sum(len(v) for v in eval_blocks_by_name.values())
    print(f"  {total_blocks} eval blocks held out across {list(eval_blocks_by_name)}")

    # Bloom-style progression: an explicit-language foundation (grammar drills +
    # simple correct stories + general prose) BEFORE the reasoning phases, then
    # the logic->math->physics->philosophy curriculum. 'language' (fineweb-edu)
    # stays on as a fluency backbone throughout. A phase may set "steps" to run
    # shorter than the global default (Phase 0 is a brief primer so the small
    # grammar sets don't over-repeat). Weights are auto-renormalised.
    phases = [
        {
            "name": "Phase_0_Language_Rules",
            "desc": "Explicit grammar correction + general prose",
            "datasets": ["grammar", "language"],
            "mixtures": [[0.40, 0.60]],
            "steps": 800,  # short primer; CoEdIT is small
        },
        {
            "name": "Phase_1_Foundation",
            "desc": "Language grammar + foundational logic",
            "datasets": ["language", "logic"],
            "mixtures": [[0.40, 0.60]],
        },
        {
            "name": "Phase_2_Math_Introduction",
            "desc": "Language, logic and math",
            "datasets": ["language", "logic", "math"],
            "mixtures": [[0.20, 0.25, 0.55], [0.20, 0.35, 0.45]],
        },
        {
            "name": "Phase_3_Physics_Application",
            "desc": "Language, logic, math and physics",
            "datasets": ["language", "logic", "math", "physics"],
            "mixtures": [
                [0.20, 0.10, 0.15, 0.55],
                [0.20, 0.10, 0.20, 0.50],
                [0.20, 0.15, 0.25, 0.40],
            ],
        },
        {
            "name": "Phase_4_Integration",
            "desc": "Language, reasoning, science, philosophy & humanities",
            "datasets": ["language", "logic", "math", "physics", "philosophy", "humanities"],
            "mixtures": [[0.20, 0.10, 0.10, 0.15, 0.20, 0.25]],
        },
    ]

    total_substeps = sum(len(p["mixtures"]) for p in phases)
    total_microsteps = sum(
        len(p["mixtures"]) * p.get("steps", cfg.steps_per_substep) for p in phases
    )
    total_steps = total_microsteps // cfg.grad_accum
    scheduler = make_scheduler(cfg, optimizer, total_steps)

    os.makedirs(cfg.output_dir, exist_ok=True)
    latest = os.path.join(cfg.output_dir, "latest.pt")
    global_step = 0
    completed_substeps = 0
    if continue_stage:
        # Continued pretraining: load only the WEIGHTS from the prior run and
        # train a new stage (new data mix) from scratch — fresh optimizer and
        # LR schedule, no substep skipping. This is how you add datasets to an
        # already-trained model without a full restart.
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model"])  # architecture must match
            print(f"CONTINUE: loaded weights from {latest} (step {ckpt.get('step', '?')}); "
                  "starting a fresh stage on the current data mix.")
        else:
            print(f"CONTINUE requested but no checkpoint at {latest} — training from scratch.")
    elif cfg.resume and os.path.exists(latest):
        global_step, completed_substeps = load_checkpoint(
            latest, model, optimizer, scheduler, device
        )
        print(
            f"Resumed from {latest}: step {global_step}, {completed_substeps} substeps done"
        )

    print("\nStarting curriculum training...")
    substep_global = 0  # flat index across all phases, for resume skip-ahead
    tokens_seen = 0

    def sample(seed_text="The physical principles governing the universe state that"):
        ids = tokenizer.encode(seed_text, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=40, temperature=0.8, top_k=50)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    for phase in phases:
        # Keep only datasets that actually loaded; renormalise the mixture.
        avail = [d for d in phase["datasets"] if d in loaded]
        if not avail:
            print(f"Skipping {phase['name']} — no datasets available.")
            continue
        header_shown = False

        for substep_idx, probs in enumerate(phase["mixtures"]):
            # Skip substeps already finished in a previous run.
            if substep_global < completed_substeps:
                substep_global += 1
                continue
            if not header_shown:
                print(f"\n{'=' * 60}\n{phase['name']}\n{phase['desc']}\n{'=' * 60}")
                header_shown = True

            kept = [
                (d, p)
                for d, p in zip(phase["datasets"], probs, strict=False)
                if d in loaded
            ]
            names = [d for d, _ in kept]
            weights = [p for _, p in kept]
            weights = [w / sum(weights) for w in weights]  # renormalise
            print(
                f"\n  Substep {substep_idx + 1}/{len(phase['mixtures'])} - "
                f"{dict(zip(names, [round(w, 3) for w in weights], strict=False))}"
            )

            # Weighted multiplex instead of interleave_datasets (which can't
            # align differently-typed columns across these datasets).
            # NOTE: streaming iterators restart from the top on resume (their
            # position isn't checkpointed); with shuffled multi-GB pools this is
            # acceptable for a research run.
            mux = WeightedMultiplex(
                [loaded[n] for n in names], weights, names, seed=3407 + substep_idx
            )
            stream = PackedTokenStream(
                mux, formatters, tokenizer, cfg.seq_len, cfg.mask_prompt_loss
            )
            streams = [stream]  # single packed stream; batch draws multiple blocks

            phase_steps = phase.get("steps", cfg.steps_per_substep)
            optimizer.zero_grad(set_to_none=True)
            ema = None
            t_log, tok_log = time.time(), tokens_seen  # throughput window
            for step in range(phase_steps):
                model.train()
                x, y = get_packed_batch(streams, cfg.batch_size, device)
                if x is None:
                    print("  Stream exhausted early.")
                    break

                # Guard: if prompt-masking left every target ignored, cross
                # entropy would be NaN — skip this (degenerate) batch.
                if (y != -100).sum() == 0:
                    continue

                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

                (loss / cfg.grad_accum).backward()
                tokens_seen += x.numel()
                lv = loss.item()
                ema = lv if ema is None else 0.95 * ema + 0.05 * lv

                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if cfg.save_every and global_step % cfg.save_every == 0:
                        save_checkpoint(
                            latest,
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            substep_global,
                            cfg,
                        )

                if step % cfg.log_every == 0 or step == phase_steps - 1:
                    dt = max(1e-6, time.time() - t_log)
                    tps = (tokens_seen - tok_log) / dt  # tokens/sec since last log
                    t_log, tok_log = time.time(), tokens_seen
                    print(
                        f"  [{phase['name']} | sub {substep_idx + 1}] step {step:04d} "
                        f"| loss {lv:.4f} (ema {ema:.4f}) | ppl {math.exp(min(20, ema)):8.1f} "
                        f"| lr {scheduler.get_last_lr()[0]:.2e} | {tokens_seen / 1e6:.1f}M tok "
                        f"| {tps / 1e3:.1f}K tok/s"
                    )

            # Flush trailing accumulated gradients only if the last
            # accumulation cycle was incomplete (avoids a spurious optimizer
            # step with stale/partial grads that corrupts the checkpoint).
            if step % cfg.grad_accum != cfg.grad_accum - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            substep_global += 1  # this substep is now complete

            out_dir = os.path.join(
                cfg.output_dir, phase["name"], f"substep_{substep_idx + 1}"
            )
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(out_dir, "checkpoint.pt"),
                model,
                optimizer,
                scheduler,
                global_step,
                substep_global,
                cfg,
            )
            save_checkpoint(
                latest, model, optimizer, scheduler, global_step, substep_global, cfg
            )
            tokenizer.save_pretrained(out_dir)

            # Per-dataset eval: report each domain + active-only average.
            active = [d for d in phase["datasets"] if d in loaded]
            per_ds, ev = evaluate_per_dataset(
                model, eval_blocks_by_name, vocab_size, device, active
            )
            print(f"  Saved checkpoint -> {out_dir}")
            parts = " | ".join(
                f"{n} {per_ds[n]:.2f}" for n in sorted(per_ds) if n in per_ds
            )
            print(f"  eval per-dataset: {parts}")
            print(
                f"  eval (active avg) loss {ev:.4f} | ppl {math.exp(min(20, ev)):.1f}"
            )
            print("  ", end="")
            print_ssm_diagnostics(model)  # how the resonators have evolved
            print(f"  sample: {sample()[:200]!r}")

            # Push to HF Hub so checkpoints survive Colab restarts.
            if cfg.hf_repo:
                try:
                    hf_push_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
                except Exception as exc:
                    print(f"  HF push failed (non-fatal): {exc}")

    print("\nTraining complete.")

    print("\n--- Generation test ---")
    seed_text = "The physical principles governing the universe state that"
    seed_idx = tokenizer.encode(seed_text, return_tensors="pt").to(device)
    generated = model.generate(seed_idx, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f"Seed: {seed_text!r}")
    print("Generated:\n" + tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    import sys

    # Colab/Jupyter injects arguments like `-f /root/.../kernel.json`. Parse only
    # the tokens we recognise: a command word and/or a size preset, in any order.
    # e.g.  `... 110m`  `... overwrite 110m`  `... 1b continue`  `... clean 500m`
    args = [a.lower() for a in sys.argv[1:]]
    preset = next((a for a in args if a in MODEL_PRESETS), None)
    commands = {"pull", "push", "clean", "overwrite", "fresh", "restart",
                "resume", "continue", "extend", "diagnose"}
    cmd = next((a for a in args if a in commands), "")

    if cmd == "diagnose":
        # Print the SSM resonator timescale readout for a size (loads its
        # checkpoint weights if present; otherwise shows the init spread).
        cfg = Config(preset=preset) if preset else Config()
        m = VectorizedDendriticLM(
            vocab_size=50257, d_model=cfg.d_model, depth=cfg.depth,
            n_states=cfg.n_states, num_branches=cfg.num_branches,
            branch_dim=cfg.branch_dim, use_checkpoint=False,
        )
        latest = os.path.join(cfg.output_dir, "latest.pt")
        if os.path.exists(latest):
            m.load_state_dict(torch.load(latest, map_location="cpu")["model"])
            print(f"[{cfg.preset}] diagnostics from {latest}:")
        else:
            print(f"[{cfg.preset}] no checkpoint; diagnostics at init:")
        print_ssm_diagnostics(m)
        sys.exit(0)

    if cmd in ["pull", "push", "clean"]:
        cfg = Config(preset=preset) if preset else Config()
        if cmd == "pull":
            if not cfg.hf_repo:
                print("Error: set hf_repo in Config first.")
                sys.exit(1)
            hf_pull_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
        elif cmd == "clean":
            # Wipe THIS size's local checkpoints + its remote Hub folder.
            hf_clean(cfg.hf_repo, cfg.output_dir, cfg.preset)
        elif cmd == "push":
            if not cfg.hf_repo:
                print("Error: set hf_repo in Config first.")
                sys.exit(1)
            hf_push_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
    elif cmd in ["overwrite", "fresh", "restart"]:
        main(resume_override=False, preset=preset)  # fresh run
    elif cmd == "resume":
        main(resume_override=True, preset=preset)  # force-resume
    elif cmd in ["continue", "extend"]:
        # Continued pretraining: keep the trained WEIGHTS, run a new stage on the
        # current data mix with a fresh schedule. Architecture must match.
        main(continue_stage=True, preset=preset)
    else:
        main(preset=preset)  # default: resume if a checkpoint exists
