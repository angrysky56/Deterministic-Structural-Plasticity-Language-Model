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

    def forward(self, length: int) -> torch.Tensor:
        """Return the real causal kernel of shape ``(d_model, length)``."""
        # Kernel math runs in float32 (complex ops are unsupported in bf16).
        dt = torch.exp(self.log_dt)  # (H,)
        c = torch.view_as_complex(self.C)  # (H, N/2)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2) stable

        # Discretise: dtA = dt * A, folding (e^{dtA}-1)/A into C (B ~ ones).
        dt_a = a * dt.unsqueeze(-1)  # (H, N/2)
        c = c * (torch.exp(dt_a) - 1.0) / a  # (H, N/2)

        # Vandermonde: powers of the discrete pole along the sequence axis.
        arange = torch.arange(length, device=a.device)
        powers = torch.exp(dt_a.unsqueeze(-1) * arange)  # (H, N/2, L)
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
    n_states: int = 64  # SSM states per channel (N/2 conjugate pole pairs)
    num_branches: int = 8  # dendritic branches per token
    branch_dim: int = 256  # width of each branch (d_ff = num_branches*branch_dim)
    use_checkpoint: bool = True

    # Data / optimisation.
    seq_len: int = 2048  # SSM gives real long context (was 256)
    batch_size: int = 8
    grad_accum: int = 8  # effective batch 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    min_lr_ratio: float = 0.1  # cosine floor as a fraction of lr

    # Curriculum.
    steps_per_substep: int = 500
    log_every: int = 100
    # Chat template is always applied (teaches User/Assistant turn structure).
    # Prompt-masking is OFF by default: for a from-scratch base model you want
    # every token to teach, and long formal prompts otherwise leave whole packed
    # blocks fully masked (wasted compute, risk of all-ignored batches). Turn it
    # ON for a later dedicated SFT/instruction-polish phase.
    mask_prompt_loss: bool = False

    output_dir: str = "./checkpoints/DSP_LM"
    resume: bool = True  # resume from latest checkpoint (skips finished substeps)

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

    def philosophy(item):  # strix: question -> answer
        q, a = item.get("question", ""), item.get("answer", "")
        if isinstance(q, str) and q and isinstance(a, str) and a:
            return (True, q, a)
        return None

    def prose(item):  # cosmopedia textbooks: plain continuation, no prompt
        t = item.get("text", "")
        return (False, "", t) if isinstance(t, str) and t else None

    return {"logic": logic, "math": prose, "physics": prose, "philosophy": philosophy}


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
    `prompt` is Arrow large_string while Cosmopedia's is string). We sample in
    plain Python, so no cross-dataset schema alignment is attempted, and each
    yielded row keeps its dataset name for correct formatting.

    Exhausted sources are dropped and weights renormalised (~ "all_exhausted").
    """

    def __init__(self, iterables, weights, names, seed=3407):
        self.iters = [iter(it) for it in iterables]
        self.weights = list(weights)
        self.names = list(names)
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        while self.iters:
            i = self.rng.choices(range(len(self.iters)), weights=self.weights)[0]
            try:
                return self.names[i], next(self.iters[i])
            except StopIteration:
                del self.iters[i], self.weights[i], self.names[i]
        raise StopIteration


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


def main() -> None:
    from datasets import load_dataset
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
    # GPT-2's tokenizer carries a legacy 1024 cap (its positional-embedding
    # limit). This model has NO such limit — token embeddings + SSM handle any
    # length — so raise it to silence a spurious "sequence too long" warning.
    tokenizer.model_max_length = int(1e12)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size} tokens (GPT-2 BPE)")

    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Robust dataset loading: drop anything that fails (e.g. gated physics).
    formatters = make_formatters()
    loaded = {}
    for name, (repo, cfg_name) in cfg.repos.items():
        try:
            loaded[name] = load_dataset(repo, cfg_name, split="train", streaming=True)
            tag = f"{repo}:{cfg_name}" if cfg_name else repo
            print(f"  loaded {name:<10} <- {tag}")
        except Exception as exc:  # gated / offline / renamed
            print(f"  SKIP  {name:<10} <- {repo}  ({type(exc).__name__}: {exc})")

    phases = [
        {
            "name": "Phase_1_Logic_Dominant",
            "desc": "Foundational Logic (90/10)",
            "datasets": ["logic", "math"],
            "mixtures": [[0.90, 0.10]],
        },
        {
            "name": "Phase_2_Math_Introduction",
            "desc": "Interpolating Logic and Math",
            "datasets": ["logic", "math"],
            "mixtures": [[0.30, 0.70], [0.50, 0.50]],
        },
        {
            "name": "Phase_3_Physics_Application",
            "desc": "Physics & Derivations",
            "datasets": ["logic", "math", "physics"],
            "mixtures": [[0.10, 0.20, 0.70], [0.15, 0.25, 0.60], [0.20, 0.30, 0.50]],
        },
        {
            "name": "Phase_4_Philosophy_Meta",
            "desc": "Philosophy & Conceptual Analysis",
            "datasets": ["logic", "math", "physics", "philosophy"],
            "mixtures": [[0.25, 0.25, 0.25, 0.25]],
        },
    ]

    total_substeps = sum(len(p["mixtures"]) for p in phases)
    total_steps = total_substeps * cfg.steps_per_substep // cfg.grad_accum
    scheduler = make_scheduler(cfg, optimizer, total_steps)

    os.makedirs(cfg.output_dir, exist_ok=True)
    latest = os.path.join(cfg.output_dir, "latest.pt")
    global_step = 0
    completed_substeps = 0
    if cfg.resume and os.path.exists(latest):
        global_step, completed_substeps = load_checkpoint(
            latest, model, optimizer, scheduler, device
        )
        print(
            f"Resumed from {latest}: step {global_step}, {completed_substeps} substeps done"
        )

    print("\nStarting curriculum training...")
    substep_global = 0  # flat index across all phases, for resume skip-ahead
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

            optimizer.zero_grad(set_to_none=True)
            for step in range(cfg.steps_per_substep):
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

                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if step % cfg.log_every == 0 or step == cfg.steps_per_substep - 1:
                    ppl = math.exp(min(20, loss.item()))
                    print(
                        f"  [{phase['name']} | sub {substep_idx + 1}] step {step:04d} "
                        f"| loss {loss.item():.4f} | ppl {ppl:8.1f} "
                        f"| lr {scheduler.get_last_lr()[0]:.2e}"
                    )

            # Flush trailing accumulated gradients, then checkpoint.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
