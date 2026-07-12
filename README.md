# DSP-LM: Deterministic Structural Plasticity Language Model

A small, from-scratch language model that replaces self-attention with two
biologically-flavoured ideas — a **state-space "resonator"** for mixing over
time and **dendritic branches** for per-token computation — trained with a
Bloom-style "curriculum of courses" that moves from explicit language rules
through logic, math and physics to philosophy and the humanities.

This is a research project: an experiment in whether this architecture can learn
language, not a production model. It currently trains cleanly and generates
fluent (small-model) text.

## Architecture

Each block has two residual, pre-norm sublayers:

**1. Resonator (diagonal SSM).** The temporal mixer is a diagonal,
damped-complex-pole state-space model (S4D-style). Each channel is a stable
damped oscillator — pole `-exp(a) + iω` — applied as a causal FFT convolution
during training. This gives an effectively unbounded receptive field at
`O(N log N)`, no quadratic attention cost, and no positional embedding (position
is carried by the recurrence). At inference it runs in **recurrent form**
(`O(1)` per token, constant memory, unbounded context).

**2. Dendritic MLP.** The per-token compute is a gated-GLU "dendrite": the token
fans into independent branches, each solves local nonlinear logic in its own
sub-space, and a steep asymmetric soma gate suppresses branches that haven't
"resolved" before a down-projection integrates the survivors. Replaces the FFN.

The output head is weight-tied to the embedding; init is GPT-2 style; AdamW
excludes the SSM poles and norms/biases from weight decay; the LR follows a WSD
schedule (warmup → stable plateau → short final decay).

**Performance.** The resonator's pole→kernel construction (a per-channel
Vandermonde sum) is a custom fused **Triton kernel** rather than plain PyTorch
— profiling showed the naive version was ~35% of the whole model's forward
pass (recomputed again under gradient checkpointing), and being complex-valued
it was also what broke `torch.compile`'s memory footprint, since Triton has no
native complex dtype and Inductor fell back to an unfused eager kernel inside
the compiled graph. With that tensor walled off from `torch.compile` as an
opaque op, compilation is now safe and on by default (`Config.torch_compile`).
CUDA-only; falls back to the equivalent PyTorch implementation on CPU or if
Triton is unavailable (no separate install needed — Triton ships with CUDA
torch on Linux). Validated against the PyTorch reference (forward + gradients)
in `tests/test_triton_vandermonde.py`.

## Curriculum & data

Training **streams** and **token-packs** several non-gated datasets, weighted
per phase. Streaming reads only the slice you consume, so pool size is free; a
weighted multiplexer samples them (and cycles small sets rather than starving a
domain). Q&A datasets are wrapped in a `User:/Assistant:` template.

| Domain     | Dataset                                      | Scale        | Role                                       |
| ---------- | -------------------------------------------- | ------------ | ------------------------------------------ |
| grammar    | `grammarly/coedit`                           | 82k pairs    | explicit grammar correction                |
| language   | `HuggingFaceFW/fineweb-edu` (`sample-10BT`)  | 10B tokens   | general English: grammar, vocab, knowledge |
| logic      | `facebook/natural_reasoning`                 | ~6GB         | question + chain-of-thought, backtranslated from real corpora (not procedural templates, so it can't just be memorised) |
| planning   | *synthetic, procedural* — see below          | unbounded    | symbolic (Blocksworld) precondition/effect/state-tracking chain-of-thought |
| math       | `open-web-math/open-web-math`                | 14.7B tokens | mathematical text                          |
| physics    | `millawell/wikipedia_field_of_science`       | ~9.6GB       | broad science                              |
| philosophy | `sayhan/strix-philosophy-qa`                 | 134k QA      | philosophy Q&A                             |
| humanities | `HuggingFaceTB/cosmopedia` (`stanford`)      | ~6.3GB       | academic textbook prose                    |

**`planning` is generated, not downloaded.** It's a from-scratch Blocksworld
STRIPS engine (`BlocksworldPlanningStream`) that procedurally builds a random
problem, solves it, and renders a PDDL-Instruct-style ([arXiv:2509.13351](https://arxiv.org/abs/2509.13351))
chain-of-thought: explicit precondition checks, effect application, and the
running state after every step, ending in a goal-achieved `VALID` plan or
(~20% of examples) a deliberately-omitted setup action whose paired follow-up
then provably violates a precondition (`INVALID`, with the exact violated
predicate named). Because the same engine both generates and checks each
example, there's no external verifier and no LLM in the loop — it's exact by
construction, and it never runs out. See `_bw_render_example` in the script
and `tests/test_planning_domain.py`.

The phases (Bloom-style: explicit foundations before higher-order use):

0. **Language Rules** — grammar correction + prose (short 300-step primer)
1. **Foundation** — language + logic + planning
2. **Math Introduction** — + math
3. **Physics Application** — + physics
4. **Integration** — + philosophy + humanities

`language` stays on throughout as a fluency backbone; `planning` runs through
phases 1–3 alongside `logic` as another exactly-checkable formal-reasoning
source. A held-out slice of every dataset is evaluated **per domain** at each
checkpoint, so you can tell generalisation from memorisation.

## Setup

This is a [uv](https://docs.astral.sh/uv/) project.

**Locally:**

```bash
uv sync                                          # build .venv from pyproject/uv.lock
uv run python colab_trainable_dendritic_lm.py    # train (see commands below)
```

`uv sync` alone installs only the base dependencies — it's declarative, so it
will also *remove* any optional extras you'd previously installed into the
same `.venv`. Optional extras (each only needed for a specific `Config` flag,
off by default):

```bash
uv sync --extra dev                              # pytest, for tests/
uv sync --extra eightbit                         # bitsandbytes, for Config.optim_8bit
uv sync --extra dev --extra eightbit             # both — or: uv sync --all-extras
```

`bitsandbytes` (~58MB, CUDA-only) is kept optional because most runs never
need it — `optim_8bit` only matters when you're VRAM-constrained (a 40GB A100,
or the `1b` preset); the default `500m` preset has headroom on an 80GB A100
without it. Triton, by contrast, needs no separate install — it ships as part
of CUDA `torch` on Linux and powers the SSM kernel automatically.

**On Google Colab:** open `colab_trainable_dendritic_lm.ipynb`. Colab runtimes
are ephemeral and come with CUDA torch (and Triton) preinstalled, so the
notebook's **first cell installs the remaining dependencies**, then the rest
is the model code:

```python
!pip install -U transformers datasets huggingface_hub accelerate
# optional, only if you set Config.optim_8bit = True:
!pip install -U bitsandbytes
```

Use an A100 (High-RAM) runtime — the default preset (`500m`) and `Config`
defaults (`torch_compile = True`) are tuned for it, not for local iteration
hardware; use `42m`/`110m` locally instead (see [Usage](#usage)).

## Updating the notebook

The **`.py` is the source of truth.** The notebook is generated from it — don't
edit the notebook by hand. After changing the script, regenerate the notebook
locally with [jupytext](https://jupytext.readthedocs.io/):

```bash
jupytext --to notebook colab_trainable_dendritic_lm.py   # -> colab_trainable_dendritic_lm.ipynb
```

Then re-upload / re-open it in Colab. (Keep the `!pip install ...` line as the
notebook's first cell so a fresh Colab runtime has its dependencies.)

## Usage

Pick a model size via `Config.preset` (`42m`, `110m`, `500m` **default**,
`1b`). The default targets an 80GB A100 (Colab), not local iteration hardware
— pass `42m` or `110m` explicitly for that. Then:

| Command                                  | What it does                                                                                                                                             |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python colab_trainable_dendritic_lm.py` | Train; resumes automatically if a checkpoint exists                                                                                                      |
| `… resume`                               | Force-resume from the latest checkpoint (skips finished substeps)                                                                                        |
| `… overwrite` (or `fresh`)               | Fresh run, ignore existing checkpoint                                                                                                                    |
| `… continue` (or `extend`)               | **Continued pretraining**: keep the trained weights, train a new stage on the current data mix (add datasets without a restart; architecture must match) |
| `… clean`                                | Wipe local checkpoints **and** the remote HF repo                                                                                                        |
| `… push` / `… pull`                      | Manually sync checkpoints to/from the HF Hub                                                                                                             |

Quick self-test (tiny model, learnable synthetic task, no downloads):

```bash
DSP_SMOKE=1 uv run python colab_trainable_dendritic_lm.py
```

You can pick the size from the CLI, in any order with the command word:

```bash
uv run python colab_trainable_dendritic_lm.py 110m              # train 110m (resume if it exists)
uv run python colab_trainable_dendritic_lm.py overwrite 110m    # fresh 110m run
uv run python colab_trainable_dendritic_lm.py 1b continue       # continued-pretrain the 1b
uv run python colab_trainable_dendritic_lm.py clean 500m        # wipe just the 500m checkpoints
uv run python colab_trainable_dendritic_lm.py diagnose 110m   # anytime, on a checkpoint
```

It prints the resolved preset and output dir at startup so you can confirm before it commits. chat.py matches — --preset 110m loads that size's checkpoint (or pass --checkpoint explicitly).

Chat / generate from a checkpoint (recurrent, unbounded-context inference):

```bash
uv run python chat.py                                    # completion REPL
uv run python chat.py --chat                             # User/Assistant mode
uv run python chat.py --prompt "The proof begins" -n 40  # one-shot
```

Training prints, at each log step, EMA loss + perplexity + LR + tokens seen +
**tokens/sec**, and at each checkpoint a per-domain eval and a generation sample.

## Configuration

All knobs live in the `Config` dataclass at the top of the script. `preset` sets
`d_model`/`depth`/`branch_dim` and a matched `batch_size`/`grad_accum`/`lr`
(effective batch ~96 across presets, tuned for an 80GB A100 — lower `batch_size`
on smaller cards). Other useful fields: `seq_len`, `steps_per_substep` (a phase
may override with its own `steps`), `decay_frac` (WSD tail), `eval_docs`,
`mask_prompt_loss` (off for base training; on for a later instruction-tuning
pass), `hf_repo` (checkpoint auto-sync), and `use_checkpoint` (gradient
checkpointing — off trades VRAM for ~30% speed).

Efficiency-related flags:

- **`torch_compile`** (default `True`) — `torch.compile()`s the model. Safe
  and memory-neutral now that the SSM's complex-valued kernel construction is
  a Triton op opaque to `torch.compile`'s tracer (see Architecture above).
- **`optim_8bit`** (default `False`) — 8-bit AdamW via `bitsandbytes`, cutting
  optimizer-state VRAM ~4x. Only useful under real VRAM pressure (a 40GB A100,
  or the `1b` preset); needs `uv sync --extra eightbit`, and falls back to
  plain AdamW with a warning if the package isn't installed.

## Testing

```bash
uv sync --extra dev && uv run pytest -q
```

`tests/` covers pure-logic pieces — the synthetic planning domain, config/
preset resolution, the data pipeline, and checkpoint save/load (including the
`torch.compile` `_orig_mod.`-prefix fix). CPU-only and fast (~2s total). The
one CUDA-only file (`test_triton_vandermonde.py`, skipped automatically
without a GPU) regression-tests the fused kernel against its PyTorch
reference on tiny tensors — a correctness check, not a benchmark.

This doesn't replace the smoke test below, which is the only thing that
exercises an actual multi-step training loop end-to-end.

## Checkpoints & Hugging Face Hub

Checkpoints save to `./checkpoints/DSP_LM/` (per-substep dirs + a rolling
`latest.pt`, plus every `save_every` optimizer steps for crash safety). Each
stores model + optimizer + scheduler + step + config, so `resume` restores
everything and `chat.py` rebuilds the model from the saved config. Set
`Config.hf_repo` to auto-push/pull to a **private** Hub repo, which is how you
train the same model across multiple Colab sessions.

## Roadmap

- [x] SSM temporal mixer + dendritic channel mixer; trains and generates
- [x] Streaming Bloom curriculum, token packing, per-domain held-out eval
- [x] Presets, WSD schedule, resume / continue / overwrite, HF Hub sync
- [x] Recurrent O(1)-per-token inference — see [`docs/recurrent_inference.md`](docs/recurrent_inference.md)
- [x] Fused Triton kernel for the SSM's pole→kernel construction; `torch.compile`
      support; synthetic self-verifying symbolic-planning curriculum domain;
      `tests/` unit suite
- [ ] Prove language learning at scale (billions of tokens)
- [ ] Instruction-tuning pass (`mask_prompt_loss = True`, re-add orca/alpaca)
- [ ] Developmental structural plasticity (overproduce branches, prune on a
      schedule) — see [`docs/developmental_sparsity.md`](docs/developmental_sparsity.md)
- [ ] (optional) Physics diagnostics — per-channel half-lives, impulse response,
      spectral decomposition, suffix-perturbation causality test
- [ ] (frontier) Mamba-style input-dependent selectivity in the SSM

## References

- Gu, Gupta, Goel, Ré — _On the Parameterization and Initialization of Diagonal
  State Space Models_ (S4D), NeurIPS 2022. The damped-complex-pole resonator our
  temporal mixer is built on.
- Chaudhury — _ResonatorLM: Causal Resonant Field Mixing for Efficient
  Long-Context Language Modeling_, 2026 (arXiv:2607.05583). Independent
  validation of damped-resonator sequence mixing; our diagonal SSM is a
  multi-mode generalisation of its single-damped-cosine-per-head kernel.
- Beniaguev, Segev, London — _Single cortical neurons as deep artificial neural
  networks_, Neuron 2021. Dendritic branches as local nonlinear units.
- Verma, La, Favier, Mishra, Shah — _Teaching LLMs to Plan: Logical
  Chain-of-Thought Instruction Tuning for Symbolic Planning_, 2025
  (arXiv:2509.13351). The `planning` curriculum domain's chain-of-thought
  format (explicit precondition checks, effect application, running state,
  and named-precondition-violation explanations for invalid plans) follows
  this paper's Phase-1 data design; the paper's verifier-feedback training
  loop itself isn't used here (see the `planning` note above).

The name nods to structural plasticity in biological neurons; the implementation
is a straightforward, fully-vectorised PyTorch model — no special hardware
required.
