# DSP-LM: Deterministic Structural Plasticity Language Model

A small, from-scratch language model that replaces self-attention with two
biologically-flavoured ideas: a **state-space "resonator"** for mixing over time
and **dendritic branches** for per-token computation. It's trained with a
four-phase "curriculum of courses" that moves from formal logic through math and
physics to philosophy and general Q&A.

This is a research project — an experiment in whether this architecture can
learn language at all, not a production model. It currently trains cleanly and
generates fluent (if small-model) text.

## Architecture

Each block has two residual sublayers, cleanly separated:

**1. Resonator (diagonal SSM).** The temporal mixer is a diagonal,
damped-complex-pole state-space model (S4D-style). Each channel is a stable
damped oscillator — pole `-exp(a) + iω`, where the negative real part damps and
the imaginary part sets a resonant frequency — applied as a causal FFT
convolution. This gives an effectively unbounded receptive field at
`O(N log N)`, with none of the quadratic cost of self-attention. There is no
architectural context limit and no positional embedding; position is carried by
the recurrence itself.

**2. Dendritic MLP.** The per-token compute is a gated-GLU "dendrite": the token
fans out into independent branches, each branch solves local nonlinear logic in
its own sub-space, and a steep asymmetric soma gate suppresses branches that
haven't "structurally resolved" before a down-projection integrates the
survivors. It replaces the transformer FFN.

Separating time-mixing (the SSM) from channel-mixing (the dendrite) is what lets
the model scale to long context without attention and without the finite window
of a plain convolution.

## Curriculum & data

Training streams and token-packs several **non-gated** datasets, weighted per
phase. Because the pipeline streams, total pool size barely matters — you only
pull what `steps × batch × seq_len` demands, so there is no multi-GB download.

| Domain | Dataset | Size | Style |
|---|---|---|---|
| Logic | `reasoning-core/procedural-pretraining-pile` | ~7.3 GB | Procedural formal reasoning, correct-by-design (PDDL, FOL, CFG, causal). |
| Math | `microsoft/orca-math-word-problems-200k` | ~225 MB | 200K grade-school math word-problem Q&A. |
| Physics | `HuggingFaceTB/cosmopedia` · `openstax` | ~668 MB | OpenStax textbook prose, incl. University Physics. |
| Philosophy | `sayhan/strix-philosophy-qa` | ~391 MB | 134K philosophy Q&A. |
| General Q&A | `yahma/alpaca-cleaned` | ~40 MB | 51K general instruction Q&A. |

The four phases: (1) logic-dominant, (2) interpolate logic↔math, (3) introduce
physics, (4) balanced mix adding philosophy and general Q&A. Q&A datasets are
wrapped in a `User: … / Assistant: …` template so the model learns turn
structure; a held-out split of each dataset is evaluated at every checkpoint.

## Setup

This is a [uv](https://docs.astral.sh/uv/) project.

**Locally:**

```bash
uv sync                      # creates .venv from pyproject.toml / uv.lock
uv run python colab_trainable_dendritic_lm.py
```

**On Google Colab** (which uses pip, not uv), install the deps in the first
cell, then convert this script to a notebook:

```python
!pip install torch transformers datasets accelerate jupytext
!jupytext --to notebook colab_trainable_dendritic_lm.py
```

The defaults target an A100. On a smaller GPU, drop `batch_size` (and, if
needed, `d_model`/`depth`/`seq_len`) in the `Config` dataclass at the top of the
script.

## Usage

Quick sanity check — trains a tiny model on a learnable synthetic task and
asserts the loss falls (no downloads):

```bash
DSP_SMOKE=1 uv run python colab_trainable_dendritic_lm.py
```

Train (edit `Config` for a real run — e.g. `steps_per_substep = 3000`):

```bash
uv run python colab_trainable_dendritic_lm.py
```

Training resumes automatically from `checkpoints/DSP_LM/latest.pt`, skipping
finished curriculum substeps. Loss (EMA), held-out eval, and a generation sample
are printed at every checkpoint.

Chat / generate from a checkpoint:

```bash
uv run python chat.py                 # completion REPL
uv run python chat.py --chat          # User/Assistant mode
uv run python chat.py --prompt "The proof begins" -n 40
```

## Configuration

All knobs live in the `Config` dataclass: model size (`d_model`, `depth`,
`n_states`, `num_branches`, `branch_dim`), optimisation (`batch_size`,
`grad_accum`, `lr`, WSD schedule via `decay_frac`), curriculum
(`steps_per_substep`), evaluation (`eval_docs`), and `mask_prompt_loss` (off for
base training; turn on for a later instruction-tuning pass).

## Roadmap

- [x] SSM temporal mixer + dendritic channel mixer, trains and generates
- [x] Streaming curriculum with token packing and held-out eval
- [x] Checkpoint resume, WSD schedule, chat CLI
- [ ] Prove language learning at scale (~1B+ tokens)
- [ ] Instruction-tuning pass (`mask_prompt_loss = True`)
- [ ] Recurrent inference for O(1)-per-token generation — see
      [`docs/recurrent_inference.md`](docs/recurrent_inference.md)

## References

- Gu, Gupta, Goel, Ré — *On the Parameterization and Initialization of Diagonal
  State Space Models* (S4D), NeurIPS 2022. The damped-complex-pole resonator.
- Beniaguev, Segev, London — *Single cortical neurons as deep artificial neural
  networks*, Neuron 2021. Inspiration for treating dendritic branches as local
  nonlinear units.

The name nods to structural plasticity in biological neurons; the
implementation is a straightforward, fully-vectorised PyTorch model — no special
hardware required.
