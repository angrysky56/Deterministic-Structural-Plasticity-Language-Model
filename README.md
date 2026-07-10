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

## Curriculum & data

Training **streams** and **token-packs** several non-gated datasets, weighted
per phase. Streaming reads only the slice you consume, so pool size is free; a
weighted multiplexer samples them (and cycles small sets rather than starving a
domain). Q&A datasets are wrapped in a `User:/Assistant:` template.

| Domain     | Dataset                                      | Scale        | Role                                       |
| ---------- | -------------------------------------------- | ------------ | ------------------------------------------ |
| grammar    | `grammarly/coedit`                           | 82k pairs    | explicit grammar correction                |
| language   | `HuggingFaceFW/fineweb-edu` (`sample-10BT`)  | 10B tokens   | general English: grammar, vocab, knowledge |
| logic      | `reasoning-core/procedural-pretraining-pile` | 3.1M rows    | formal reasoning, correct-by-design        |
| math       | `open-web-math/open-web-math`                | 14.7B tokens | mathematical text                          |
| physics    | `millawell/wikipedia_field_of_science`       | ~9.6GB       | broad science                              |
| philosophy | `sayhan/strix-philosophy-qa`                 | 134k QA      | philosophy Q&A                             |
| humanities | `HuggingFaceTB/cosmopedia` (`stanford`)      | ~6.3GB       | academic textbook prose                    |

The phases (Bloom-style: explicit foundations before higher-order use):

0. **Language Rules** — grammar correction + prose (short 800-step primer)
1. **Foundation** — language + logic
2. **Math Introduction** — + math
3. **Physics Application** — + physics
4. **Integration** — + philosophy + humanities

`language` stays on throughout as a fluency backbone. A held-out slice of every
dataset is evaluated **per domain** at each checkpoint, so you can tell
generalisation from memorisation.

## Setup

This is a [uv](https://docs.astral.sh/uv/) project.

**Locally:**

```bash
uv sync                                          # build .venv from pyproject/uv.lock
uv run python colab_trainable_dendritic_lm.py    # train (see commands below)
```

**On Google Colab:** open `Colab_Trainable_Dendritic_LM.ipynb`. Colab runtimes
are ephemeral and come with CUDA torch preinstalled, so the notebook's **first
cell installs the remaining dependencies**, then the rest is the model code:

```python
!pip install -U transformers datasets huggingface_hub accelerate
```

Use an A100 (High-RAM) runtime for the 500M/1B presets.

## Updating the notebook

The **`.py` is the source of truth.** The notebook is generated from it — don't
edit the notebook by hand. After changing the script, regenerate the notebook
locally with [jupytext](https://jupytext.readthedocs.io/):

```bash
jupytext --to notebook colab_trainable_dendritic_lm.py   # -> Colab_Trainable_Dendritic_LM.ipynb
```

Then re-upload / re-open it in Colab. (Keep the `!pip install ...` line as the
notebook's first cell so a fresh Colab runtime has its dependencies.)

## Usage

Pick a model size via `Config.preset` (`42m`, `110m` default, `500m`, `1b`), then:

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

The name nods to structural plasticity in biological neurons; the implementation
is a straightforward, fully-vectorised PyTorch model — no special hardware
required.
