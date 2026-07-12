# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DSP-LM is a from-scratch research language model that replaces self-attention
with two biologically-flavoured mechanisms: a state-space "resonator" for time
mixing and "dendritic" branches for per-token computation, trained with a
Bloom-style curriculum (grammar → logic → math → physics → philosophy/humanities).
It is a single-file, fully-vectorised PyTorch model — no custom kernels, no
distributed training framework.

The repo is intentionally small and monolithic:

- `colab_trainable_dendritic_lm.py` — **the source of truth.** Model, data
  pipeline, training loop, checkpointing, HF Hub sync, CLI entry point. ~1700
  lines, organised in numbered sections (see the `# ====` banner comments):
  1. Temporal mixer (`ResonatorSSMKernel`, `ResonatorSSM`)
  2. Channel mixer (`DendriticMLP`)
  3. Block + full model (`DendriticResonatorBlock`, `VectorizedDendriticLM`)
  4. `Config` dataclass + `MODEL_PRESETS`
  5. Data: formatters, `WeightedMultiplex`, `PackedTokenStream`, `BatchPrefetcher`
  6. Checkpointing (save/load, HF Hub push/pull/clean)
  7. Training loop (`main`), evaluation, smoke test, CLI dispatch
- `colab_trainable_dendritic_lm.ipynb` — **generated from the `.py`.** Never
  edit by hand; regenerate with `jupytext --to notebook colab_trainable_dendritic_lm.py`
  after changing the script (keep the `!pip install ...` line as its first cell).
- `chat.py` — standalone inference CLI; rebuilds the model from a checkpoint's
  saved config and runs FFT-path generation.
- `docs/recurrent_inference.md`, `docs/developmental_sparsity.md` — **design
  plans for unimplemented future work**, not descriptions of current code.
  Don't assume anything described there exists in `colab_trainable_dendritic_lm.py`
  unless you check.

`tests/` (pytest) covers pure-logic pieces — the synthetic planning domain,
config/preset resolution, data pipeline, checkpoint save/load, and (CUDA-only,
skipped otherwise) the Triton Vandermonde kernel against its PyTorch
reference. It's deliberately narrow: no test drives an actual multi-step
training run, that's still the smoke test's job (below) plus per-domain
held-out eval.

## Commands

This is a [uv](https://docs.astral.sh/uv/) project (`pyproject.toml` + `uv.lock`).

```bash
uv sync                                          # build .venv from pyproject/uv.lock
uv run python colab_trainable_dendritic_lm.py    # train (default: resume if checkpoint exists)
```

```bash
uv sync --extra dev && uv run pytest -q   # unit tests (tests/); CUDA ones skip without a GPU
uv sync --extra eightbit                  # bitsandbytes (~58MB), for Config.optim_8bit
```

`uv.lock` is committed and should stay that way (reproducible installs, like
any other lockfile) — it's not something to gitignore. `uv sync` alone is
declarative: it installs only the base dependency set and *prunes* any extras
(`dev`, `eightbit`) previously installed into the same `.venv`, so re-run with
`--extra ...` (or `--all-extras`) after a bare `uv sync` if you need them back.
`bitsandbytes` is optional rather than a base dependency because most work
here never touches `optim_8bit` (default `False`; only useful under real VRAM
pressure — a 40GB A100, or the `1b` preset).

Quick self-test — tiny model, learnable synthetic task, **no downloads**, an
integration-level check that a full train step actually learns (pytest above
covers unit-level pieces; this doesn't overlap it). Run after touching
model/training code:

```bash
DSP_SMOKE=1 uv run python colab_trainable_dendritic_lm.py
```

Model size and run mode are CLI words, order-independent, parsed in
`if __name__ == "__main__"` at the bottom of the script:

```bash
uv run python colab_trainable_dendritic_lm.py                    # default (500m, resume-if-exists)
uv run python colab_trainable_dendritic_lm.py 42m                # tiny preset, fastest smoke/iteration
uv run python colab_trainable_dendritic_lm.py overwrite 110m     # fresh run, ignore checkpoint
uv run python colab_trainable_dendritic_lm.py 1b continue        # continued pretraining (new stage, same weights)
uv run python colab_trainable_dendritic_lm.py diagnose 110m      # SSM pole/timescale readout, no training
uv run python colab_trainable_dendritic_lm.py clean 500m         # wipe local + remote checkpoints for one preset
uv run python colab_trainable_dendritic_lm.py push               # manual HF Hub sync
uv run python colab_trainable_dendritic_lm.py pull
```

Presets: `42m`, `110m`, `500m` (default), `1b` — set via `Config.preset` or the
CLI word above; each fixes `d_model`/`depth`/`branch_dim` and a matched
`batch_size`/`grad_accum`/`lr` (see `MODEL_PRESETS`, effective batch ~96,
tuned for an 80GB A100 — this is the intended training target; the default
preset is deliberately too large for local iteration on a small card, use
`42m`/`110m` explicitly for that). `main()` auto-scales `batch_size`/
`grad_accum` down on smaller GPUs (<70GB) to keep the effective batch constant.

Inference / chat, from a checkpoint (recurrent-capable, unbounded context):

```bash
uv run python chat.py                                    # completion REPL, latest 110m checkpoint
uv run python chat.py --chat                              # User/Assistant mode
uv run python chat.py --preset 42m                        # load a different size's checkpoint
uv run python chat.py --prompt "The proof begins" -n 40   # one-shot completion
```

Linting (trunk-managed, configs in `.trunk/configs/`): ruff (`select = ["B",
"D3", "E", "F"]`, `E501` ignored — formatters own line length), black, isort
(`profile=black`). Run via `trunk check` / `trunk fmt` if trunk is installed,
or `ruff check .` / `black .` / `isort .` directly.

## Architecture notes worth knowing before editing

- **Two-sublayer block, not attention.** Each `DendriticResonatorBlock` is
  pre-norm residual × 2: `ResonatorSSM` (time mixing) then `DendriticMLP`
  (per-token nonlinear compute, the FFN replacement). There is no attention
  anywhere and no positional embedding — position is carried entirely by the
  SSM recurrence.
- **`ResonatorSSM` runs in convolutional (FFT) form during training** — a
  diagonal, damped-complex-pole S4D-style kernel applied as a causal FFT
  convolution, `O(N log N)`, unbounded receptive field. Complex-pole math must
  stay in `float32` (unsupported in bf16); this is deliberate, don't
  autocast it away. A true `O(1)`-per-token recurrent inference mode is
  *planned* (`docs/recurrent_inference.md`) but not yet implemented — today's
  `chat.py` generation reprocesses the whole prefix every step via the FFT path.
- **`ResonatorSSMKernel.forward`'s pole→kernel construction is a custom
  Triton op** (`_VandermondeKernelFn`, guarded by `_HAS_TRITON`, CUDA only —
  falls back to the equivalent naive PyTorch einsum on CPU or if Triton is
  unavailable). It fuses what was a `(d_model, n_states/2, seq_len)` complex
  Vandermonde tensor + einsum reduction into one kernel that never
  materialises that tensor, both because it was ~35% of the whole model's
  forward time (recomputed again under `use_checkpoint=True`'s backward) and
  because that tensor being complex-valued is what caused `torch.compile`
  to blow up VRAM (Triton has no native complex dtype, so Inductor falls back
  to an eager kernel for it, breaking fusion). If you touch the pole
  parameterisation (`log_A_real`/`A_imag`/`log_dt`/`C`) in
  `ResonatorSSMKernel._discretise`, the custom kernel's forward+backward math
  must be re-derived to match — it is NOT auto-differentiated, it hand-codes
  the gradient w.r.t. the four `(H, N/2)` tensors it consumes
  (`dt_A.real`/`.imag`, `c_mod.real`/`.imag`); everything upstream of those
  (the discretisation itself) still uses ordinary autograd.
- **`DendriticMLP`** fans a token into `num_branches` independent GLU branches
  (`d_ff = num_branches * branch_dim`), each with a steep asymmetric "soma"
  gate that suppresses branches that haven't "resolved" before a
  down-projection integrates survivors. `docs/developmental_sparsity.md`
  describes a planned (unimplemented) overproduce-then-prune scheme built on
  top of this gate — don't assume any pruning/masking exists yet.
- **Weight tying + init/optim conventions**: output head is tied to the input
  embedding; init is GPT-2-style; AdamW excludes SSM poles and norms/biases
  from weight decay; LR follows a WSD schedule (warmup → stable plateau → short
  final decay via `decay_frac`) so every curriculum phase trains at full LR.
- **Config is one dataclass** (`Config` in the file, ~line 494) — all
  hyperparameters, dataset slate (`repos`), and checkpoint/Hub settings live
  there. `__post_init__` resolves `None` fields from the chosen `MODEL_PRESETS`
  entry and namespaces `output_dir` by preset so different sizes never clobber
  each other's checkpoints (`./checkpoints/DSP_LM/<preset>/`).
- **Data pipeline is streaming, not materialised.** `WeightedMultiplex` samples
  across dataset streams by weight in plain Python (not
  `datasets.interleave_datasets`, which breaks on mismatched Arrow column
  types across sibling datasets) and **cycles** exhausted small sources rather
  than starving that domain. `PackedTokenStream` packs tokenized examples into
  dense `seq_len+1` blocks with a loss mask (no padding waste; prompt tokens
  masked to `-100` when `mask_prompt_loss=True`). `BatchPrefetcher` runs this
  on a background thread so CPU tokenization overlaps GPU compute — the
  training loop was previously data-starved without it.
- **Curriculum phases are a list of dicts inside `main()`** (each with
  `name`/`desc`/`datasets`/`mixtures`, optionally `steps` to override the
  global `steps_per_substep`), not a separate config file — read `main()`
  directly to see/change the phase schedule.
- **Checkpoints** save model + optimizer + scheduler + step + full config, so
  `resume` restores everything and `chat.py` rebuilds the model architecture
  from the saved config (only needs `_MODEL_KEYS` from it). `Config.hf_repo`
  (private HF Hub repo) enables push/pull/clean so a run can continue across
  multiple ephemeral Colab sessions — this is the primary reason Hub sync
  exists, not general model distribution.
- **Base pretraining vs SFT**: `chat_format=False`/`mask_prompt_loss=False` is
  the default (plain prose, no `User:/Assistant:` scaffolding, so the model
  doesn't learn chat structure as a spurious attractor). A later
  instruction-tuning phase is expected to flip both to `True`.

## Environments

- **Local**: Pop!_OS with an RTX 3060 — `requirements.txt` assumes CUDA torch
  is already present; `uv sync` is the supported path.
- **Colab**: ephemeral runtime, CUDA torch preinstalled. The notebook's first
  cell installs `transformers`/`datasets`/`huggingface_hub`/`accelerate`. Use
  an A100 (High-RAM) runtime for the 500m/1b presets. `_is_colab()` gates
  auto-pull-on-resume behavior in `main()`.
