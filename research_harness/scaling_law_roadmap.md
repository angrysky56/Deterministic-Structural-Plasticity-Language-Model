# Scaling-law roadmap

Follow-on from the muP work in `train_harness.py`/`autoresearch_notebook.py`
(commit `aaa88ba`) and Ty's literature review. That pass fixed one axis --
hidden-layer LR/init not scaling correctly with width. This doc plans the
remaining axes he flagged: optimizer choice (Muon), weight decay vs batch
size, critical batch size, and LR annealing shape as its own dimension.
Ordered by how cheap-to-test-and-well-grounded each one is, not by how
interesting it sounds.

## 1. Muon optimizer -- implemented, ablation running

**Status:** `research_harness/muon.py` vendors Keller Jordan's
`SingleDeviceMuonWithAuxAdam` (single-GPU, no `torch.distributed`).
`DSPLMHarness.setup_optimizer_muon()` in `train_harness.py` routes the same
3-group split already built for muP -- embedding, hidden matrices, no_decay
-- through Muon-for-hidden + AdamW-for-the-rest, matching upstream's own
recipe ("embeddings, classifier heads, and hidden gains/biases should be
optimized using standard AdamW"). Toggle with `OPTIMIZER=muon MUON_LR=<val>
uv run research_harness/train_harness.py` (default `OPTIMIZER=adamw`).

**Why this is the highest-value item of the four:** "Same Architecture,
Different Capacity: Optimizer-Induced Spectral Scaling Laws"
([2605.21803](https://arxiv.org/pdf/2605.21803)) found that Muon reaches a
meaningfully higher effective-rank exponent than AdamW at *every* learning
rate tried, in both directions -- AdamW's best exponent (0.44) is below
Muon's *worst* (0.80). That's a documented, LR-tuning-proof gap, not a
close call worth skipping. Muon (or the specific technique of orthogonalized
updates) is also now behind several current frontier open-weight models
(Kimi-K2, GLM-5, DeepSeek-V4, per the same search).

**DSP-LM-specific wrinkle, already handled:** the SSM's pole parameters
(`log_A_real`, `A_imag` in `ResonatorSSMKernel`) are themselves 2D tensors
`(d_model, n_states/2)`, so an ndim-only Muon/AdamW split would incorrectly
feed them to Newton-Schulz orthogonalization -- meaningless for a
per-channel pole array, which isn't a "layer" in the sense Muon targets.
The existing param split already excludes them by *name* (`.kernel.`), not
ndim, so this was already safe going into the Muon work -- worth stating
explicitly since it's the kind of thing that's easy to get wrong silently.

**Open question, genuinely unsettled in the literature:** whether Muon's LR
needs muP-style width rescaling at all. Upstream's own README claims
"built-in muP scaling... shouldn't need to retune it," but a 2026 survey
found two competing theoretical prescriptions that disagree on the exponent
(steepest-descent-under-operator-norm gives width-independent scaling;
"large-scale industrial" heuristic gives `Θ(1/√width)`) -- see
[On the Width Scaling of Neural Optimizers Under Matrix Operator
Norms](https://arxiv.org/pdf/2603.09952). `setup_optimizer_muon()` currently
passes `muon_lr` through unscaled (no `d_model_base/d_model` factor),
following upstream's claim. If the eventual A100 transfer doesn't hold, this
is the first thing to revisit -- test by running the same `MUON_LR` at two
different local widths (e.g. d_model=128 and 256) and checking whether the
optimum shifts.

**Result: confirms the literature's claim, ablation complete.** Full local
grid (`muon_sweep_results.tsv`), same reference config as the muP sweep,
`embed_lr`/no_decay lr fixed at the grounded AdamW value (0.008):

| muon_lr | val_bpb |
|---|---|
| 0.005 | 1.4846 |
| 0.01  | 1.4855 |
| **0.02**  | **1.4755** (best) |
| 0.05  | 1.5141 |
| 0.1   | 1.5362 |

Clean peak exactly at Muon's upstream default (0.02), reached with zero
tuning -- beats AdamW's best grid point (1.5220 at the swept/grounded base
LR 0.008, or 1.5129 at the extended 0.016 point) by ~0.037-0.047 bpb, a
real margin at this scale, not noise. **Recommendation: switch the A100
notebook's optimizer to Muon-for-hidden-matrices before the next push.**
`DSPLM.setup_optimizer_muon()` needs to be added to
`autoresearch_notebook.py` (currently only in `train_harness.py`) --
straightforward port, same 3-group split already exists there for muP.

**Not implemented, flagged as a separate/later effort:** MuonSSM
([2606.30461](https://arxiv.org/pdf/2606.30461), ICML 2026 Oral) is a more
specialized technique -- it conditions the geometry of the *recurrent
memory update itself* (a Newton-Schulz transform on low-rank input
injections, plus a momentum pathway), not just "run vanilla Muon on
whatever 2D matrices exist in the model." It targets exactly the kind of
long-horizon memory degradation DSP-LM's resonator could plausibly hit, but
it's validated on different SSM variants and is a real implementation
project, not a param-group toggle. Worth revisiting *if* vanilla Muon
already helps and long-context stability becomes the next bottleneck --
premature otherwise.

## 2. Weight decay vs batch size -- currently inconsistent, cheap to fix

**The finding:** "Power Lines: Scaling Laws for Weight Decay and Batch Size
in LLM Pre-training" ([2505.13738](https://arxiv.org/pdf/2505.13738)) shows
the optimal AdamW weight decay `lambda` scales *linearly* with batch size
`B` at fixed model/data size (the "timescale" `tau = B/(eta*lambda*D)`
should stay constant).

**The problem this exposes in our own setup:** `WEIGHT_DECAY = 0.1` is used
as a flat constant at every scale tried so far -- the local 3060 harness
(`TOTAL_BATCH_SIZE = 2**14` = 16,384) *and* the A100 notebook's push3/4
(`TOTAL_BATCH_SIZE = 393,216`, ~24x larger). Under `lambda ~ B`, a weight
decay grounded at the local batch size implies the A100 runs should be
using roughly 24x that value, not the same number -- we've been holding
this one fixed by accident, not by a considered choice, the same failure
mode the muP work just fixed for LR.

**Concrete next step:** treat `WEIGHT_DECAY` as a third swept axis at the
local reference config (same d_model=256/depth=6 setup), *at the local
batch size*, then apply the linear-in-B transfer rule when moving to the
A100 config, mirroring exactly how `LR` is already transferred via
`d_model_base/d_model`. Cheap: reuses the exact same sweep harness, just a
different env var (`WD_SWEEP=<value>` following the `SWEEP_LR` pattern).
Should happen before the next A100 push, since it directly stacks with the
architecture-size question (bigger model + same huge batch = further from
whatever `lambda` was implicitly tuned for).

## 3. Critical batch size -- useful sanity check, not urgent

**What it's for:** confirms whether `TOTAL_BATCH_SIZE` on either the local
harness or the A100 notebook is in a sane regime, rather than being picked
from "whatever fits in VRAM" (which is how push2's batch size was chosen,
per the existing commit history). McCandlish et al.
([1812.06162](https://arxiv.org/pdf/1812.06162)) show the *gradient noise
scale* `B_noise = tr(Sigma) / ||G||^2` (trace of the per-example gradient
covariance over squared full-batch gradient norm) predicts the batch size
past which more parallelism stops helping -- training scales ~linearly with
batch size below `B_noise` and hits diminishing returns above it.

**Practical estimation, cheap:** `B_noise` can be estimated from quantities
already computed during training -- compare the squared norm of a
small-batch gradient average against a large-batch gradient average across
a few steps at 2-3 different batch sizes; no separate infra needed, just a
short instrumented run. A more recent, simpler empirical approach exists in
"Critical Batch Size Revisited" ([2505.23971](https://arxiv.org/pdf/2505.23971),
NeurIPS 2025) if the noise-scale estimator turns out fiddly in practice --
worth reading first since it's specifically framed as a simpler alternative
for large-batch LM pretraining.

**Priority:** lower than items 1-2. This doesn't change what to *do*
differently, just gives a sanity bound on batch size choices already made.
Reasonable to defer until after the Muon ablation and the WD sweep land,
then run once as a check against whatever batch size the next A100 push
settles on.

## 4. LR annealing shape -- worth one more sweep axis, not zero-priority

**The finding:** current best practice for LR schedules is "constant (or
warmup-stable) for most of training, short cooldown at the end" -- e.g. the
functional-scaling-law work on warmup-stable-decay
([2602.06797](https://arxiv.org/html/2602.06797)) and the original
LR-annealing scaling law ([2408.11029](https://arxiv.org/abs/2408.11029)),
which found annealing gives a consistent ~2.45% improvement and, notably,
*does not change the optimal values of other hyperparameters* -- meaning
it's safe to tune independently rather than needing to be re-swept jointly
with LR/WD.

**The problem this exposes:** `WARMDOWN_RATIO = 0.3` (30% of the whole run
spent ramping down) has been fixed since the very first local experiment
(`exp4`, shortened from an even longer 0.5) and never actually re-swept --
it was chosen by a single before/after comparison, not a grid. Recent
literature leans toward a *much* shorter cooldown fraction being sufficient.
30% may be leaving real gains on the table, or may be fine -- genuinely
don't know without testing.

**Concrete next step:** small local sweep of `WARMDOWN_RATIO` alone
(e.g. 0.05, 0.1, 0.2, 0.3) at the grounded LR (8e-3, AdamW) and reference
architecture, holding everything else fixed. Cheapest of the four items to
test (no optimizer/architecture changes needed, just the existing
`get_lr_multiplier` schedule), but sequenced last here because the expected
payoff (~2-3% per the literature) is smaller than muP (already captured,
6.7x LR correction) or Muon (documented large rank-quality gap).

## Suggested order

1. Finish the running Muon ablation (already in flight, ~25 min).
2. Weight-decay-vs-batch-size sweep (cheap, exposes a real existing gap).
3. LR annealing shape sweep (cheap, smaller expected payoff).
4. Critical batch size estimate (sanity check, not a lever by itself).
5. Only then: next A100 push, now informed by optimizer choice + WD
   transfer rule + (possibly) a shorter cooldown -- stacking all of this
   into one push rather than re-running the A100 notebook after every
   local finding.
