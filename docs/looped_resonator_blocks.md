# Looped Resonator Blocks — Design Plan

Status: **planned, not implemented.** This is a design note inspired by an
external paper, not a spec of existing code. Captured here so the idea
survives past the conversation that surfaced it (Ty, 2026-07-14).

Source: Yu, Kojima, Matsuo, Iwasawa, *"Looped State-Space Language Models
with Adaptive Exit-State Selection"* (arXiv:2607.10110, July 2026). First
controlled study of the "looped model" trick (Universal Transformers →
Looped Transformers → this) applied to Mamba-2 and Hybrid Mamba-Transformer
backbones instead of only attention-only Transformers.

## Why

Every DSP-LM preset currently scales depth by adding more *distinct*
parameterized `DendriticResonatorBlock` instances (6 → 12 → 18 across
42m/110m/500m). The paper's core finding: instead of adding more distinct
layers, you can repeatedly apply a *shared* block — weight-tied across the
repeats — and get comparable or better effective depth for a fraction of the
parameters. In their notation, `N ⊗ R` means `N` independently-parameterized
layers applied `R` times recurrently, giving effective depth `N·R` while the
parameter count scales only with `N`. On their synthetic compositional/
recursive tasks (modular-arithmetic manipulation, multi-hop induction), a
looped `4⊗6` model (24 effective layers, 4 layers' worth of parameters)
reached 99.35% accuracy on the hardest setting vs. 37.45% for a
non-looped 24-layer baseline of the *same parameter budget*. That's not a
subtle effect — it's the kind of result worth at least asking "does this
apply to us" before the next preset just adds more layers again.

## What looping would actually mean here

Nothing about the block itself changes. `ResonatorSSM` and `DendriticMLP`
stay exactly as they are — looping is purely a change to how
`VectorizedDendriticLM.forward` calls the existing `ModuleList` of blocks:
instead of `depth` distinct blocks called once each, call `N` distinct
blocks `R` times each (sequentially, same blocks reused). Effective depth is
still `N·R`; only which blocks are distinct parameters changes.

## Key results worth carrying over

- **Iso-parameter** (same param count, more effective depth via looping):
  looped models consistently beat their non-looped, same-size counterpart —
  the effect is large on compositional/recursive synthetic tasks, smaller
  but real on general language-model validation perplexity and downstream
  benchmarks.
- **Iso-FLOPs** (same effective depth, looped vs. genuinely-deep): looped
  models are competitive but don't fully close the gap — a real 24-layer
  model still beats a 4-layer block looped 6× on strict validation
  perplexity, even though the looped version can match or beat it on
  downstream benchmark averages. Perplexity and downstream performance
  don't move together here; worth measuring both if this gets tried.
- **Adaptive exit-gate** (adapted from a separate paper, "Ouro"): a
  lightweight per-token linear gate learns, per loop iteration, whether
  continuing is worth it — trained in two stages (first with entropy
  regularization to avoid collapsing to a single fixed depth, then a frozen-
  backbone calibration stage that matches the gate's continuation
  probability to the *realized* loss improvement from taking that step).
  This lets different tokens effectively use different depths.

## The catch that matters most for us: state continuity

This is the part worth internalizing before getting excited about the
headline numbers. Unlike a Transformer, where a token's exit decision at one
layer doesn't mechanically depend on every other position, an SSM's
recurrent scan at loop step `r+1` consumes the step-`r+1` state built from
*every preceding position*. A token can't just be pulled out of the loop
early without corrupting the state every later position depends on. The
paper's own exit-gate therefore does **not** save any compute — all `R` loop
passes execute for every token regardless; the gate only selects *which*
pass's hidden state gets handed to the LM head for the loss/prediction. Real
inference-time savings would need additional state-handling machinery,
which the paper explicitly leaves as future work.

This interacts directly with [`recurrent_inference.md`](recurrent_inference.md)'s
planned `O(1)`-per-token decode mode: a looped resonator block run
recurrently at decode time would need to carry `R` separate SSM states (one
per loop iteration) rather than the single state `recurrent_inference.md`
currently plans for, and the training-time FFT/convolutional path would need
`R` kernel constructions per forward pass instead of one. If looping is ever
adopted, design the loop state and the recurrent-decode state together —
retrofitting one onto the other after the fact would be more painful than
planning for both up front.

## Relationship to the dendritic branch-gate / developmental_sparsity.md

Conceptual cousin, different axis. `DendriticMLP`'s branch-gate already does
per-token adaptive compute *within* a layer, across branches (the
overproduce-then-prune plan in
[`developmental_sparsity.md`](developmental_sparsity.md) builds directly on
that same gate). The paper's exit-gate does the analogous thing *across*
recurrent depth, across loop iterations, instead. Both are ultimately
"does another unit of compute help this token, and let the model decide
per-token" mechanisms. If looping is ever combined with exit-gating here,
the paper's two-stage training recipe (broad entropy-regularized exploration
→ frozen-backbone focused calibration against realized loss improvement) is
directly reusable regardless of which axis — branches or loop depth — it
ends up gating.

## Interaction with the muP/scaling-law work (research_harness/)

muP's transfer guarantee (see `research_harness/scaling_law_roadmap.md`) is
specifically about *width* at *fixed depth*. Looping introduces a genuinely
new axis — loop count `R` — that decouples effective depth from parameter
count in a way the standard muP width formula doesn't cover, and the paper
doesn't touch muP at all. If looping gets tried, don't assume the grounded
LR from the existing sweep (`mup_sweep_results.tsv`) just transfers: it would
need its own small sweep over `R` at fixed `N`, following the same
never-guess-ground-it-first discipline as everything else in
`scaling_law_roadmap.md` (and the `ml-scaling-laws` skill, if using it).

## Sketch of a first experiment

1. Add a `loop_count` (R) field to `DSPLMConfig` in `train_harness.py`
   alongside `depth` (N); change `forward` to call the first `N` blocks `R`
   times sequentially instead of `depth`-many distinct blocks. Small change
   — it reuses the existing blocks and eval pipeline as-is.
2. **Iso-parameter run:** fix `N` at the existing muP reference config
   (d_model=256, depth=6), vary `R ∈ {1, 2, 4}`, compare val_bpb against a
   `depth = N·R` non-looped baseline at the *same parameter count* — directly
   mirrors the paper's own Mano/p-hop protocol, reusing the local harness's
   existing val_bpb metric.
3. **Iso-FLOPs run:** same effective depth, compare against a genuinely
   `N·R`-deep non-looped model, to see whether DSP-LM's SSM+dendrite combo
   behaves like the paper's plain-Mamba results (real but partial gap) or
   its hybrid results (different pattern with attention layers mixed in —
   not directly applicable here since DSP-LM has no attention, but worth
   checking which plain-Mamba pattern we're closer to).
4. Only build the adaptive exit-gate (a real second project: frozen-backbone
   continued training, threshold sweep) after a plain looped variant shows a
   real win here — per the paper's own 370M-scale results, exit-gating
   doesn't reliably beat a well-tuned fixed loop count at larger sizes.

## Milestones

- M1 — `loop_count` param + iso-parameter comparison at the reference width
  on the local 3060. (~half a day, reuses existing sweep infrastructure)
- M2 — iso-FLOPs comparison at matched effective depth. (a training run)
- M3 — if M1/M2 show a real win, re-run the muP sweep *for this
  architecture's loop-count axis specifically* before trusting any LR
  transferred from the non-looped sweep.
- M4 (later, only if the above pays off) — adaptive exit-gate, two-stage
  training, designed jointly with `recurrent_inference.md`'s state handling.

## Why it's deferred

This is a real architecture change — new forward-pass control flow, a new
iso-parameter/iso-FLOPs comparison protocol, and likely its own follow-on
scaling-law work — not a hyperparameter tweak, so it doesn't belong in the
same fast-iteration loop as the LR/optimizer sweeps already running. It's
also only an interesting comparison *relative to* a non-looped baseline
that's actually well-tuned — which is exactly what the muP/Muon work in
`research_harness/` is establishing now. Worth returning to once that
baseline (push4 and whatever follows it) is solid, not before.
