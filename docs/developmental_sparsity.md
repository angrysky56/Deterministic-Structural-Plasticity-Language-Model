# Developmental Structural Plasticity — Design Plan

Status: **planned (v2), not implemented.** Run this *after* a dense base run
finishes, so there is a dense baseline to measure against. Do not switch it on
mid-pretraining of an undertrained model (see Caveats).

## Why

Biological memory circuits (e.g. hippocampal CA3) don't start sparse and grow —
they **overproduce** connections early, then **prune** heavily with experience,
freezing into a sparse, efficient, distributed network. Synaptic
overproduction-then-pruning is well established developmental neuroscience
(Huttenlocher and much since); the "tabula plena → carve away" framing is a
clean way to state it.

This repo is literally *Deterministic Structural Plasticity*, and the dendritic
block already contains the hook: the asymmetric soma gate multiplies each
branch's value vector by a steep sigmoid gate, zeroing branches that don't
"structurally resolve." That is a **soft, reversible** shadow of pruning. This
plan makes it **hard and structural**: branches that stay dead get removed
(optionally regrown elsewhere), on a developmental schedule.

Prior ML art to build on (this is not an untried idea): the Lottery Ticket
Hypothesis (Frankle & Carbin), dynamic sparse training (SET, RigL), and gradual
magnitude pruning (Zhu & Gupta). The novelty here is doing it at the granularity
of **dendritic branches** and tying the schedule to the training curriculum.

## Where sparsity lives here

`DendriticMLP` splits the hidden width into `num_branches` groups of `branch_dim`
(so `d_ff = num_branches * branch_dim`). A branch is the natural structural unit:
pruning branch `b` removes its slice of `value_proj` (rows) and `out_proj`
(cols) plus its `branch_gate` row — a `~2 * d_model * branch_dim` block.

Design choice that makes this faithful to the biology: **overproduce, then
carve.** Initialise with *many* branches (e.g. `num_branches = 32` instead of 8,
`branch_dim` shrunk to keep `d_ff` constant), then prune down toward the dense
baseline's effective width. Start dense to absorb early data; end sparse.

## Mechanism

Per branch, maintain an EMA of its gate activation (how often/strongly it fires):

```
activity[l, b] <- (1-m) * activity[l, b] + m * gate[l][:, :, b].mean()
```

Then, on a schedule:

1. **Prune.** Branches with `activity < prune_thresh` sustained over a window are
   masked off permanently (mask = 0). Their params are zeroed and their grads
   frozen. (A binary mask is equivalent in effect; to actually save FLOPs/params
   the module is *compacted* — rebuilt smaller — after training.)
2. **Regrow (RigL-style, optional).** To conserve capacity while reallocating it,
   for every `k` branches pruned, reactivate `k` dormant branches: reinitialise
   their weights and set the `branch_gate` bias so they fire again, letting
   gradients decide if they earn their place. Keeps *active* count ~constant.
3. **Schedule (developmental).** Target sparsity ramps with the curriculum:
   0% through Phases 1–2 (absorb), rising to the target (e.g. keep 8 of 32
   branches) across Phases 3–4 (carve). Mirrors overproduction → experience-
   dependent pruning; also respects "don't prune before capacity is filled."

## Sketch (kept out of the live training file until v2)

```python
class BranchPruner:
    """Tracks per-branch gate activity and prunes/regrows on a schedule."""

    def __init__(self, model, prune_thresh=0.02, ema=0.99):
        self.thresh, self.ema = prune_thresh, ema
        # one activity vector + persistent keep-mask per DendriticMLP
        self.activity, self.masks = {}, {}
        for name, mod in model.named_modules():
            if isinstance(mod, DendriticMLP):
                self.activity[name] = torch.ones(mod.num_branches)
                self.masks[name] = torch.ones(mod.num_branches)

    @torch.no_grad()
    def observe(self, name, gate):            # gate: (B, T, num_branches)
        a = gate.mean(dim=(0, 1)).float().cpu()
        self.activity[name].mul_(self.ema).add_(a, alpha=1 - self.ema)

    @torch.no_grad()
    def step(self, model, target_keep):       # e.g. keep=8 of 32
        for name, mod in model.named_modules():
            if not isinstance(mod, DendriticMLP):
                continue
            act = self.activity[name] * self.masks[name]  # ignore already-dead
            n_prune = int(self.masks[name].sum().item()) - target_keep
            if n_prune > 0:
                dead = act.topk(n_prune, largest=False).indices
                self.masks[name][dead] = 0.0
            _apply_branch_mask(mod, self.masks[name])     # zero rows/cols + freeze
```

Wiring: `DendriticMLP.forward` exposes `gate` (already computed) to
`pruner.observe(...)`; the training loop calls `pruner.step(...)` at each
substep boundary with a `target_keep` from the schedule. A `_apply_branch_mask`
helper zeroes the corresponding `value_proj`/`out_proj`/`branch_gate` slices and
sets their `.grad` to zero each step (or uses a forward mask). Post-training,
`compact()` rebuilds each `DendriticMLP` with only the surviving branches for
real inference savings.

## Validation / what to measure

Run at equal token budget and compare against the dense baseline:

1. **Capability retention.** Per-domain eval loss (logic/math/physics/…) of
   overproduce-then-prune vs dense. Target: match or beat dense at fewer active
   params.
2. **Active parameters & inference FLOPs** after compaction.
3. **Where it carves.** Which branches/layers survive — is the surviving
   structure interpretable (e.g. more branches kept in later, abstract phases)?
4. **Ablations.** prune-without-regrow vs RigL-regrow; prune-early (should hurt)
   vs prune-late (should be safe) — this is the empirical test of the
   "don't prune an undertrained model" claim.

## Caveats (read before enabling)

- **Timing.** Pruning helps when the model is *over-parameterised for its data*.
  A base run that is undertrained (our case: ~1.4B tokens vs a ~10B appetite) is
  the opposite — pruning early removes capacity that was never filled. Prune late
  and gradually, or after the base model is well-trained.
- **Masking ≠ speed.** A binary mask gives representational sparsity and
  regularisation but *no* compute/memory savings until the module is compacted
  and (ideally) run with sparse kernels. Be clear which goal you're after.
- **Stability.** Aggressive pruning can collapse training; use the gradual
  schedule, gradient clipping (already on), and regrow to hedge.
- **Not Hebbian.** This stays with global backprop and only edits *structure*.
  Local Hebbian learning rules are a separate, much riskier research direction
  and are explicitly out of scope here.

## Milestones

- M1 — `BranchPruner` + `_apply_branch_mask`, activity tracking, a fixed-schedule
  prune at substep boundaries; unit test that masked branches stay zero and
  their grads are frozen. (~1 day)
- M2 — overproduce config (`num_branches=32`) + developmental schedule; a v2 run
  vs the dense baseline on the same token budget. (a training run)
- M3 — `compact()` for real inference savings + a RigL regrow ablation.
