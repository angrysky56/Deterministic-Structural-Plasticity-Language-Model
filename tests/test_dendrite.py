"""Tests for DendriticMLP's variants.

These cover the pure-logic properties the dendrite ablation depends on: that
the historical gate really is dead (so the ablation has a meaningful control),
that the replacement gate can actually silence a branch, that the tail-up
decay mask degenerates correctly at both ends of lambda, and that all variants
stay parameter-matched so a val_bpb difference measures mechanism rather than
capacity.

No training here -- that's the harness's job (research_harness/) and the smoke
test's. See tests/README rationale in CLAUDE.md.
"""

import math

import pytest
import torch

from colab_trainable_dendritic_lm import (
    DENDRITE_VARIANTS,
    DendriticMLP,
    _branch_tree_distance,
)

D_MODEL = 64
NUM_BRANCHES = 8
BRANCH_DIM = 16


def _layer(variant: str, **kw) -> DendriticMLP:
    return DendriticMLP(
        D_MODEL,
        num_branches=NUM_BRANCHES,
        branch_dim=BRANCH_DIM,
        variant=variant,
        **kw,
    )


# ---------------------------------------------------------------------------
# The baseline gate is measurably dead -- this is the control the ablation
# compares against, so the claim needs to be pinned by a test rather than
# asserted in a docstring.
# ---------------------------------------------------------------------------


def test_baseline_gate_cannot_suppress_a_branch():
    """sigmoid(10*(sigmoid(l) - 0.1)) is bounded well away from zero.

    Its floor is sigmoid(-1) ~= 0.269, so the "unresolved branches are pushed
    toward zero" behaviour the layer was designed for cannot occur.
    """
    logit = torch.tensor([-1e9, -8.0, 0.0, 8.0, 1e9])
    gate = torch.sigmoid(10.0 * (torch.sigmoid(logit) - 0.1))
    assert gate.min() > 0.26, "floor should be ~0.269, not ~0"
    assert math.isclose(gate[0].item(), 0.2689, abs_tol=1e-3)
    # At init the branch_gate logits are ~0, so every branch starts wide open.
    assert gate[2] > 0.98


def test_baseline_gate_gradient_is_negligible_at_init():
    """d(gate)/d(logit) at logit=0 is ~0.044 -- the gate barely trains."""
    logit = torch.zeros(1, requires_grad=True)
    torch.sigmoid(10.0 * (torch.sigmoid(logit) - 0.1)).backward()
    assert logit.grad is not None
    assert logit.grad.abs().item() < 0.05


# ---------------------------------------------------------------------------
# The replacement gate
# ---------------------------------------------------------------------------


def test_nmda_gate_spans_the_full_range():
    """The self-gated NMDA threshold must reach both 0 and 1, unlike baseline."""
    layer = _layer("nmda")
    with torch.no_grad():
        layer.gate_log_k.fill_(math.log(4.0))  # steepen
    # Drive far below / far above threshold.
    pre_low = torch.full((1, 1, NUM_BRANCHES, BRANCH_DIM), -5.0)
    pre_high = torch.full((1, 1, NUM_BRANCHES, BRANCH_DIM), 5.0)
    assert layer._branch_gate_values(pre_low).max() < 1e-4
    assert layer._branch_gate_values(pre_high).min() > 0.999


def test_nmda_gate_is_supralinear_through_the_knee():
    """Output magnitude must grow FASTER than linearly across the threshold.

    That is the whole point of the NMDA analogue: sublinear below, supralinear
    through the knee, saturating above. A plain SiLU FFN has no such regime.
    """
    layer = _layer("nmda")
    with torch.no_grad():
        layer.gate_log_k.fill_(math.log(8.0))
        layer.gate_theta.fill_(1.0)

    @torch.no_grad()
    def branch_out_norm(scale: float) -> float:
        pre = torch.full((1, 1, NUM_BRANCHES, BRANCH_DIM), scale / layer._drive_scale)
        gate = layer._branch_gate_values(pre)
        return float((torch.nn.functional.silu(pre) * gate.unsqueeze(-1)).norm())

    # Ratio of output growth to input growth, below vs across the knee.
    below = branch_out_norm(0.6) - branch_out_norm(0.3)
    across = branch_out_norm(1.2) - branch_out_norm(0.9)
    assert across > below, "no supralinear regime across the threshold"


@pytest.mark.parametrize("variant", ["nmda", "compart", "tree"])
def test_new_variants_have_no_dead_gate_at_init(variant):
    """At init the gate must sit in the responsive band, not saturated.

    The baseline's failure mode was starting at 0.982 with a vanishing
    gradient; a gate that can't move can't learn a threshold.
    """
    layer = _layer(variant)
    torch.manual_seed(0)
    x = torch.randn(4, 8, D_MODEL)
    pre = layer.value_proj(x).view(4, 8, NUM_BRANCHES, BRANCH_DIM)
    gate = layer._branch_gate_values(pre)
    assert 0.05 < gate.mean() < 0.95, f"gate saturated at init: {gate.mean():.3f}"


# ---------------------------------------------------------------------------
# Temporal NMDA kinetics (nmda_t)
# ---------------------------------------------------------------------------


def test_nmda_kernel_has_epsp_shape():
    """Difference of exponentials: zero at t=0, single peak, monotone decay.

    This is the synaptic conductance form used in the biophysical models
    (Aizenbud et al. 2026, Methods Eq. 5/6), not an arbitrary smoothing
    window.
    """
    k = _layer("nmda_t")._nmda_kernel().detach()
    assert k.shape == (NUM_BRANCHES, 16)
    row = k[0]
    assert abs(float(row[0])) < 1e-6, "conductance must start at zero"
    peak = int(row.argmax())
    assert 0 < peak < 5, f"peak should be early, got t={peak}"
    tail = row[peak:]
    assert torch.all(tail[1:] <= tail[:-1] + 1e-6), "decay must be monotone"
    assert math.isclose(float(row.sum()), 1.0, rel_tol=1e-4)


def test_nmda_kernel_is_asymmetric_fast_rise_slow_decay():
    """Asymmetry is what makes a branch sensitive to input ORDER."""
    row = _layer("nmda_t")._nmda_kernel().detach()[0]
    peak = int(row.argmax())
    rise = float(row[: peak + 1].sum())
    decay = float(row[peak + 1 :].sum())
    assert decay > rise, "decay tail should carry more mass than the rise"


def test_nmda_t_drive_is_causal():
    """A token must never influence the drive at an earlier position.

    Tested by perturbation rather than by looking for a flat prefix: the first
    few positions legitimately differ from each other because the causal
    convolution pads with zeros and history ramps up. Causality is the claim
    that changing token t leaves every position < t untouched.
    """
    layer = _layer("nmda_t")
    x = torch.randn(1, 12, D_MODEL)
    perturbed = x.clone()
    perturbed[0, 6] += 9.0

    with torch.no_grad():
        shape = (1, 12, NUM_BRANCHES, BRANCH_DIM)
        before = layer._branch_drive(layer.value_proj(x).view(*shape))
        after = layer._branch_drive(layer.value_proj(perturbed).view(*shape))

    assert torch.allclose(before[:, :6], after[:, :6], atol=1e-6), (
        "perturbing token 6 changed the drive before it -- convolution leaked "
        "backwards"
    )
    assert not torch.allclose(
        before[:, 7:], after[:, 7:], atol=1e-4
    ), "perturbing token 6 had no forward effect -- the window is not working"


def test_nmda_t_gate_depends_on_context_not_current_token():
    """g(0)=0 for a difference of exponentials, so the gate reads only history.

    Deliberate, and the same motif as Numenta's active dendrites: recent
    context gates the current token's value rather than the token gating
    itself. Pinned because it is a real behavioural property, not an accident.
    """
    layer = _layer("nmda_t")
    pre = torch.zeros(1, 6, NUM_BRANCHES, BRANCH_DIM)
    quiet = layer._branch_gate_values(pre)[0, -1].clone()
    pre[0, -1] = 9.0  # blast the CURRENT token only
    assert torch.allclose(quiet, layer._branch_gate_values(pre)[0, -1], atol=1e-6)


def test_nmda_t_decay_stays_slower_than_rise():
    """Kernel must stay positive even if the optimiser pushes tau_rise up."""
    layer = _layer("nmda_t")
    with torch.no_grad():
        layer.log_tau_rise.fill_(math.log(50.0))  # invert the intended order
        layer.log_tau_decay.fill_(math.log(0.6))
    k = layer._nmda_kernel().detach()
    assert torch.all(k >= -1e-6), "kernel went negative; clamp failed"


# ---------------------------------------------------------------------------
# Tree geometry and the tail-up decay mask
# ---------------------------------------------------------------------------


def test_branch_tree_distance_matches_binary_tree():
    h = _branch_tree_distance(8)
    assert h.shape == (8, 8)
    assert torch.equal(h, h.T)
    assert h.diagonal().abs().max() == 0
    assert h[0, 1] == 2  # siblings
    assert h[0, 2] == 4 and h[0, 3] == 4  # same quarter-tree
    assert h[0, 4] == 6 and h[0, 7] == 6  # opposite halves
    assert h[6, 7] == 2  # siblings again


def test_branch_tree_distance_falls_back_to_chain():
    """Non-power-of-two branch counts have no balanced binary tree."""
    h = _branch_tree_distance(6)
    assert h[0, 5] == 5 and h[2, 3] == 1


def test_decay_mask_degenerates_correctly():
    """lambda -> 0 is block-diagonal; lambda -> inf is uniform (dense FFN)."""
    layer = _layer("compart")
    n_terr = D_MODEL // NUM_BRANCHES

    with torch.no_grad():
        layer.log_lambda.fill_(math.log(1e-3))
    tight = layer._decay_mask(torch.float32)
    # Only each branch's own territory survives.
    own = tight[:BRANCH_DIM, :n_terr]
    other = tight[:BRANCH_DIM, n_terr:]
    assert own.min() > 0
    assert other.abs().max() < 1e-3

    with torch.no_grad():
        layer.log_lambda.fill_(math.log(1e4))
    loose = layer._decay_mask(torch.float32)
    assert (loose.max() - loose.min()).abs() < 1e-2, "should be ~uniform"


def test_decay_mask_preserves_init_scale():
    """Unit second moment, so compartmentalisation isn't confounded with a
    smaller effective init -- otherwise the ablation measures init, not
    structure."""
    for lam in (0.5, 2.0, 4.0, 50.0):
        layer = _layer("compart")
        with torch.no_grad():
            layer.log_lambda.fill_(math.log(lam))
        mask = layer._decay_mask(torch.float32).detach()
        assert math.isclose(float(mask.pow(2).mean()), 1.0, rel_tol=1e-4)


def test_compart_requires_divisible_d_model():
    with pytest.raises(ValueError, match="divisible"):
        DendriticMLP(70, num_branches=8, branch_dim=16, variant="compart")


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="not in"):
        _layer("dendrite-of-theseus")


# ---------------------------------------------------------------------------
# Parity, shapes, gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", DENDRITE_VARIANTS)
def test_forward_shape_and_finiteness(variant):
    layer = _layer(variant)
    x = torch.randn(3, 7, D_MODEL)
    out = layer(x)
    assert out.shape == (3, 7, D_MODEL)
    assert torch.isfinite(out).all()


def _param_count(variant: str, d_model: int, branch_dim: int) -> int:
    return sum(
        p.numel()
        for p in DendriticMLP(
            d_model,
            num_branches=NUM_BRANCHES,
            branch_dim=branch_dim,
            variant=variant,
        ).parameters()
    )


@pytest.mark.parametrize("variant", DENDRITE_VARIANTS)
def test_new_variants_are_never_larger_than_baseline(variant):
    """Capacity must not be able to explain a win.

    baseline spends ``d_model*num_branches + num_branches`` on its (dead) gate
    projection. The new variants delete that and buy only a few per-branch
    scalars back, so they are strictly SMALLER. That asymmetry is deliberate:
    if a smaller model wins the ablation, extra capacity cannot be the reason.
    """
    base = _param_count("baseline", D_MODEL, BRANCH_DIM)
    got = _param_count(variant, D_MODEL, BRANCH_DIM)
    assert got <= base, f"{variant} grew: {got} vs baseline {base}"


@pytest.mark.parametrize("variant", ["nmda", "compart", "tree"])
def test_parameter_deficit_is_negligible_at_production_width(variant):
    """At the harness's real dimensions the gap must be well under 1%.

    The tiny dimensions used elsewhere in this file exaggerate it (the deleted
    gate projection is a much larger fraction of a 64-wide layer), so parity
    is asserted at the width the ablation actually runs at: the research
    harness config, d_model=256 / branch_dim=128.
    """
    base = _param_count("baseline", 256, 128)
    got = _param_count(variant, 256, 128)
    assert 0 <= (base - got) / base < 0.01, f"{variant}: {got} vs baseline {base}"


@pytest.mark.parametrize("variant", ["nmda", "compart", "tree"])
def test_all_new_parameters_receive_gradient(variant):
    """A mechanism that gets no gradient is decoration, not a mechanism."""
    layer = _layer(variant)
    layer(torch.randn(2, 5, D_MODEL)).square().mean().backward()
    new_params = {
        "gate_log_k",
        "gate_theta",
        "log_lambda",
        "junction_log_k",
        "junction_theta",
    }
    checked = 0
    for name, p in layer.named_parameters():
        if name in new_params:
            assert p.grad is not None, f"{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
            assert p.grad.abs().sum() > 0, f"{name} grad is identically zero"
            checked += 1
    assert checked >= 2


@pytest.mark.parametrize("variant", DENDRITE_VARIANTS)
def test_diagnostics_report_learned_values(variant):
    d = _layer(variant).diagnostics()
    assert d["variant"] == variant
    if variant == "baseline":
        assert "lambda" not in d
    else:
        assert math.isclose(d["gate_k_mean"], 1.0, rel_tol=1e-5)  # init k = 1
    if variant in ("compart", "tree"):
        assert math.isclose(d["lambda"], 4.0, rel_tol=1e-5)  # init lambda = 4
    if variant == "tree":
        assert "junction_k_mean" in d


def test_baseline_is_bit_for_bit_unchanged():
    """Old checkpoints and prior results.tsv numbers must still reproduce."""
    torch.manual_seed(0)
    layer = _layer("baseline")
    x = torch.randn(2, 4, D_MODEL)
    value = torch.nn.functional.silu(layer.value_proj(x)).view(
        2, 4, NUM_BRANCHES, BRANCH_DIM
    )
    logit = layer.branch_gate(x)
    gate = torch.sigmoid(10.0 * (torch.sigmoid(logit) - 0.1))
    expected = layer.out_proj((value * gate.unsqueeze(-1)).reshape(2, 4, -1))
    assert torch.allclose(layer(x), expected, atol=1e-6)


@pytest.mark.parametrize("variant", DENDRITE_VARIANTS)
def test_state_dict_roundtrip(variant):
    """Checkpoint save/load must survive per variant (chat.py depends on it)."""
    a, b = _layer(variant), _layer(variant)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, D_MODEL)
    a.eval()
    b.eval()
    assert torch.allclose(a(x), b(x), atol=1e-6)
