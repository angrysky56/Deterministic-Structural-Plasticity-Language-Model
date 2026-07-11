"""Tests for Config/preset resolution. Pure CPU, no GPU/network."""

import pytest

import colab_trainable_dendritic_lm as m


@pytest.mark.parametrize("preset", list(m.MODEL_PRESETS))
def test_every_preset_resolves_architecture_fields(preset):
    cfg = m.Config(preset=preset)
    assert cfg.d_model == m.MODEL_PRESETS[preset]["d_model"]
    assert cfg.depth == m.MODEL_PRESETS[preset]["depth"]
    assert cfg.branch_dim == m.MODEL_PRESETS[preset]["branch_dim"]
    assert cfg.batch_size == m.MODEL_PRESETS[preset]["batch_size"]
    assert cfg.grad_accum == m.MODEL_PRESETS[preset]["grad_accum"]
    assert cfg.lr == m.MODEL_PRESETS[preset]["lr"]


@pytest.mark.parametrize("preset", list(m.MODEL_PRESETS))
def test_output_dir_namespaced_per_preset(preset):
    cfg = m.Config(preset=preset)
    assert cfg.output_dir.endswith(f"/{preset}")


def test_explicit_fields_override_preset_defaults():
    cfg = m.Config(preset="42m", d_model=999, lr=1e-5)
    assert cfg.d_model == 999  # explicit override wins
    assert cfg.lr == 1e-5
    assert cfg.depth == m.MODEL_PRESETS["42m"]["depth"]  # untouched fields still inherit


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        m.Config(preset="does-not-exist")


def test_default_preset_targets_a100_per_recommendation_comment():
    # Config.preset's default is meant to match the "500m ... RECOMMENDED
    # max for a single A100 (DEFAULT)" comment on MODEL_PRESETS -- this is a
    # regression test for that specific mismatch, not a claim that 500m must
    # always be the right default.
    assert m.Config().preset == "500m"
