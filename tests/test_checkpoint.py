"""Tests for checkpoint save/load, including the torch.compile() "_orig_mod."
key-stripping fix (see save_checkpoint/load_checkpoint). CPU-only, tiny model,
fast -- no GPU needed since this is testing state_dict plumbing, not compute.
"""

import torch

import colab_trainable_dendritic_lm as m


def _tiny_model():
    return m.VectorizedDendriticLM(
        vocab_size=64, d_model=16, depth=1, n_states=4,
        num_branches=2, branch_dim=4, use_checkpoint=False,
    )


def _tiny_setup():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = m.make_scheduler(m.Config(preset="42m", warmup_steps=1), optimizer, total_steps=10)
    return model, optimizer, scheduler


def test_save_load_round_trip_preserves_weights(tmp_path):
    model, optimizer, scheduler = _tiny_setup()
    path = tmp_path / "ckpt.pt"
    m.save_checkpoint(str(path), model, optimizer, scheduler, step=3, completed_substeps=1, cfg=m.Config(preset="42m"))

    fresh = _tiny_model()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = m.make_scheduler(m.Config(preset="42m", warmup_steps=1), fresh_opt, total_steps=10)
    step, substeps = m.load_checkpoint(str(path), fresh, fresh_opt, fresh_sched, device="cpu")

    assert step == 3 and substeps == 1
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.equal(p1, p2)


class _FakeCompiledWrapper:
    """Stands in for torch.compile()'s OptimizedModule without actually
    invoking Triton/Inductor: same "_orig_mod" attribute + prefixed
    state_dict() that save_checkpoint/load_checkpoint must see through.
    """

    def __init__(self, orig_mod):
        self._orig_mod = orig_mod

    def state_dict(self):
        return {f"_orig_mod.{k}": v for k, v in self._orig_mod.state_dict().items()}

    def load_state_dict(self, sd):
        raise AssertionError("save/load_checkpoint must operate on _orig_mod, not the wrapper")


def test_save_checkpoint_strips_orig_mod_prefix(tmp_path):
    model, optimizer, scheduler = _tiny_setup()
    wrapped = _FakeCompiledWrapper(model)
    path = tmp_path / "ckpt.pt"
    m.save_checkpoint(str(path), wrapped, optimizer, scheduler, step=1, completed_substeps=0, cfg=m.Config(preset="42m"))

    ckpt = torch.load(str(path), map_location="cpu")
    assert not any(k.startswith("_orig_mod.") for k in ckpt["model"])

    # A plain (uncompiled) model must be able to load it directly, matching
    # what chat.py does with a checkpoint written by a compiled training run.
    plain = _tiny_model()
    plain.load_state_dict(ckpt["model"])


def test_load_checkpoint_loads_into_orig_mod_of_a_wrapped_model(tmp_path):
    model, optimizer, scheduler = _tiny_setup()
    path = tmp_path / "ckpt.pt"
    m.save_checkpoint(str(path), model, optimizer, scheduler, step=7, completed_substeps=2, cfg=m.Config(preset="42m"))

    fresh = _tiny_model()
    wrapped = _FakeCompiledWrapper(fresh)  # load_state_dict() on the wrapper itself would raise
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = m.make_scheduler(m.Config(preset="42m", warmup_steps=1), fresh_opt, total_steps=10)
    step, substeps = m.load_checkpoint(str(path), wrapped, fresh_opt, fresh_sched, device="cpu")

    assert (step, substeps) == (7, 2)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.equal(p1, p2)
