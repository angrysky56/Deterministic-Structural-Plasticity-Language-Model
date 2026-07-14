"""Muon optimizer, vendored (single-device variant only -- this project trains
on one GPU at a time, local 3060 or a single Colab A100, never multi-GPU).

Source: Keller Jordan et al., https://github.com/KellerJordan/Muon (MIT-style,
see upstream repo for license/citation). Trimmed to drop the
torch.distributed all-gather machinery in the original `Muon` /
`MuonWithAuxAdam` classes -- `SingleDeviceMuonWithAuxAdam` below is upstream's
own single-GPU variant, copied verbatim (see their README's "Usage" section).

Muon is for 2D hidden weight matrices only. Embeddings, the (tied) output
head, and any 1D gains/biases should stay on AdamW -- this file provides
`SingleDeviceMuonWithAuxAdam` specifically so a model's existing
Muon-group/AdamW-group param split (already built for DSP-LM's own muP
implementation -- see train_harness.py's DSPLMHarness.setup_optimizer) can
drive one optimizer object instead of stepping two separately.

DSP-LM-specific note: the SSM kernel's pole parameters (log_A_real, A_imag,
log_dt, C in ResonatorSSMKernel) are NOT weight matrices in the sense Muon
targets -- log_A_real and A_imag happen to be 2D tensors (d_model, n_states/2)
for vectorised-per-channel storage, not a linear layer's input*W. Orthogonalizing
rows/columns of a per-channel pole array has no clear meaning the way it does
for an actual projection matrix, so these must stay excluded from the Muon
group by NAME (the existing ".kernel." check), not by ndim -- ndim alone
would incorrectly sweep them in since ndim(log_A_real) == ndim(A_imag) == 2.
"""

import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration approximating the orthogonalization (zeroth
    power) of G. Quintic iteration, coefficients chosen to maximize the
    slope at zero -- see upstream docstring for the full derivation note.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """One optimizer object driving Muon for `use_muon=True` groups and
    AdamW for `use_muon=False` groups. Non-distributed (no all-gather) --
    the right variant for a single local GPU or a single Colab GPU.

    Usage mirrors DSPLMHarness.setup_optimizer's existing 3-group split:
        param_groups = [
            dict(params=embedding_params, use_muon=False, lr=..., betas=..., weight_decay=0.0),
            dict(params=hidden_params,    use_muon=True,  lr=..., momentum=0.95, weight_decay=...),
            dict(params=no_decay_params,  use_muon=False, lr=..., betas=..., weight_decay=0.0),
        ]
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
