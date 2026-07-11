"""Regression test for the fused Triton Vandermonde-kernel-construction op
(_VandermondeKernelFn / ResonatorSSMKernel.forward's fast path). Compares
against the naive PyTorch einsum reference on both forward output and
end-to-end gradients.

CUDA-only (skipped otherwise); tensors are deliberately tiny so this runs in
well under a second -- not a benchmark, just a correctness check.
"""

import pytest
import torch

import colab_trainable_dendritic_lm as m

requires_triton_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and m._HAS_TRITON),
    reason="needs a CUDA GPU with Triton available",
)


def _reference_forward(kernel_mod, length):
    a_bar, b_bar, c, dt_a = kernel_mod._discretise()
    c_mod = c * b_bar
    arange = torch.arange(length, device=a_bar.device)
    powers = torch.exp(dt_a.unsqueeze(-1) * arange)
    return 2.0 * torch.einsum("hn,hnl->hl", c_mod, powers).real


@requires_triton_cuda
@pytest.mark.parametrize(
    "d_model,n_states,length",
    [
        (20, 12, 37),      # odd/non-power-of-2 sizes
        (1, 2, 5),         # single channel
        (8, 64, 1),        # length-1 edge case
    ],
)
def test_triton_kernel_matches_reference_forward_and_grad(d_model, n_states, length):
    torch.manual_seed(0)
    kernel_mod = m.ResonatorSSMKernel(d_model, n_states=n_states).to("cuda")

    out_triton = kernel_mod(length)
    out_ref = _reference_forward(kernel_mod, length)
    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-3)

    grad_out = torch.randn(d_model, length, device="cuda")

    kernel_mod.zero_grad(set_to_none=True)
    (kernel_mod(length) * grad_out).sum().backward()
    grads_triton = {n: p.grad.clone() for n, p in kernel_mod.named_parameters()}

    kernel_mod.zero_grad(set_to_none=True)
    (_reference_forward(kernel_mod, length) * grad_out).sum().backward()
    grads_ref = {n: p.grad.clone() for n, p in kernel_mod.named_parameters()}

    for name in grads_triton:
        torch.testing.assert_close(grads_triton[name], grads_ref[name], atol=1e-4, rtol=1e-3)


@requires_triton_cuda
def test_triton_kernel_output_is_finite_at_odd_length_beyond_one_block():
    # Regression check for the backward kernel's over-L loop: length not a
    # multiple of its internal BLOCK_L (256) must still cover every position.
    kernel_mod = m.ResonatorSSMKernel(4, n_states=4).to("cuda")
    length = 257  # one full block + a 1-element remainder
    out = kernel_mod(length)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in kernel_mod.parameters())
