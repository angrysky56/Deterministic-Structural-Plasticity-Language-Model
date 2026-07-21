# -*- coding: utf-8 -*-
"""DSP-LM: Deterministic Structural Plasticity Language Model.

Colab-trainable dendritic language model with an SSM temporal mixer.

This script is the source of truth; convert to a notebook for Colab:
    jupytext --to notebook colab_trainable_dendritic_lm.py

Quick local sanity check (no downloads, tiny synthetic data):
    DSP_SMOKE=1 python colab_trainable_dendritic_lm.py

Architecture (v2):
    Each block is TWO residual sublayers, cleanly separated:

      1. ResonatorSSM  -- the temporal/sequence mixer.
         A diagonal, damped-complex-pole state-space model (S4D-style).
         Each channel is literally a damped resonator: pole = -exp(a) + i*w
         (negative real part = damping, imaginary part = oscillation).
         Unbounded receptive field, O(N log N) via FFT convolution.
         THIS is the "causal resonant field mixing" the project describes,
         and it is what makes long sequence length actually work -- the old
         dilated Conv1d only saw ~100 tokens no matter how long the input.

      2. DendriticMLP  -- the per-token nonlinear compute (attention-free
         FFN replacement). Input is fanned out into many independent
         "branches", each of which locally solves nonlinear logic before an
         asymmetric soma gate integrates them (Type-II-error avoidance:
         structurally unresolved states are pushed toward zero).

Separating time-mixing (SSM) from channel-mixing (dendrite) is what lets the
model scale to long context without the quadratic cost of self-attention and
without the finite window of a plain convolution.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import time
from dataclasses import dataclass, field

# Reduce CUDA fragmentation OOMs (the large logits tensor is the usual culprit).
# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = torch.cuda.is_available()
except ImportError:
    _HAS_TRITON = False

if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _dynamo_disable = torch.compiler.disable
else:  # very old torch without the public compiler API

    def _dynamo_disable(fn):
        return fn


# ==========================================================================
# 0. TRITON KERNEL  --  fused SSM-kernel Vandermonde construction
# ==========================================================================
#
# ResonatorSSMKernel.forward built its (d_model, length) convolution kernel by
# materialising a (d_model, n_states/2, length) complex Vandermonde tensor
# (`exp(dt_a * arange(length))`) and immediately contracting it away with an
# einsum. Profiled on the 110m preset (d_model=768, n_states=64, seq_len=2048)
# that ~400MB throwaway tensor was ~45% of ResonatorSSMKernel's own time and
# ~35% of the WHOLE model's forward pass -- and with use_checkpoint=True
# (the default) it's recomputed again in the backward pass, so ~35% of
# forward compute is paid twice per step. It also happens to be complex
# valued, and TorchInductor cannot generate Triton code for complex ops (the
# Triton language itself has no complex dtype) -- not a version/config
# limitation, a structural one -- so this tensor is exactly what poisoned
# torch.compile's memory footprint (see Config.torch_compile below).
#
# The fix: never materialise the (H, N, L) tensor at all. Since
# a_bar = exp(dt_a) with dt_a = neg_alpha + i*theta (both already real
# tensors from ResonatorSSMKernel._discretise, no polar<->cartesian
# conversion needed), the kernel value is
#     kernel[h, l] = 2 * sum_n exp(neg_alpha[h,n]*l) *
#                        (cm_real[h,n]*cos(theta[h,n]*l) - cm_imag[h,n]*sin(theta[h,n]*l))
# which is a pure real-valued reduction over the small n_states/2 axis (32 by
# default) -- Triton has no trouble with exp/cos/sin, only with a native
# complex dtype. The forward kernel below computes and reduces this per
# (channel, position) block without ever writing the (H, N, L) intermediate
# to global memory (FlashAttention-style); the backward kernel recomputes the
# same decay/cos/sin terms on the fly (never storing them either) and reduces
# over the sequence axis to get gradients w.r.t. neg_alpha/theta/cm_real/
# cm_imag. Ordinary PyTorch autograd handles everything before/after this op
# (the cheap (H, N)-sized pole discretisation), so no manual gradient for
# log_A_real/A_imag/log_dt/C is needed -- only for the 4 tensors this op
# actually touches.
if _HAS_TRITON:

    @triton.jit
    def _vandermonde_fwd_kernel(
        neg_alpha_ptr,
        theta_ptr,
        cm_real_ptr,
        cm_imag_ptr,
        out_ptr,
        N,
        L,
        BLOCK_L: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_l = tl.program_id(1)

        n_off = tl.arange(0, BLOCK_N)
        n_mask = n_off < N
        base_hn = pid_h * N + n_off
        neg_alpha = tl.load(neg_alpha_ptr + base_hn, mask=n_mask, other=0.0)
        theta = tl.load(theta_ptr + base_hn, mask=n_mask, other=0.0)
        cm_real = tl.load(cm_real_ptr + base_hn, mask=n_mask, other=0.0)
        cm_imag = tl.load(cm_imag_ptr + base_hn, mask=n_mask, other=0.0)

        l_off = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        l_mask = l_off < L
        pos = l_off.to(tl.float32)

        decay = tl.exp(neg_alpha[None, :] * pos[:, None])  # (BLOCK_L, BLOCK_N)
        ang = theta[None, :] * pos[:, None]
        term = decay * (cm_real[None, :] * tl.cos(ang) - cm_imag[None, :] * tl.sin(ang))
        term = tl.where(n_mask[None, :], term, 0.0)
        kernel_val = 2.0 * tl.sum(term, axis=1)

        tl.store(out_ptr + pid_h * L + l_off, kernel_val, mask=l_mask)

    @triton.jit
    def _vandermonde_bwd_kernel(
        grad_out_ptr,
        neg_alpha_ptr,
        theta_ptr,
        cm_real_ptr,
        cm_imag_ptr,
        d_neg_alpha_ptr,
        d_theta_ptr,
        d_cm_real_ptr,
        d_cm_imag_ptr,
        N,
        L,
        BLOCK_L: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        n_off = tl.arange(0, BLOCK_N)
        n_mask = n_off < N
        base_hn = pid_h * N + n_off
        neg_alpha = tl.load(neg_alpha_ptr + base_hn, mask=n_mask, other=0.0)
        theta = tl.load(theta_ptr + base_hn, mask=n_mask, other=0.0)
        cm_real = tl.load(cm_real_ptr + base_hn, mask=n_mask, other=0.0)
        cm_imag = tl.load(cm_imag_ptr + base_hn, mask=n_mask, other=0.0)

        acc_cm_real = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_cm_imag = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_neg_alpha = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_theta = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for l_start in range(0, L, BLOCK_L):
            l_off = l_start + tl.arange(0, BLOCK_L)
            l_mask = l_off < L
            pos = l_off.to(tl.float32)
            g = tl.load(grad_out_ptr + pid_h * L + l_off, mask=l_mask, other=0.0)
            g2 = tl.where(l_mask, 2.0 * g, 0.0)[:, None]  # (BLOCK_L, 1)

            decay = tl.exp(neg_alpha[None, :] * pos[:, None])  # (BLOCK_L, BLOCK_N)
            ang = theta[None, :] * pos[:, None]
            cos_a = tl.cos(ang)
            sin_a = tl.sin(ang)
            term = cm_real[None, :] * cos_a - cm_imag[None, :] * sin_a

            acc_cm_real += tl.sum(g2 * decay * cos_a, axis=0)
            acc_cm_imag += tl.sum(-g2 * decay * sin_a, axis=0)
            acc_neg_alpha += tl.sum(g2 * pos[:, None] * decay * term, axis=0)
            acc_theta += tl.sum(
                g2
                * decay
                * pos[:, None]
                * (-cm_real[None, :] * sin_a - cm_imag[None, :] * cos_a),
                axis=0,
            )

        tl.store(d_cm_real_ptr + base_hn, acc_cm_real, mask=n_mask)
        tl.store(d_cm_imag_ptr + base_hn, acc_cm_imag, mask=n_mask)
        tl.store(d_neg_alpha_ptr + base_hn, acc_neg_alpha, mask=n_mask)
        tl.store(d_theta_ptr + base_hn, acc_theta, mask=n_mask)

    class _VandermondeKernelFn(
        torch.autograd.Function
    ):  # pylint: disable=abstract-method
        """kernel[h,l] = 2*sum_n exp(neg_alpha[h,n]*l) *
        (cm_real[h,n]*cos(theta[h,n]*l) - cm_imag[h,n]*sin(theta[h,n]*l)),
        fused forward+backward, (H, N, L) never materialised."""

        @staticmethod
        def forward(ctx, neg_alpha, theta, cm_real, cm_imag, length):
            h, n = neg_alpha.shape
            block_n = triton.next_power_of_2(max(n, 1))
            block_l = 256
            out = torch.empty((h, length), device=neg_alpha.device, dtype=torch.float32)
            grid = (h, triton.cdiv(length, block_l))
            _vandermonde_fwd_kernel[grid](
                neg_alpha,
                theta,
                cm_real,
                cm_imag,
                out,
                n,
                length,
                BLOCK_L=block_l,
                BLOCK_N=block_n,
            )
            ctx.save_for_backward(neg_alpha, theta, cm_real, cm_imag)
            ctx.length, ctx.block_n, ctx.block_l = length, block_n, block_l
            return out

        @staticmethod
        def backward(ctx, grad_out):  # pylint: disable=arguments-differ
            neg_alpha, theta, cm_real, cm_imag = ctx.saved_tensors
            h, n = neg_alpha.shape
            grad_out = grad_out.contiguous()
            d_neg_alpha = torch.empty_like(neg_alpha)
            d_theta = torch.empty_like(theta)
            d_cm_real = torch.empty_like(cm_real)
            d_cm_imag = torch.empty_like(cm_imag)
            grid = (h,)
            _vandermonde_bwd_kernel[grid](
                grad_out,
                neg_alpha,
                theta,
                cm_real,
                cm_imag,
                d_neg_alpha,
                d_theta,
                d_cm_real,
                d_cm_imag,
                n,
                ctx.length,
                BLOCK_L=ctx.block_l,
                BLOCK_N=ctx.block_n,
            )
            return d_neg_alpha, d_theta, d_cm_real, d_cm_imag, None


# ==========================================================================
# 1. TEMPORAL MIXER  --  Diagonal damped-resonator SSM (S4D-style)
# ==========================================================================


class ResonatorSSMKernel(nn.Module):
    """Computes the SSM convolution kernel from diagonal damped-complex poles.

    Follows the S4D parameterization (Gu et al., NeurIPS 2022). Each of the
    ``d_model`` channels owns ``n_states`` complex conjugate poles. A pole is
    ``A = -exp(log_A_real) + i * A_imag`` so the real part is always negative
    (a stable, damped resonator) and the imaginary part sets its resonant
    frequency. The kernel is materialised only when needed and consumed by an
    FFT convolution, giving an effectively infinite, causal receptive field.
    """

    def __init__(
        self,
        d_model: int,
        n_states: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        # Store N/2 conjugate pairs; the kernel takes 2*Re(...) to recover the
        # full real response.
        half = n_states // 2

        # Per-channel timestep (discretisation step of the continuous system).
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(
            dt_min
        )
        self.log_dt = nn.Parameter(log_dt)

        # Output/readout weights C (complex), stored as real view for autograd.
        c = torch.randn(d_model, half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c))

        # Pole parameterisation. log_A_real -> damping; A_imag -> frequency.
        # S4D-Lin init: real part 1/2, frequencies pi * n.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, half)))
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(half).repeat(d_model, 1).float()
        )

    def _discretise(self):
        """Shared discretisation used by both convolutional and recurrent paths.

        Returns ``(A_bar, B_bar, C, dt_A)`` where:
        - ``A_bar = exp(dt * A)`` — discrete pole (damped complex), (H, N/2)
        - ``B_bar = (A_bar - 1) / A`` — ZOH-discretised input matrix, (H, N/2)
        - ``C`` — readout weights (unmodified), (H, N/2)
        - ``dt_A = dt * A`` — log-domain discrete pole (H, N/2); returned so
          callers needing it (the Vandermonde kernel construction) don't have
          to recover it via ``log(A_bar)``, a redundant exp/log round-trip
          that also risks a branch-cut wraparound on the imaginary part.
        """
        dt = torch.exp(self.log_dt)  # (H,)
        c = torch.view_as_complex(self.C)  # (H, N/2)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        dt_a = a * dt.unsqueeze(-1)  # (H, N/2)
        a_bar = torch.exp(dt_a)  # (H, N/2)
        b_bar = (a_bar - 1.0) / a  # (H, N/2)
        return a_bar, b_bar, c, dt_a

    def forward(self, length: int) -> torch.Tensor:
        """Return the real causal kernel of shape ``(d_model, length)``."""
        # Kernel math runs in float32 (complex ops are unsupported in bf16).
        a_bar, b_bar, c, dt_a = self._discretise()
        c_mod = c * b_bar  # fold B into C for the convolutional form

        if _HAS_TRITON and dt_a.is_cuda:
            # Fused Triton path: builds + reduces the Vandermonde sum without
            # ever materialising the (H, N/2, L) tensor. See the module
            # comment above _vandermonde_fwd_kernel.
            kernel = _VandermondeKernelFn.apply(
                dt_a.real.contiguous(),
                dt_a.imag.contiguous(),
                c_mod.real.contiguous(),
                c_mod.imag.contiguous(),
                length,
            )
        else:
            # CPU / no-triton fallback: identical math, naive materialisation.
            arange = torch.arange(length, device=a_bar.device)
            powers = torch.exp(dt_a.unsqueeze(-1) * arange)  # (H, N/2, L)
            kernel = 2.0 * torch.einsum("hn,hnl->hl", c_mod, powers).real  # (H, L)
        return kernel

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised hidden state: ``(B, H, N/2)`` complex float32."""
        half = self.log_A_real.shape[1]
        h = self.log_A_real.shape[0]
        return torch.zeros(batch_size, h, half, dtype=torch.cfloat, device=device)

    def step(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step (the dual of the FFT convolutional form).

        Args:
            x_t: Input token embedding, ``(B, H)`` real.
            h:   Hidden state from the previous step, ``(B, H, N/2)`` complex.

        Returns:
            ``(y_t, h_new)`` where ``y_t`` is ``(B, H)`` real and ``h_new``
            has the same shape as ``h``.
        """
        a_bar, b_bar, c, _ = self._discretise()  # (H, N/2) each
        # State update: h_new = A_bar * h + B_bar * x_t
        h_new = a_bar * h + b_bar * x_t.to(torch.cfloat).unsqueeze(-1)
        # Readout: y_t = 2 * Re(C · h_new)  (conjugate-pair reconstruction)
        y_t = 2.0 * torch.einsum("hn,bhn->bh", c, h_new).real
        return y_t, h_new


class ResonatorSSM(nn.Module):
    """Causal SSM sequence mixer using FFT convolution with a skip term."""

    def __init__(self, d_model: int, n_states: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel = ResonatorSSMKernel(d_model, n_states=n_states)
        self.D = nn.Parameter(torch.randn(d_model))  # direct feed-through skip
        # Gated output projection (GLU) — standard in modern SSM blocks.
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)

    @_dynamo_disable
    def _conv_mix(self, x: torch.Tensor) -> torch.Tensor:
        """FFT convolution + skip term -- everything complex-valued in this
        block. Deliberately kept OUT of torch.compile's trace entirely (not
        just the Vandermonde tensor the custom Triton kernel already
        isolates): on at least one Colab torch/Inductor build, compiling this
        path crashed outright with
        ``InductorError: AttributeError: 'complex' object has no attribute
        'get_name'`` inside ``inductor/ir.py``'s ``add_alias`` -- a different,
        harder failure than the memory blowup seen locally (which only
        warned "Torchinductor does not support code generation for complex
        operators" and fell back, slower but correct). Different Inductor
        versions appear to vary in how gracefully they degrade on complex
        dtypes, so the safe fix is to never let dynamo trace into ANY of it,
        not to rely on graceful fallback. The gated readout below has no
        complex ops and stays compilable.
        """
        # x: (B, T, H) -> operate along time.
        length = x.size(1)
        u = x.transpose(1, 2)  # (B, H, T)

        # Kernel + FFT convolution in float32 for numerical stability.
        kernel = self.kernel(length).to(torch.float32)  # (H, T)
        u32 = u.to(torch.float32)
        n_fft = 2 * length
        k_f = torch.fft.rfft(kernel, n=n_fft)  # (H, T_f)
        u_f = torch.fft.rfft(u32, n=n_fft)  # (B, H, T_f)
        y = torch.fft.irfft(u_f * k_f, n=n_fft)[..., :length]  # (B, H, T)
        y = y + u32 * self.D.unsqueeze(-1)
        return y.transpose(1, 2).to(x.dtype)  # (B, T, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional (training) path: process a full sequence via FFT."""
        y = self._conv_mix(x)
        # Gated readout -- pure real-valued; safe for torch.compile to trace
        # and fuse.
        return self.out_proj(y) * F.silu(self.gate_proj(x))

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised SSM hidden state."""
        return self.kernel.initial_state(batch_size, device)

    def step(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent (generation) path: process ONE token, O(1) per step.

        Args:
            x_t: Pre-norm'd token embedding, ``(B, H)``.
            h:   SSM hidden state from prior step, ``(B, H, N/2)`` complex.

        Returns:
            ``(out, h_new)``.
        """
        x32 = x_t.to(torch.float32)
        y_raw, h_new = self.kernel.step(x32, h)
        y_raw = y_raw + x32 * self.D  # skip connection
        y_raw = y_raw.to(x_t.dtype)
        # Gated readout (same gate as forward, but for a single token).
        return self.out_proj(y_raw) * F.silu(self.gate_proj(x_t)), h_new


# ==========================================================================
# 2. CHANNEL MIXER  --  Dendritic branch logic (per-token, attention-free)
# ==========================================================================


DENDRITE_VARIANTS = ("baseline", "nmda", "nmda_t", "compart", "tree")


def _branch_tree_distance(num_branches: int) -> torch.Tensor:
    """Pairwise distance between branches on a binary dendritic tree.

    Branch indices are read as paths through a balanced binary tree, so the
    distance between two branches is twice the number of levels back to their
    lowest common ancestor::

        h(i, j) = 0                             if i == j
        h(i, j) = 2 * (floor(log2(i XOR j)) + 1) otherwise

    For 8 branches this yields h in {0, 2, 4, 6}: siblings are close, branches
    in opposite halves of the tree are maximally far. This is the "stream
    distance" of the spatial-stream-network literature (Ver Hoef & Peterson
    2010) — a river network and a dendritic tree are the same object, a
    branching graph with flow toward one outlet.

    If ``num_branches`` is not a power of two there is no balanced binary tree,
    so we fall back to chain distance ``|i - j|`` (a single unbranched stream).

    Returns:
        ``(num_branches, num_branches)`` float32 distance matrix.
    """
    idx = torch.arange(num_branches)
    is_pow2 = num_branches > 0 and (num_branches & (num_branches - 1)) == 0
    if not is_pow2:
        return (idx[:, None] - idx[None, :]).abs().float()
    xor = torch.bitwise_xor(idx[:, None], idx[None, :])
    # floor(log2(xor)) + 1 == bit_length(xor); 0 for the diagonal.
    lvl = torch.where(
        xor > 0,
        torch.floor(torch.log2(xor.clamp(min=1).float())) + 1.0,
        torch.zeros_like(xor, dtype=torch.float32),
    )
    return 2.0 * lvl


class DendriticMLP(nn.Module):
    """Per-token nonlinear compute via semi-independent dendritic branches.

    The layer fans the token into ``num_branches`` branches, each of width
    ``branch_dim``, gates each branch, and integrates the survivors back to
    ``d_model`` through a soma projection.

    Four variants, nested so an ablation isolates one factor at a time. All
    are parameter-matched to within a handful of per-branch scalars, so a
    val_bpb difference is attributable to mechanism, not capacity:

    ``baseline``
        The historical layer, kept bit-for-bit so older runs reproduce.
        NOTE: its gate ``sigmoid(10*(sigmoid(w.x) - 0.1))`` is effectively
        dead. The double sigmoid bounds it to [0.269, 0.9999] — it can never
        suppress a branch — and at init (logits~0) every branch sits at 0.982
        with d(gate)/d(logit) = 0.044. In practice this trains as a plain
        two-layer SiLU FFN paying ``d_model * num_branches`` params for a
        near-constant multiplier. Measured, not assumed; see tests.

    ``nmda``
        Replaces that gate with a *self-gated* supralinear threshold. The
        gate reads the branch's OWN summed drive rather than an independent
        projection of the input, which is what makes it a coincidence
        detector: ``g = sigmoid(k * (drive - theta))`` with ``k`` and
        ``theta`` learnable per branch. Because ``drive`` scales with the
        branch's own activation, ``d|out|/d|in| = g + drive * g'`` — output
        grows *faster* than linearly through the knee (supralinear), is
        suppressed below it (sublinear), and saturates above. That is the
        NMDA summation curve Beniaguev et al. (2021) showed to be the sole
        source of a cortical neuron's I/O depth: delete NMDA from their
        biophysical model and a 7-layer temporal CNN collapses to one hidden
        layer. ``k`` is the steepness the human/rat comparison turns on, and
        it is learnable *and* logged — if training drives it up, the steep
        nonlinearity is earning its place; if it decays to 0, this mechanism
        is inert for language and we should say so.

    ``compart``
        ``nmda`` plus electrical decoupling. ``value_proj`` is masked by
        ``exp(-h(i,j) / lambda)`` over the branch tree, so a branch reads
        mostly its own territory of the input and neighbouring territories
        only weakly. This is the tail-up covariance of Ver Hoef & Peterson
        (2010): influence flows along the tree and decays with distance
        along it. ``lambda`` is one learnable scalar per layer and *is* the
        electrotonic length constant — ``lambda -> 0`` is hard block-diagonal
        (total compartmentalisation), ``lambda -> inf`` is the dense baseline.
        Rather than hand-picking a sparsity pattern we let each layer learn
        how compartmentalised it wants to be, then read it off. The mask is
        renormalised to unit second moment so it constrains *structure*
        without also shrinking the init scale (that would confound the
        ablation with an init change).

    ``tree``
        ``compart`` plus a nonlinear confluence. Sibling branches merge at a
        bifurcation and the merged drive passes a second, coarser gate that
        modulates both children — a branch-point spike gating everything
        distal to it. Applied multiplicatively so ``out_proj`` keeps its
        width: a *linear* mixing step before a dense ``out_proj`` would be a
        no-op (composition of two linear maps is a linear map the dense layer
        can already express), so the confluence has to be nonlinear to add
        anything at all.

    History: v1 reduced each branch to a single SCALAR before integrating, so
    a huge synaptic projection fed a d_model-wide scalar bottleneck and threw
    away almost all of its capacity. v2 (``baseline`` here) fixed that by
    gating the full branch VECTOR. v3 adds the mechanisms above.
    """

    def __init__(
        self,
        d_model: int,
        num_branches: int = 8,
        branch_dim: int = 256,
        threshold: float = 0.1,
        gate_steepness: float = 10.0,
        variant: str = "baseline",
        lambda_init: float = 4.0,
        nmda_window: int = 16,
    ) -> None:
        super().__init__()
        if variant not in DENDRITE_VARIANTS:
            raise ValueError(f"variant={variant!r} not in {DENDRITE_VARIANTS}")
        self.d_model = d_model
        self.num_branches = num_branches
        self.branch_dim = branch_dim
        self.d_ff = num_branches * branch_dim  # total hidden width
        self.threshold = threshold
        self.gate_steepness = gate_steepness
        self.variant = variant

        self.value_proj = nn.Linear(d_model, self.d_ff)  # branch value vectors
        self.out_proj = nn.Linear(self.d_ff, d_model)  # soma integration

        if variant == "nmda_t":
            # NMDA kinetics as a difference of exponentials -- the same form
            # Aizenbud et al. (2026) use for synaptic conductance (their
            # Methods Eq. 5/6): g(t) ~ exp(-t/tau_d) - exp(-t/tau_r).
            # Initialised to a short, asymmetric window (rise ~1 token, decay
            # ~4) because NMDA rises fast and decays slowly; that asymmetry is
            # what makes the branch sensitive to input ORDER, not just
            # coincidence (Branco & Hausser, dendritic discrimination of
            # temporal sequences -- ref 25/54 in the paper).
            self.nmda_window = nmda_window
            self.log_tau_rise = nn.Parameter(torch.zeros(num_branches))  # tau=1
            self.log_tau_decay = nn.Parameter(
                torch.full((num_branches,), math.log(4.0))
            )
            self.register_buffer(
                "_kernel_t",
                torch.arange(nmda_window, dtype=torch.float32),
                persistent=False,
            )

        if variant == "baseline":
            self.branch_gate = nn.Linear(d_model, num_branches)
        else:
            # Self-gating: no input projection, the branch reads its own drive.
            # Scale so `drive` has O(1) variance at init regardless of width
            # (mean of `branch_dim` iid terms shrinks as 1/sqrt(branch_dim)),
            # which puts theta=0 / k=1 in the responsive part of the sigmoid
            # instead of the saturated tail the baseline gate starts in.
            self._drive_scale = math.sqrt(branch_dim)
            self.gate_log_k = nn.Parameter(torch.zeros(num_branches))  # k = 1
            self.gate_theta = nn.Parameter(torch.zeros(num_branches))

        if variant in ("compart", "tree"):
            if d_model % num_branches != 0:
                raise ValueError(
                    f"variant={variant!r} needs d_model ({d_model}) divisible "
                    f"by num_branches ({num_branches}) to assign input "
                    f"territories to branches."
                )
            self.register_buffer(
                "_tree_dist", _branch_tree_distance(num_branches), persistent=False
            )
            self.log_lambda = nn.Parameter(torch.tensor(math.log(lambda_init)))

        if variant == "tree":
            if num_branches % 2 != 0:
                raise ValueError(
                    f"variant='tree' needs an even num_branches, got {num_branches}"
                )
            n_junction = num_branches // 2
            self.junction_log_k = nn.Parameter(torch.zeros(n_junction))
            self.junction_theta = nn.Parameter(torch.zeros(n_junction))

    # -- internals ---------------------------------------------------------

    def _decay_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """Tail-up decay mask over ``value_proj``, shape ``(d_ff, d_model)``.

        Computed in float32 (exp of a learnable reciprocal is bf16-hostile),
        renormalised to unit second moment so masking changes *which* weights
        matter without changing the init scale, then broadcast from the small
        ``(num_branches, num_branches)`` territory matrix up to the full
        weight shape.
        """
        lam = self.log_lambda.float().exp().clamp(min=1e-3, max=1e4)
        m = torch.exp(-self._tree_dist / lam)  # (nb, nb)
        # Unit second moment: expansion repeats each entry uniformly, so the
        # second moment of the expanded mask equals that of this small one.
        m = m / m.pow(2).mean().sqrt().clamp(min=1e-6)
        return (
            m.repeat_interleave(self.branch_dim, dim=0)
            .repeat_interleave(self.d_model // self.num_branches, dim=1)
            .to(dtype)
        )

    def _nmda_kernel(self) -> torch.Tensor:
        """Difference-of-exponentials NMDA conductance kernel, ``(nb, window)``.

        Follows the synaptic form used in the biophysical models this is
        derived from: ``g(t) ~ exp(-t/tau_decay) - exp(-t/tau_rise)``,
        peak-normalised. Causal by construction (t >= 0 only). tau_decay is
        held above tau_rise so the kernel stays positive and asymmetric --
        fast rise, slow decay, the shape that makes the branch respond to
        input ORDER rather than to a symmetric coincidence window.
        """
        t = self._kernel_t  # (window,)
        tau_r = self.log_tau_rise.float().exp().clamp(min=0.5, max=64.0)
        tau_d = self.log_tau_decay.float().exp().clamp(min=0.5, max=256.0)
        tau_d = torch.maximum(tau_d, tau_r * 1.05)  # keep decay slower
        k = torch.exp(-t[None, :] / tau_d[:, None]) - torch.exp(
            -t[None, :] / tau_r[:, None]
        )
        return k / k.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def _branch_drive(self, pre: torch.Tensor) -> torch.Tensor:
        """Per-branch synaptic drive, ``(B, T, num_branches)``.

        For ``nmda_t`` the drive is integrated over a causal time window with
        NMDA kinetics rather than read off the current token. This is the
        structural point: the neuron whose complexity this layer is imitating
        is measured by a *temporally* convolutional network at 1 ms
        resolution, and its supralinear threshold is crossed by inputs that
        coincide within an NMDA time constant -- roughly tens of
        milliseconds -- not within an instant. An instantaneous gate is a
        transformer idiom wearing a biological name; this is the mechanism.
        """
        drive = pre.mean(dim=-1) * self._drive_scale  # (B, T, nb)
        if self.variant != "nmda_t":
            return drive
        b, t, nb = drive.shape
        kernel = self._nmda_kernel().to(drive.dtype)  # (nb, window)
        # Depthwise causal conv over time: one kernel per branch, so this is
        # (B, nb, T) not (B, d_ff, T) -- negligible next to value_proj.
        x = drive.transpose(1, 2)  # (B, nb, T)
        x = F.pad(x, (kernel.size(-1) - 1, 0))
        out = F.conv1d(x, kernel.flip(-1).unsqueeze(1), groups=nb)
        return out.transpose(1, 2)  # (B, T, nb)

    def _branch_gate_values(self, pre: torch.Tensor) -> torch.Tensor:
        """Self-gated NMDA nonlinearity from each branch's own drive.

        Args:
            pre: Branch pre-activations, ``(B, T, num_branches, branch_dim)``.

        Returns:
            Gate in ``[0, 1)``, ``(B, T, num_branches)``. Unlike the baseline
            gate this reaches 0, so a branch can actually be silenced.
        """
        drive = self._branch_drive(pre)
        k = self.gate_log_k.exp()  # positive steepness
        return torch.sigmoid(k * (drive - self.gate_theta))

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        if self.variant == "baseline":
            value = F.silu(self.value_proj(x)).view(
                b, t, self.num_branches, self.branch_dim
            )
            logit = self.branch_gate(x)  # (B, T, num_branches)
            gate = torch.sigmoid(
                self.gate_steepness * (torch.sigmoid(logit) - self.threshold)
            )
            gated = value * gate.unsqueeze(-1)
            return self.out_proj(gated.reshape(b, t, self.d_ff))

        # Synaptic projection, optionally compartmentalised by tree distance.
        if self.variant in ("compart", "tree"):
            weight = self.value_proj.weight * self._decay_mask(
                self.value_proj.weight.dtype
            )
            pre = F.linear(x, weight, self.value_proj.bias)
        else:
            pre = self.value_proj(x)
        pre = pre.view(b, t, self.num_branches, self.branch_dim)

        gate = self._branch_gate_values(pre)  # (B, T, nb)

        if self.variant == "tree":
            # Confluence: siblings merge, the junction gate modulates both.
            merged = pre.view(b, t, -1, 2, self.branch_dim).sum(dim=3)
            j_drive = merged.mean(dim=-1) * self._drive_scale  # (B, T, nb/2)
            j_gate = torch.sigmoid(
                self.junction_log_k.exp() * (j_drive - self.junction_theta)
            )
            gate = gate * j_gate.repeat_interleave(2, dim=-1)

        gated = F.silu(pre) * gate.unsqueeze(-1)
        return self.out_proj(gated.reshape(b, t, self.d_ff))

    # -- diagnostics -------------------------------------------------------

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        """Learned nonlinearity/compartmentalisation readout for this layer.

        ``gate_k`` is the NMDA steepness the layer chose (up => the steep
        supralinear transition is being used; ~0 => the layer linearised the
        gate away and the mechanism is inert). ``lambda`` is the electrotonic
        length constant in branch-tree units (small => compartmentalised,
        large => the layer reverted to a dense FFN).
        """
        out: dict[str, float] = {"variant": self.variant}  # type: ignore[dict-item]
        if self.variant != "baseline":
            out["gate_k_mean"] = float(self.gate_log_k.exp().mean())
            out["gate_k_max"] = float(self.gate_log_k.exp().max())
            out["gate_theta_mean"] = float(self.gate_theta.mean())
        if self.variant == "nmda_t":
            out["tau_rise_mean"] = float(self.log_tau_rise.exp().mean())
            out["tau_decay_mean"] = float(self.log_tau_decay.exp().mean())
        if self.variant in ("compart", "tree"):
            out["lambda"] = float(self.log_lambda.exp())
        if self.variant == "tree":
            out["junction_k_mean"] = float(self.junction_log_k.exp().mean())
        return out


# ==========================================================================
# 3. BLOCK + FULL MODEL
# ==========================================================================


class DendriticResonatorBlock(nn.Module):
    """One block: pre-norm SSM time-mix + pre-norm dendritic channel-mix."""

    def __init__(
        self,
        d_model: int,
        n_states: int = 64,
        num_branches: int = 8,
        branch_dim: int = 256,
        dendrite_variant: str = "baseline",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = ResonatorSSM(d_model, n_states=n_states)
        self.norm2 = nn.LayerNorm(d_model)
        self.dendrite = DendriticMLP(
            d_model,
            num_branches=num_branches,
            branch_dim=branch_dim,
            variant=dendrite_variant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.dendrite(self.norm2(x))
        return x

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised per-block SSM hidden state."""
        return self.ssm.initial_state(batch_size, device)

    def step(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent single-token step through this block.

        Args:
            x_t: Token representation, ``(B, H)``.
            h:   SSM hidden state, ``(B, H, N/2)`` complex.

        Returns:
            ``(x_out, h_new)``.
        """
        ssm_out, h_new = self.ssm.step(self.norm1(x_t), h)
        x_t = x_t + ssm_out
        # Dendrite is per-token — add a T=1 dim, apply, squeeze back.
        x_t = x_t + self.dendrite(self.norm2(x_t).unsqueeze(1)).squeeze(1)
        return x_t, h_new


class VectorizedDendriticLM(nn.Module):
    """DSP-LM: dendritic branches over a diagonal-SSM temporal backbone.

    No positional embeddings are needed: the SSM is inherently sequential and
    causal, so position is encoded by the recurrence itself.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        depth: int = 6,
        n_states: int = 64,
        num_branches: int = 8,
        branch_dim: int = 256,
        use_checkpoint: bool = True,
        dendrite_variant: str = "baseline",
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                DendriticResonatorBlock(
                    d_model,
                    n_states=n_states,
                    num_branches=num_branches,
                    branch_dim=branch_dim,
                    dendrite_variant=dendrite_variant,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        # Weight init (GPT-2 style). Without this, nn.Embedding defaults to
        # N(0,1); tied to the output head that yields logits ~sqrt(d_model) in
        # scale, so initial loss is ~10x above ln(vocab) and training stalls.
        self.apply(self._init_weights)
        # Scale residual output projections by 1/sqrt(2*depth) so the residual
        # stream doesn't grow with depth (GPT-2 / nanoGPT trick).
        residual_std = 0.02 / math.sqrt(2 * depth)
        for pname, p in self.named_parameters():
            if pname.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm_out(x)
        # return_hidden lets the loss compute logits in chunks (fused CE) so the
        # full (B, T, vocab) tensor is never materialised — the main memory sink.
        return x if return_hidden else self.lm_head(x)

    # -- Recurrent (generation) interface --------------------------------

    def initial_states(
        self, batch_size: int, device: torch.device
    ) -> list[torch.Tensor]:
        """Zero-initialised hidden states for every block."""
        return [block.initial_state(batch_size, device) for block in self.blocks]

    def step(
        self, token_ids: torch.Tensor, states: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process a single token through all blocks using the recurrent SSM.

        Args:
            token_ids: ``(B,)`` — one token id per sequence in the batch.
            states:    Per-block hidden states (from ``initial_states`` or a
                       previous ``step`` call).

        Returns:
            ``(logits, new_states)`` where logits is ``(B, vocab_size)``.
        """
        x = self.embedding(token_ids)  # (B, H)
        new_states: list[torch.Tensor] = []
        for block, h in zip(self.blocks, states, strict=False):
            x, h_new = block.step(x, h)
            new_states.append(h_new)
        logits = self.lm_head(self.norm_out(x))  # (B, vocab)
        return logits, new_states

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Autoregressive generation using the O(1)-per-step recurrent form.

        The prompt is processed token-by-token through the recurrent SSM to
        build up hidden state (compressed context), then each new token is
        generated with a single ``step`` call — no FFT re-convolution over
        the growing sequence.
        """
        was_training = self.training
        self.eval()

        batch_size = start_tokens.size(0)
        device = start_tokens.device
        states = self.initial_states(batch_size, device)

        # Prefill: step through the prompt to build up hidden states.
        for t in range(start_tokens.size(1)):
            logits, states = self.step(start_tokens[:, t], states)

        # Generate: each new token is O(1) — just one recurrent step.
        idx = start_tokens
        for _ in range(max_new_tokens):
            scaled = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[:, [-1]]] = -float("inf")
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            logits, states = self.step(next_token.squeeze(1), states)

        if was_training:
            self.train()
        return idx


# ==========================================================================
# 4. CONFIG
# ==========================================================================


# Model-size presets. Each sets architecture (d_model/depth/branch_dim, keeping
# d_ff = 4*d_model with num_branches=8) plus batch/grad_accum/lr. Effective batch
# is held at ~96 across presets. Token needs are ~20 tokens/param (Chinchilla).
#   42m  ~0.8B tokens  fastest, proof-of-concept
#   110m ~2.2B tokens  strong capability-per-hour
#   500m ~6-10B tokens RECOMMENDED max for a single A100 (DEFAULT)
#   1b   ~20B tokens   fits an 80GB A100 but needs ~10+ A100-days — will be badly
#                      undertrained on one GPU; included so you can experiment.
#
# batch_size is tuned for an **80GB** A100 with gradient checkpointing on; watch
# the GPU-RAM gauge and push batch_size higher to fill ~60-70GB (raise grad_accum
# to keep the same effective batch). On a 40GB A100 or a 12GB card, halve/quarter
# batch_size. To trade the freed VRAM for ~30% more speed instead, you can also
# set use_checkpoint=False (works at smaller batch; it OOMs with a big batch).
# batch sizes assume fused cross-entropy (logits are no longer the memory sink),
# so memory is activation-bound (~0.2GB/sample for 110m). Tuned to use ~30-45GB
# of an 80GB A100; raise batch_size to fill more (watch the GPU gauge). The
# auto-scaler in main() shrinks these on smaller cards.
MODEL_PRESETS = {
    "42m": {
        "d_model": 512,
        "depth": 6,
        "branch_dim": 256,
        "batch_size": 256,
        "grad_accum": 1,
        "lr": 3e-4,
    },
    "110m": {
        "d_model": 768,
        "depth": 12,
        "branch_dim": 384,
        "batch_size": 128,
        "grad_accum": 2,
        "lr": 3e-4,
    },
    "500m": {
        "d_model": 1536,
        "depth": 18,
        "branch_dim": 768,
        "batch_size": 48,
        "grad_accum": 4,
        "lr": 2e-4,
    },
    "1b": {
        "d_model": 2048,
        "depth": 22,
        "branch_dim": 1024,
        "batch_size": 24,
        "grad_accum": 4,
        "lr": 1.5e-4,
    },
}


@dataclass
class Config:
    # Pick model scale here; override any field below to customise. Targets
    # an A100 (Colab) by default, per the "500m RECOMMENDED... (DEFAULT)"
    # comment above -- local iteration on a smaller card should pass an
    # explicit preset (e.g. `Config(preset="42m")`) rather than relying on
    # this default.
    preset: str = "500m"

    # Architecture — left None to inherit from the preset.
    d_model: int | None = None
    depth: int | None = None
    branch_dim: int | None = (
        None  # width of each branch (d_ff = num_branches*branch_dim)
    )
    n_states: int = 64  # SSM states per channel (N/2 conjugate pole pairs)
    num_branches: int = 8  # dendritic branches per token
    # Dendrite mechanism — see DendriticMLP's docstring. Nested ablation:
    #   baseline -> historical layer (its soma gate is measurably dead; it
    #               trains as a plain SiLU FFN). Kept so old runs reproduce.
    #   nmda     -> + self-gated supralinear branch threshold (learnable
    #               steepness k, the human/rat NMDA variable)
    #   compart  -> + tail-up exp(-h/lambda) masking of value_proj (learnable
    #               electrotonic length constant lambda per layer)
    #   tree     -> + nonlinear confluence gate at each bifurcation
    # Parameter-matched to within a few per-branch scalars, so an ablation
    # measures mechanism rather than capacity. Run `diagnose` after training
    # to read the learned k / lambda back out.
    dendrite_variant: str = "baseline"
    use_checkpoint: bool = True
    # 8-bit AdamW (bitsandbytes): quantizes the momentum/variance buffers to
    # ~1 byte/param instead of fp32's 8, cutting optimizer VRAM ~4x (e.g. ~12GB
    # -> ~3GB at the 1b preset). GPU-resident, no host transfer, so unlike CPU
    # offload it doesn't touch the training loop -- just the optimizer ctor.
    # Needs `pip install bitsandbytes` (Colab) or `uv sync --extra eightbit`
    # (local); falls back to plain AdamW with a warning if unavailable. Off by
    # default since it changes optimizer numerics and only 500m/1b need it.
    optim_8bit: bool = False
    # torch.compile() the model. History here matters -- read before flipping
    # this on:
    #   1) Originally, Inductor couldn't codegen the complex rfft/irfft SSM
    #      kernel (Triton has no complex dtype) and fell back to an unfused
    #      eager kernel *inside* the compiled graph, ballooning peak VRAM
    #      ~2.4x for only ~3% speedup (measured on a local RTX 3060).
    #   2) ResonatorSSMKernel.forward's Vandermonde tensor construction (the
    #      biggest complex-valued piece) was moved into a custom Triton op,
    #      opaque to dynamo, which fixed that memory blowup locally.
    #   3) On an actual Colab run (torch_compile=True was briefly the
    #      default), that was NOT enough: a *different* torch/Inductor build
    #      crashed outright -- InductorError: AttributeError: 'complex'
    #      object has no attribute 'get_name', inside inductor/ir.py's
    #      add_alias -- from tracing the REMAINING complex ops (pole
    #      discretisation, rfft/irfft, complex multiply) in
    #      ResonatorSSM._conv_mix, which didn't reproduce locally (only
    #      warned there). Inductor's handling of complex dtypes evidently
    #      isn't consistent across versions/builds.
    #   4) Fix: ResonatorSSM._conv_mix (the whole complex-valued FFT-conv
    #      path, not just the Vandermonde piece) is now decorated with
    #      @_dynamo_disable, so dynamo never traces into ANY complex op
    #      there, regardless of Inductor version. The real-valued gated
    #      readout in ResonatorSSM.forward is unaffected and stays
    #      compilable.
    # Left OFF by default pending verification of (4) on the environment
    # where (3) actually happened -- I can only test locally (RTX 3060,
    # where it never crashed in the first place). If you confirm
    # torch_compile=True trains cleanly on your Colab image, it's safe to
    # flip this back on.
    torch_compile: bool = False

    # Data / optimisation (batch_size/grad_accum/lr inherit from the preset if None).
    seq_len: int = 2048  # SSM gives real long context (was 256)
    batch_size: int | None = None
    grad_accum: int | None = None
    lr: float | None = None
    # 0.1 is standard (Chinchilla/Llama). Data-constrained scaling work (Lovelace
    # et al. 2026) finds strong decay ~1.0 cuts repetition-overfitting ~70%; go
    # higher (up to ~1.0) if you lean heavily on small/repeated domains.
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    # z-loss: tiny penalty on logit magnitude (log-sum-exp^2). Stabilises SSM
    # training and prevents loss spikes / logit blow-up (as in PaLM, Gemma).
    z_loss_weight: float = 1e-4
    min_lr_ratio: float = 0.1  # LR floor at the end of the final decay
    # WSD schedule: warmup -> stable plateau at full LR -> decay only over the
    # final fraction of training. Keeps every curriculum phase at full LR
    # (cosine-over-whole-run otherwise starves the later physics/philosophy
    # phases). decay_frac is the tail portion spent decaying.
    decay_frac: float = 0.2

    # Curriculum.
    steps_per_substep: int = 3000  # try 500+ for a trial run (see token math note)
    log_every: int = 100
    save_every: int = (
        100  # write latest.pt locally every N optimizer steps (crash safety)
    )
    hf_push_every: int = 300  # also PUSH to HF Hub every N steps (mid-phase durability)

    # Held-out evaluation: hold out the first eval_docs rows of each dataset
    # (train skips them), pack a few blocks each, and report eval loss at every
    # checkpoint to distinguish generalisation from memorisation.
    eval_docs: int = 200
    eval_blocks_per: int = 3
    # BASE pretraining (default): plain-prose only, no User:/Assistant: template.
    # This stops the model learning chat scaffolding as a generation attractor
    # or regurgitating templated prompts (e.g. reasoning-core's "Premise: ...").
    # Set chat_format=True (and usually mask_prompt_loss=True) for a later SFT
    # phase to teach turn-taking; supervise only the response then.
    chat_format: bool = False
    mask_prompt_loss: bool = False

    output_dir: str = "./checkpoints/DSP_LM"
    resume: bool = True  # resume from latest checkpoint (skips finished substeps)

    # Hugging Face Hub: set this to auto-sync checkpoints between Colab and
    # your local machine. Format: "username/repo-name". Created as a PRIVATE
    # repo on first push. Pull locally with: python colab_trainable_dendritic_lm.py pull
    hf_repo: str = "angrysky/dendritic-lm"  # e.g. "your_hf_name/dendritic-lm"

    # Dataset slate: balanced, non-gated, all-streamable. Each entry is
    # (repo, config_or_None). "language" (fineweb-edu) is the general-English
    # backbone — it teaches grammar, vocabulary and world knowledge, present in
    # every phase so fluency is always reinforced (the formal/technical corpora
    # alone would leave the model stilted).
    #
    #   language   HuggingFaceFW/fineweb-edu:sample-10BT      10B tokens          grammar + fluency + knowledge
    #   logic      reasoning-core/procedural-pretraining-pile ~7.3GB / 3.1M rows  formal, correct-by-design
    #   math       open-web-math/open-web-math                14.7B tokens        never runs out
    #   physics    millawell/wikipedia_field_of_science       ~9.6GB science wiki broad science, never runs out
    #   philosophy sayhan/strix-philosophy-qa                 ~391MB / 134k       philosophy Q&A (final phase)
    #
    # Streaming reads only the consumed slice, so pool size is free; these pools
    # are large enough that nothing cycles at these step counts. Q&A/instruction
    # (alpaca, orca-math) is intentionally deferred to a later SFT phase, not the
    # base run. Swap options (all non-gated):
    #   grammar bootstrap -> ("roneneldan/TinyStories", None)     simple stories, teaches basic syntax
    #   logic scale-up    -> ("reasoning-core/basic-procedural", None)   7.6M rows
    #   humanities scale  -> ("HuggingFaceTB/cosmopedia", "stanford")    6.3GB (broader than philosophy)
    repos: dict = field(
        default_factory=lambda: {
            "grammar": ("grammarly/coedit", None),  # explicit grammar correction
            "language": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
            # Diverse natural-language reasoning (question + chain-of-thought),
            # backtranslated from real corpora — NOT procedural templates, so a
            # small model can't just memorise it (reasoning-core saturated at
            # ~0.4 eval loss). ~6GB, CC-BY-NC (research use).
            "logic": ("facebook/natural_reasoning", None),
            "math": ("open-web-math/open-web-math", None),
            "physics": ("millawell/wikipedia_field_of_science", None),
            "philosophy": ("sayhan/strix-philosophy-qa", None),
            "humanities": (
                "HuggingFaceTB/cosmopedia",
                "stanford",
            ),  # ~6.3GB academic prose
        }
    )

    def __post_init__(self):
        # Fill any architecture / batch field left as None from the chosen preset.
        if self.preset not in MODEL_PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose {list(MODEL_PRESETS)}"
            )
        for key, value in MODEL_PRESETS[self.preset].items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        # Key checkpoints by preset so different sizes never overwrite each other
        # (local dir and HF Hub subfolder both become e.g. .../DSP_LM/110m).
        self.output_dir = os.path.join(self.output_dir, self.preset)


# ==========================================================================
# 5. DATA  --  schema-aware extraction, robust loading, sequence packing
# ==========================================================================


CHAT_TEMPLATE = "User: {prompt}\nAssistant:"  # response follows after this


def make_formatters():
    """Structured per-dataset formatters -> (is_instruction, prompt, response).

    QA/instruction datasets become User/Assistant turns; during training the
    prompt tokens are masked out of the loss so the model is only supervised to
    produce the response (standard SFT). Prose corpora (textbooks) are plain
    continuation with full supervision. Returns None when the schema doesn't
    match (used to detect provenance after interleave loses it).
    """

    def logic(item):  # natural_reasoning: question -> chain-of-thought response
        q = item.get("question", "")
        if not (isinstance(q, str) and q):
            return None
        resp = ""
        r = item.get("responses")
        if isinstance(r, list) and r and isinstance(r[0], dict):
            resp = r[0].get("response", "") or ""
        if not resp:  # fall back to the short reference answer
            resp = item.get("reference_answer", "") or ""
        return (True, q, resp) if isinstance(resp, str) and resp else None

    def qa_pair(item):  # orca-math / strix: question -> answer
        q, a = item.get("question", ""), item.get("answer", "")
        if isinstance(q, str) and q and isinstance(a, str) and a:
            return (True, q, a)
        return None

    def alpaca(item):  # alpaca-cleaned: instruction (+input) -> output
        instr, inp, out = (
            item.get("instruction", ""),
            item.get("input", ""),
            item.get("output", ""),
        )
        if not (isinstance(instr, str) and instr and isinstance(out, str) and out):
            return None
        prompt = f"{instr}\n{inp}" if isinstance(inp, str) and inp else instr
        return (True, prompt, out)

    def coedit(item):  # CoEdIT: src (instruction+text) -> tgt (corrected)
        s, t = item.get("src", ""), item.get("tgt", "")
        if isinstance(s, str) and s and isinstance(t, str) and t:
            return (True, s, t)
        return None

    def prose(item):  # open-web-math / science-wiki: plain continuation ('text')
        t = item.get("text", "")
        if isinstance(t, str) and t:
            return (False, "", t)
        # robustness: fall back to any other long string field
        for v in item.values():
            if isinstance(v, str) and len(v) > 200:
                return (False, "", v)
        return None

    # language/math/physics are large prose corpora; philosophy is Q&A.
    # qa_pair/alpaca are kept defined for a future SFT pass (orca-math, alpaca).
    # planning rows are already fully-rendered CoT text (see
    # BlocksworldPlanningStream below), so the plain prose formatter applies
    # unchanged.
    return {
        "grammar": coedit,
        "language": prose,
        "logic": logic,
        "math": prose,
        "physics": prose,
        "philosophy": qa_pair,
        "humanities": prose,
        "planning": prose,
        "qa": alpaca,
    }


# --------------------------------------------------------------------------
# Synthetic symbolic-planning domain: PDDL-Instruct-style chain-of-thought
# (Verma et al., "Teaching LLMs to Plan", arXiv:2509.13351). The paper
# fine-tunes an already-capable LLM with an external VAL verifier in the
# training loop, feeding back precondition/effect violations over several
# iterations -- that machinery needs a model that can already produce
# coherent multi-step text and a verifier binary, neither of which fits a
# from-scratch base-pretraining curriculum. What DOES transfer is the paper's
# core insight and its training DATA shape: verbose, explicit
# precondition-check / effect-application / state-tracking chain-of-thought
# over STRIPS-style planning problems, mixed with deliberately invalid plans
# that state exactly which precondition was violated (Phase 1 of the paper).
# That data is entirely procedurally generatable and self-verifying -- the
# same STRIPS engine that generates a problem also IS the ground-truth
# checker, so this needs no external VAL binary, no LLM feedback loop, and no
# download; like math/physics it "never runs out". It teaches the model
# exact multi-step state tracking, a skill that sits naturally between the
# 'logic' and 'math' curriculum domains.
def _bw_random_towers(rng, blocks):
    """Partition blocks into random stacks (bottom, ..., top) of size <= 3."""
    order = blocks[:]
    rng.shuffle(order)
    towers, i = [], 0
    while i < len(order):
        k = rng.randint(1, min(3, len(order) - i))
        towers.append(order[i : i + k])
        i += k
    return towers


def _bw_state_from_towers(towers):
    on, ontable, clear = {}, set(), set()
    for tower in towers:
        ontable.add(tower[0])
        for i in range(1, len(tower)):
            on[tower[i]] = tower[i - 1]
        clear.add(tower[-1])
    return {"on": on, "ontable": ontable, "clear": clear, "holding": None}


def _bw_state_preds(state):
    preds = [f"(ontable {b})" for b in sorted(state["ontable"])]
    preds += [f"(on {b} {base})" for b, base in sorted(state["on"].items())]
    preds += [f"(clear {b})" for b in sorted(state["clear"])]
    preds.append(f"(holding {state['holding']})" if state["holding"] else "(handempty)")
    return preds


def _bw_action_str(action):
    return "(" + " ".join(action) + ")"


def _bw_action_spec(state, action):
    """Preconditions (name, holds-in-state) plus the add/delete effect lists
    the action would apply IF every precondition holds -- standard STRIPS
    4-operator Blocksworld (pick-up, put-down, stack, unstack), same
    predicate set the paper uses.
    """
    kind = action[0]
    on, ontable, clear, holding = (
        state["on"],
        state["ontable"],
        state["clear"],
        state["holding"],
    )
    if kind == "pick-up":
        (x,) = action[1:]
        preconds = [
            (f"(clear {x})", x in clear),
            (f"(ontable {x})", x in ontable),
            ("(handempty)", holding is None),
        ]
        add, del_ = [f"(holding {x})"], [
            f"(ontable {x})",
            f"(clear {x})",
            "(handempty)",
        ]
    elif kind == "put-down":
        (x,) = action[1:]
        preconds = [(f"(holding {x})", holding == x)]
        add, del_ = [f"(ontable {x})", f"(clear {x})", "(handempty)"], [
            f"(holding {x})"
        ]
    elif kind == "stack":
        x, y = action[1:]
        preconds = [(f"(holding {x})", holding == x), (f"(clear {y})", y in clear)]
        add, del_ = [f"(on {x} {y})", f"(clear {x})", "(handempty)"], [
            f"(holding {x})",
            f"(clear {y})",
        ]
    elif kind == "unstack":
        x, y = action[1:]
        preconds = [
            (f"(on {x} {y})", on.get(x) == y),
            (f"(clear {x})", x in clear),
            ("(handempty)", holding is None),
        ]
        add, del_ = [f"(holding {x})", f"(clear {y})"], [
            f"(on {x} {y})",
            f"(clear {x})",
            "(handempty)",
        ]
    else:
        raise ValueError(kind)
    return preconds, add, del_


def _bw_apply(state, action):
    """Apply an already-checked-applicable action; returns the new state."""
    kind = action[0]
    on, ontable, clear = dict(state["on"]), set(state["ontable"]), set(state["clear"])
    holding = state["holding"]
    if kind == "pick-up":
        (x,) = action[1:]
        ontable.discard(x)
        clear.discard(x)
        holding = x
    elif kind == "put-down":
        (x,) = action[1:]
        ontable.add(x)
        clear.add(x)
        holding = None
    elif kind == "stack":
        x, y = action[1:]
        on[x] = y
        clear.discard(y)
        clear.add(x)
        holding = None
    elif kind == "unstack":
        x, y = action[1:]
        del on[x]
        clear.discard(x)
        clear.add(y)
        holding = x
    return {"on": on, "ontable": ontable, "clear": clear, "holding": holding}


def _bw_solve(init_towers, goal_towers):
    """Disassemble every initial tower onto the table, then rebuild each goal
    tower bottom-up. Always correct by construction (not necessarily
    optimal -- the paper targets satisficing plans too, see Sec. 5.1)."""
    plan = []
    for tower in init_towers:
        for idx in range(len(tower) - 1, 0, -1):
            plan.append(("unstack", tower[idx], tower[idx - 1]))
            plan.append(("put-down", tower[idx]))
    for tower in goal_towers:
        for idx in range(1, len(tower)):
            plan.append(("pick-up", tower[idx]))
            plan.append(("stack", tower[idx], tower[idx - 1]))
    return plan


def _bw_goal_preds(goal_towers):
    return [
        f"(on {tower[idx]} {tower[idx - 1]})"
        for tower in goal_towers
        for idx in range(1, len(tower))
    ]


def _bw_render_example(rng, n_blocks, invalid_frac=0.2):
    """One PDDL-Instruct-style CoT training example: header + step-by-step
    precondition/effect/state trace, ending either in a goal-achieved VALID
    plan or (invalid_frac of the time) a deliberately-omitted setup action
    whose paired follow-up then provably violates a precondition.
    """
    blocks = [chr(ord("a") + i) for i in range(n_blocks)]
    init_towers = _bw_random_towers(rng, blocks)
    goal_towers = _bw_random_towers(rng, blocks)
    plan = _bw_solve(init_towers, goal_towers)
    goal_preds = _bw_goal_preds(goal_towers)

    # Omitting a pick-up/unstack guarantees its paired stack/put-down fails
    # (holding x) next -- a clean, always-valid way to manufacture a plan
    # that is invalid for a known, checkable reason.
    skip_candidates = [
        i for i, a in enumerate(plan[:-1]) if a[0] in ("pick-up", "unstack")
    ]
    skip_idx = (
        rng.choice(skip_candidates)
        if skip_candidates and rng.random() < invalid_frac
        else None
    )

    state = _bw_state_from_towers(init_towers)
    lines = [
        "[BLOCKSWORLD PLANNING PROBLEM]",
        f"Objects: {' '.join(blocks)}",
        f"Initial state: {', '.join(_bw_state_preds(state))}",
        f"Goal: {', '.join(goal_preds)}",
        "",
        "[STEP BY STEP PLANNING]",
    ]

    step = 0
    for i, action in enumerate(plan):
        if i == skip_idx:
            continue  # deliberately omitted from the executed trace
        step += 1
        preconds, add, del_ = _bw_action_spec(state, action)
        applicable = all(ok for _, ok in preconds)
        lines.append(
            f"\n[Step {step}: State s{step - 1}  Action a{step}  State s{step}]"
        )
        lines.append(
            f"- Current state s{step - 1}: {', '.join(_bw_state_preds(state))}"
        )
        lines.append(f"- Proposed action a{step}: {_bw_action_str(action)}")
        lines.append("- Precondition check:")
        for name, ok in preconds:
            lines.append(f"  - {name}: {'TRUE' if ok else 'FALSE'} in s{step - 1}")
        if not applicable:
            failed = next(name for name, ok in preconds if not ok)
            lines.append(f"- VIOLATION: The precondition {failed} is not satisfied")
            lines.append("- Action is NOT APPLICABLE")
            lines.append(
                f"\n[PLAN VALIDITY] This plan is INVALID. Action {_bw_action_str(action)} at "
                f"step {step} cannot be applied because {failed} does not hold in s{step - 1}."
            )
            return "\n".join(lines)
        lines.append("- Action is APPLICABLE")
        state = _bw_apply(state, action)
        lines.append(
            f"- Effect application:\n  - Add: {', '.join(add)}\n  - Delete: {', '.join(del_)}"
        )
        lines.append(f"- Resulting state s{step}: {', '.join(_bw_state_preds(state))}")

    # No forced violation was hit -> the full plan ran; confirm the goal.
    cur_preds = set(_bw_state_preds(state))
    achieved = all(g in cur_preds for g in goal_preds)
    assert (
        achieved
    ), "solver-generated plan did not reach its own goal"  # self-verified, should never fire
    lines.append("\n[GOAL ACHIEVEMENT CHECK]")
    lines.append(f"Required: {', '.join(goal_preds)}")
    for g in goal_preds:
        lines.append(f"- {g}: TRUE in s{step}")
    lines.append("Goal is ACHIEVED.")
    executed = [a for i, a in enumerate(plan) if i != skip_idx]
    lines.append("\n[PLAN VALIDITY] This plan is VALID.")
    lines.append(f"[FINAL PLAN] {', '.join(_bw_action_str(a) for a in executed)}")
    return "\n".join(lines)


class BlocksworldPlanningStream:
    """Infinite synthetic Blocksworld symbolic-planning CoT stream.

    Each call advances a persistent RNG (``__iter__`` returns ``self``, so
    repeated ``iter()`` calls from WeightedMultiplex continue rather than
    replaying the same sequence) and yields ``{"text": ...}`` rows already
    formatted for the plain-prose formatter. See the module comment above
    ``_bw_random_towers`` for the rationale.
    """

    def __init__(self, seed=1, min_blocks=3, max_blocks=6, invalid_frac=0.2):
        self.rng = random.Random(seed)
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.invalid_frac = invalid_frac

    def __iter__(self):
        return self

    def __next__(self):
        n = self.rng.randint(self.min_blocks, self.max_blocks)
        return {"text": _bw_render_example(self.rng, n, self.invalid_frac)}


def encode_example(name, item, formatters, tokenizer, mask_prompt, chat_format):
    r"""Tokenise one (dataset-name, row) into (ids, loss_mask).

    loss_mask[i] == 1 -> token i is supervised; 0 -> ignored (prompt tokens).

    chat_format=False (BASE pretraining): everything is plain prose — Q&A rows
    are concatenated "prompt\\n\\nresponse". No User:/Assistant: scaffolding, so
    the model never learns to emit it or to regurgitate templated prompts.
    chat_format=True (SFT): wrap in the User/Assistant template and (optionally)
    mask the prompt so only the response is supervised.
    """
    r = formatters[name](item)
    if r is None:
        return [], []

    is_instruction, prompt, response = r
    eos = tokenizer.eos_token_id
    if is_instruction and chat_format:
        pre = tokenizer.encode(CHAT_TEMPLATE.format(prompt=prompt))
        ans = tokenizer.encode(f" {response}") + [eos]
        ids = pre + ans
        mask = ([0] * len(pre) + [1] * len(ans)) if mask_prompt else [1] * len(ids)
    elif is_instruction:
        # Base pretraining: plain-text concatenation, fully supervised.
        ids = tokenizer.encode(f"{prompt}\n\n{response}") + [eos]
        mask = [1] * len(ids)
    else:
        ids = tokenizer.encode(response) + [eos]
        mask = [1] * len(ids)
    return ids, mask


class WeightedMultiplex:
    """Weighted round-robin over several raw streaming iterators.

    Replaces datasets.interleave_datasets, which fails when sibling datasets
    have differently-typed columns of the same name (e.g. reasoning-core's
    ``prompt`` is Arrow large_string while Cosmopedia's is string). We sample
    in plain Python, so no cross-dataset schema alignment is attempted, and
    each yielded row keeps its dataset name for correct formatting.

    Exhausted sources are **cycled** (restarted from the top) so the training
    mix ratio stays stable for the entire substep. A warning is printed the
    first time each source restarts.
    """

    def __init__(self, iterables, weights, names, seed=3407):
        self._iterables = list(iterables)  # keep originals for recycling
        self.iters = [iter(it) for it in self._iterables]
        self.weights = list(weights)
        self.names = list(names)
        self._cycled: set[int] = set()  # indices that have been restarted
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iters:
            raise StopIteration
        i = self.rng.choices(range(len(self.iters)), weights=self.weights)[0]
        try:
            return self.names[i], next(self.iters[i])
        except StopIteration:
            # Cycle: restart from the beginning instead of dropping.
            if i not in self._cycled:
                print(
                    f"    [WeightedMultiplex] cycling exhausted source '{self.names[i]}'"
                )
                self._cycled.add(i)
            self.iters[i] = iter(self._iterables[i])
            return self.names[i], next(self.iters[i])


class PackedTokenStream:
    """Packs tokenised examples into (seq_len+1) blocks with an aligned loss mask.

    No padding waste; every block is full. Prompt tokens carry mask 0 (ignored
    by the loss), response / prose tokens carry mask 1. Examples are separated
    by EOS (added inside encode_example).
    """

    def __init__(
        self,
        multiplex,
        formatters,
        tokenizer,
        seq_len,
        mask_prompt=True,
        chat_format=False,
    ):
        self.mux = iter(multiplex)
        self.formatters = formatters
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_prompt = mask_prompt
        self.chat_format = chat_format
        self.ids_buf: list[int] = []
        self.mask_buf: list[int] = []

    def get_block(self, device):
        need = self.seq_len + 1
        while len(self.ids_buf) < need:
            try:
                name, item = next(self.mux)
            except StopIteration:
                if len(self.ids_buf) < need:
                    return None, None
                break
            ids, mask = encode_example(
                name,
                item,
                self.formatters,
                self.tokenizer,
                self.mask_prompt,
                self.chat_format,
            )
            if len(ids) > 5:
                self.ids_buf.extend(ids)
                self.mask_buf.extend(mask)
        ids = torch.tensor(self.ids_buf[:need], dtype=torch.long, device=device)
        mask = torch.tensor(self.mask_buf[:need], dtype=torch.long, device=device)
        self.ids_buf = self.ids_buf[need:]
        self.mask_buf = self.mask_buf[need:]
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].clone()
        y[mask[1:] == 0] = -100  # ignore prompt targets (cross_entropy default)
        return x, y.unsqueeze(0)


def get_packed_batch(streams, batch_size, device):
    """Assemble a batch by drawing packed blocks from the stream(s)."""
    xs, ys = [], []
    for _ in range(batch_size):
        stream = streams[len(xs) % len(streams)]
        x, y = stream.get_block(device)
        if x is None:
            break
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return torch.cat(xs, 0), torch.cat(ys, 0)


class BatchPrefetcher:
    """Produces batches on a background thread so CPU tokenisation/streaming
    overlaps GPU compute (the training loop was data-starved, pinned at the
    tokeniser's rate regardless of model or batch size). Batches are built on
    the CPU by the worker; the main loop moves them to the GPU (a tiny copy).
    """

    def __init__(self, streams, batch_size, depth=3):
        import queue
        import threading

        self.streams = streams
        self.batch_size = batch_size
        self._q = queue.Queue(maxsize=depth)
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            xy = get_packed_batch(self.streams, self.batch_size, "cpu")
            self._q.put(xy)
            if xy[0] is None:  # stream exhausted
                break

    def next(self, device):
        x, y = self._q.get()
        if x is None:
            return None, None
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def close(self):
        import queue

        self._stop = True
        try:  # unblock the worker if it's waiting on a full queue
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass


# ==========================================================================
# 6. CHECKPOINTING
# ==========================================================================


def save_checkpoint(path, model, optimizer, scheduler, step, completed_substeps, cfg):
    # torch.compile() wraps the module and prefixes state_dict keys with
    # "_orig_mod." -- always save/load the plain (uncompiled) module's keys so
    # checkpoints stay compatible with chat.py and with toggling
    # cfg.torch_compile across resumes.
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "completed_substeps": completed_substeps,
            "config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    getattr(model, "_orig_mod", model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("step", 0), ckpt.get("completed_substeps", 0)


# ==========================================================================
# 7. TRAINING
# ==========================================================================


def build_model_and_optim(cfg: Config, vocab_size: int, device: str):
    model = VectorizedDendriticLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        depth=cfg.depth,
        n_states=cfg.n_states,
        num_branches=cfg.num_branches,
        branch_dim=cfg.branch_dim,
        use_checkpoint=cfg.use_checkpoint,
        dendrite_variant=cfg.dendrite_variant,
    ).to(device)

    # Weight-decay only the 2-D projection/embedding matrices. Exclude biases,
    # LayerNorm gains (ndim < 2) and the SSM kernel poles (log_A_real, A_imag,
    # log_dt, C) + skip D — decaying resonator parameters is harmful.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or ".kernel." in name or name.endswith(".D"):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = None
    if cfg.optim_8bit and device == "cuda":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
            # Embedding (tied to lm_head) is unusually quantization-sensitive;
            # bitsandbytes' own docs recommend keeping its optimizer state in
            # fp32 rather than 8-bit.
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                model.embedding, "weight", {"optim_bits": 32}
            )
            print("Optimizer: bitsandbytes AdamW8bit (quantized optimizer state)")
        except ImportError:
            print(
                "optim_8bit=True but bitsandbytes is not installed "
                "(pip install bitsandbytes) -- falling back to plain AdamW."
            )
    if optimizer is None:
        optimizer = optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
    return model, optimizer


def fit_batch_size(model, cfg, vocab_size, device):
    """Probe one forward+backward at cfg.batch_size, halving on CUDA OOM until it
    fits. Modifies cfg.batch_size / cfg.grad_accum in place (effective batch kept
    roughly constant). Ends the OOM guessing game — it self-tunes to the card,
    the model size, and the gradient-checkpointing setting.
    """
    if device != "cuda":
        return
    model.train()
    b = cfg.batch_size
    while b >= 1:
        try:
            torch.cuda.empty_cache()
            x = torch.randint(0, vocab_size, (b, cfg.seq_len), device=device)
            y = torch.randint(0, vocab_size, (b, cfg.seq_len), device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(x, return_hidden=True)
                loss = fused_lm_loss(hidden, model.lm_head.weight, y, cfg.z_loss_weight)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del x, y, hidden, loss
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            b //= 2
            print(f"  batch probe OOM -> retrying at batch {b}")
    b = max(1, b)
    if b != cfg.batch_size:
        eff = cfg.batch_size * cfg.grad_accum
        cfg.batch_size = b
        cfg.grad_accum = max(1, eff // b)
        print(
            f"Auto-fit batch -> {b} x accum {cfg.grad_accum} (effective {b * cfg.grad_accum})"
        )
    else:
        print(f"Batch {b} fits.")


def make_scheduler(cfg: Config, optimizer, total_steps: int):
    """WSD: linear warmup -> stable plateau at full LR -> final decay to floor."""
    decay_steps = int(total_steps * cfg.decay_frac)
    stable_end = max(cfg.warmup_steps, total_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if step < stable_end:
            return 1.0  # full LR for every curriculum phase
        progress = (step - stable_end) / max(1, total_steps - stable_end)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_eval_blocks(
    eval_loaded, formatters, tokenizer, seq_len, blocks_per, chat_format=False
):
    """Pack a fixed set of held-out blocks (CPU tensors) **per dataset**.

    Returns ``dict[str, list[tuple[Tensor, Tensor]]]`` so callers can evaluate
    on individual domains or a filtered subset. Uses the same format as training
    (plain prose for the base run) so eval loss is comparable.
    """
    blocks_by_name: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, ds in eval_loaded.items():
        mux = WeightedMultiplex([ds], [1.0], [name], seed=0)
        stream = PackedTokenStream(
            mux,
            formatters,
            tokenizer,
            seq_len,
            mask_prompt=False,
            chat_format=chat_format,
        )
        ds_blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(blocks_per):
            x, y = stream.get_block("cpu")
            if x is None:
                break
            ds_blocks.append((x, y))
        if ds_blocks:
            blocks_by_name[name] = ds_blocks
    return blocks_by_name


def _chunk_ce_z(hidden_chunk, head_weight, target_chunk, z_weight):
    """Loss contributions for one chunk of tokens (checkpointed in backward)."""
    logits = (hidden_chunk @ head_weight.t()).float()  # (chunk, vocab)
    ce = F.cross_entropy(logits, target_chunk, ignore_index=-100, reduction="sum")
    if z_weight:
        z = torch.logsumexp(logits, dim=-1)  # (chunk,)
        zsum = (z * z).sum()
    else:
        zsum = logits.new_zeros(())
    return ce, zsum


def fused_lm_loss(hidden, head_weight, targets, z_weight=0.0, chunk=2048):
    """Cross-entropy (+ optional z-loss) computed from hidden states in chunks.

    The full (B, T, vocab) logits tensor is never materialised — it is the
    single largest allocation in the model. Instead each chunk of tokens is
    projected to logits, scored, and (via checkpointing) recomputed in the
    backward pass, so peak logit memory is one chunk, not the whole batch.
    """
    h = hidden.reshape(-1, hidden.size(-1))  # (N, d)
    t = targets.reshape(-1)  # (N,)
    n_valid = (t != -100).sum().clamp_min(1)
    ce_sum = h.new_zeros((), dtype=torch.float32)
    z_sum = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        hc, tc = h[i : i + chunk], t[i : i + chunk]
        ce, zs = checkpoint(
            _chunk_ce_z, hc, head_weight, tc, z_weight, use_reentrant=False
        )
        ce_sum = ce_sum + ce
        z_sum = z_sum + zs
    loss = ce_sum / n_valid
    if z_weight:
        loss = loss + z_weight * z_sum / h.size(0)
    return loss


@torch.no_grad()
def evaluate(model, blocks, vocab_size, device):
    """Mean cross-entropy over a flat list of held-out blocks."""
    if not blocks:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    for x, y in blocks:
        x, y = x.to(device), y.to(device)
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def evaluate_per_dataset(
    model, eval_blocks_by_name, vocab_size, device, active_datasets=None
):
    """Eval loss per dataset and an average over the active datasets only.

    ``active_datasets`` limits which datasets contribute to the reported
    average; if *None* all datasets are included.
    """
    per_ds: dict[str, float] = {}
    for name, blocks in eval_blocks_by_name.items():
        per_ds[name] = evaluate(model, blocks, vocab_size, device)

    if active_datasets is not None:
        active_vals = [per_ds[n] for n in active_datasets if n in per_ds]
    else:
        active_vals = list(per_ds.values())
    avg = sum(active_vals) / max(1, len(active_vals)) if active_vals else float("nan")
    return per_ds, avg


@torch.no_grad()
def ssm_diagnostics(model) -> dict:
    """Interpretability readout of the resonator poles (à la ResonatorLM's
    physics diagnostics): the timescales the SSM has learned to occupy.

    Per channel/state: effective per-step damping alpha = exp(log_A_real)*dt, so
    half-life = ln2 / alpha (in tokens); frequency = |A_imag| * dt (rad/step).
    A healthy model spreads half-lives across short and long timescales rather
    than collapsing to one regime.
    """
    half_lives, freqs = [], []
    for blk in model.blocks:
        k = blk.ssm.kernel
        dt = torch.exp(k.log_dt).unsqueeze(-1)  # (H, 1)
        alpha = (torch.exp(k.log_A_real) * dt).clamp_min(1e-8)  # (H, N/2)
        half_lives.append((math.log(2) / alpha).flatten().float())
        freqs.append((k.A_imag.abs() * dt).flatten().float())
    hl = torch.cat(half_lives)
    fr = torch.cat(freqs)
    qs = torch.quantile(hl, torch.tensor([0.05, 0.5, 0.95], device=hl.device))
    return {
        "channels": hl.numel(),
        "half_life_tokens": {
            "p5": round(qs[0].item(), 1),
            "median": round(qs[1].item(), 1),
            "p95": round(qs[2].item(), 1),
            "max": round(hl.max().item(), 1),
        },
        "freq_rad_per_step": {
            "min": round(fr.min().item(), 4),
            "max": round(fr.max().item(), 4),
        },
        "frac_longmemory_gt_1024tok": round((hl > 1024).float().mean().item(), 3),
    }


def print_ssm_diagnostics(model) -> None:
    d = ssm_diagnostics(model)
    hl = d["half_life_tokens"]
    print(
        f"SSM resonators: {d['channels']} modes | half-life tokens "
        f"p5={hl['p5']} median={hl['median']} p95={hl['p95']} max={hl['max']} | "
        f"{d['frac_longmemory_gt_1024tok']:.0%} long-memory (>1024 tok)"
    )


def dendrite_diagnostics(model) -> list[dict]:
    """Per-layer learned dendrite parameters, in block order."""
    return [
        blk.dendrite.diagnostics()
        for blk in model.blocks
        if hasattr(blk, "dendrite") and hasattr(blk.dendrite, "diagnostics")
    ]


def print_dendrite_diagnostics(model) -> None:
    """Report what each layer LEARNED about its own dendritic mechanism.

    This is the measurement the whole dendrite ablation exists to produce.
    Read it as follows:

    ``gate_k`` — the NMDA steepness. Initialised at 1.0. Trending up means the
    layer is sharpening the supralinear transition, i.e. the mechanism the
    human/rat literature turns on is being actively used. Collapsing toward 0
    linearises the gate away and says the mechanism is inert for this task.

    ``lambda`` — the electrotonic length constant in branch-tree units
    (distances are 0/2/4/6 for 8 branches). Initialised at 4.0. Falling means
    the layer is choosing compartmentalisation; growing large means it is
    reverting to a dense FFN and rejecting the premise. A layer is allowed to
    disagree with the hypothesis, and reporting that honestly is the point.
    """
    rows = dendrite_diagnostics(model)
    if not rows or rows[0].get("variant") == "baseline":
        variant = rows[0].get("variant") if rows else "n/a"
        print(f"Dendrites: variant={variant!r} — no learned gate/decay to report.")
        return
    print(f"Dendrites: variant={rows[0]['variant']!r}, {len(rows)} layers")
    header = f"  {'layer':>5} {'gate_k':>9} {'gate_k_max':>11} {'theta':>9}"
    has_lambda = "lambda" in rows[0]
    has_junction = "junction_k_mean" in rows[0]
    if has_lambda:
        header += f" {'lambda':>9}"
    if has_junction:
        header += f" {'junc_k':>9}"
    print(header)
    for i, r in enumerate(rows):
        line = (
            f"  {i:>5} {r['gate_k_mean']:>9.3f} {r['gate_k_max']:>11.3f} "
            f"{r['gate_theta_mean']:>9.3f}"
        )
        if has_lambda:
            line += f" {r['lambda']:>9.3f}"
        if has_junction:
            line += f" {r['junction_k_mean']:>9.3f}"
        print(line)
    k_all = [r["gate_k_mean"] for r in rows]
    print(
        f"  gate_k: mean={sum(k_all)/len(k_all):.3f} "
        f"min={min(k_all):.3f} max={max(k_all):.3f} (init 1.0)"
    )
    if has_lambda:
        l_all = [r["lambda"] for r in rows]
        print(
            f"  lambda: mean={sum(l_all)/len(l_all):.3f} "
            f"min={min(l_all):.3f} max={max(l_all):.3f} (init 4.0; "
            f"small=compartmentalised, large=dense)"
        )


def smoke_test(cfg: Config, device: str) -> None:
    """Validate the whole train step on a LEARNABLE synthetic task.

    The task is next = (token + 1) mod vocab: a deterministic pattern the model
    must actually learn, so a healthy run shows loss falling well below the
    random-guess baseline ln(vocab). (If x and y are independent random noise,
    loss can never drop below ln(vocab) — that tests plumbing, not learning.)
    """
    import math as _math

    print("=== SMOKE TEST (learnable synthetic task, tiny model) ===")
    cfg.d_model, cfg.depth, cfg.seq_len = 64, 2, 64
    cfg.batch_size, cfg.grad_accum, cfg.n_states = 8, 1, 16
    cfg.num_branches, cfg.branch_dim = 4, 32
    cfg.warmup_steps, cfg.lr = 5, 3e-3
    # Exercise whichever dendrite mechanism is under test:
    #   DSP_SMOKE=1 DSP_DENDRITE=tree uv run python colab_trainable_dendritic_lm.py
    cfg.dendrite_variant = os.environ.get("DSP_DENDRITE", cfg.dendrite_variant)
    print(f"dendrite_variant: {cfg.dendrite_variant}")
    vocab_size = 256
    n_steps = 60
    baseline = _math.log(vocab_size)

    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    scheduler = make_scheduler(cfg, optimizer, total_steps=n_steps)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    print(
        f"random-guess loss (ln vocab) = {baseline:.3f}; a healthy run drops below it"
    )

    first_loss = None
    last_loss = None
    for step in range(n_steps):
        x = torch.randint(0, vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        y = (x + 1) % vocab_size
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), "NaN/Inf loss"
        first_loss = first_loss or loss.item()
        last_loss = loss.item()
        if step % 10 == 0 or step == n_steps - 1:
            print(
                f"  step {step:02d} | loss {loss.item():.4f} | ppl {_math.exp(min(20, loss.item())):.1f}"
            )

    gen = model.generate(x[:, :4], max_new_tokens=8, top_k=10)
    assert gen.shape == (cfg.batch_size, 12)
    print(f"\ninitial loss {first_loss:.3f} -> final loss {last_loss:.3f}")
    assert (
        first_loss < baseline * 2
    ), f"initial loss {first_loss:.1f} >> ln(vocab) {baseline:.1f} — logits mis-scaled (init bug)"
    assert last_loss < first_loss * 0.8, "loss did not fall — model is not learning"
    print(
        f"generate OK -> {tuple(gen.shape)}\n=== SMOKE TEST PASSED (model learns) ==="
    )


# ==========================================================================
# 8. CHECKPOINT SYNC  --  Hugging Face Hub (Colab <-> local machine)
# ==========================================================================


def _is_colab() -> bool:
    """Detect whether we're running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def hf_push_checkpoint(hf_repo: str, output_dir: str, subdir: str = "") -> None:
    """Push the checkpoint directory to a private Hugging Face Hub repo.

    ``subdir`` (the model-size preset) keeps each size in its own folder in the
    repo so sizes never overwrite each other. Requires ``huggingface-cli login``
    or an ``HF_TOKEN`` env var.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=hf_repo, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo,
        folder_path=output_dir,
        path_in_repo=subdir or ".",
        commit_message=f"checkpoint update ({subdir or 'root'})",
    )
    print(f"  Pushed checkpoint to https://huggingface.co/{hf_repo}/{subdir}")


def hf_pull_checkpoint(hf_repo: str, output_dir: str, subdir: str = "") -> bool:
    """Download this size's checkpoints from the Hub into output_dir.

    Returns True if the pull succeeded. Only the ``subdir`` (preset) folder is
    fetched, and it lands directly in output_dir.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        # output_dir ends in the preset; download repo/<preset>/* into its parent
        # so files land back in output_dir.
        parent = os.path.dirname(output_dir.rstrip("/")) or "."
        snapshot_download(
            repo_id=hf_repo,
            local_dir=parent if subdir else output_dir,
            allow_patterns=[f"{subdir}/*"] if subdir else None,
            local_dir_use_symlinks=False,
        )
        print(f"  Pulled checkpoint from https://huggingface.co/{hf_repo}/{subdir}")
        return True
    except RepositoryNotFoundError:
        print(f"  No remote checkpoint found at {hf_repo}")
        return False
    except Exception as exc:
        print(f"  Could not pull checkpoint: {type(exc).__name__}: {exc}")
        return False


def hf_clean(hf_repo: str, output_dir: str, subdir: str = "") -> None:
    """Delete this size's local checkpoints and its remote Hub folder.

    Only the current preset is removed; other sizes in the repo are left alone.
    """
    import shutil

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"  Deleted local checkpoints: {output_dir}")
    else:
        print("  No local checkpoints to delete.")

    if hf_repo and subdir:
        from huggingface_hub import HfApi

        try:
            HfApi().delete_folder(path_in_repo=subdir, repo_id=hf_repo)
            print(f"  Deleted remote folder: {hf_repo}/{subdir}")
        except Exception as exc:
            print(f"  Could not delete remote folder ({type(exc).__name__}: {exc})")


def main(
    resume_override: bool | None = None,
    continue_stage: bool = False,
    preset: str | None = None,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = Config(preset=preset) if preset else Config()
    print(f"Preset: {cfg.preset} ({cfg.d_model}d x {cfg.depth}L) -> {cfg.output_dir}")
    if resume_override is not None:
        cfg.resume = resume_override  # CLI 'resume' / 'overwrite' switch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Preset batch sizes are tuned for an 80GB A100. Colab often hands out a
    # 40GB A100, where the fp32 logits in cross_entropy (batch*seq*vocab) OOM.
    # Auto-scale the microbatch to the detected VRAM, keeping effective batch
    # constant (raise grad_accum). Override cfg.batch_size manually to opt out.
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_gb < 70:
            eff = cfg.batch_size * cfg.grad_accum
            cfg.batch_size = max(2, int(cfg.batch_size * total_gb / 80))
            cfg.grad_accum = max(1, eff // cfg.batch_size)
            print(
                f"GPU {total_gb:.0f}GB (<80GB) -> scaled to batch {cfg.batch_size} "
                f"x accum {cfg.grad_accum} (effective {cfg.batch_size * cfg.grad_accum})"
            )
    if continue_stage:
        mode = "CONTINUE (load weights, fresh schedule, new data stage)"
    elif cfg.resume:
        mode = "RESUME from latest checkpoint"
    else:
        mode = "OVERWRITE (fresh start)"
    print(f"Mode: {mode}")

    if os.environ.get("DSP_SMOKE") == "1":
        smoke_test(cfg, device)
        return

    # On Colab with hf_repo configured, pull latest checkpoint for resume/continue.
    if _is_colab() and cfg.hf_repo and (cfg.resume or continue_stage):
        print("Checking HF Hub for existing checkpoint...")
        hf_pull_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)

    print(f"Checkpoint directory: {cfg.output_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # GPT-2's tokenizer carries a legacy 1024 cap (its positional-embedding
    # limit). This model has NO such limit — token embeddings + SSM handle any
    # length — so raise it to silence a spurious "sequence too long" warning.
    tokenizer.model_max_length = int(1e12)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size} tokens (GPT-2 BPE)")

    model, optimizer = build_model_and_optim(cfg, vocab_size, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print_ssm_diagnostics(model)  # timescale coverage at init
    fit_batch_size(
        model, cfg, vocab_size, device
    )  # probe VRAM; never OOM on batch again

    # Robust dataset loading: drop anything that fails. Each dataset is split
    # into a held-out eval set (first eval_docs rows) and a train set (the rest)
    # so eval never overlaps training.
    formatters = make_formatters()
    loaded, eval_loaded = {}, {}
    for name, (repo, cfg_name) in cfg.repos.items():
        try:
            base = load_dataset(repo, cfg_name, split="train", streaming=True)
            eval_loaded[name] = base.take(cfg.eval_docs)
            loaded[name] = base.skip(cfg.eval_docs)
            tag = f"{repo}:{cfg_name}" if cfg_name else repo
            print(f"  loaded {name:<10} <- {tag}")
        except Exception as exc:  # gated / offline / renamed
            print(f"  SKIP  {name:<10} <- {repo}  ({type(exc).__name__}: {exc})")

    # Synthetic symbolic-planning domain (PDDL-Instruct-style CoT, see the
    # comment above BlocksworldPlanningStream): procedurally generated and
    # self-verified by its own STRIPS engine, so it needs no download and
    # never runs out. Eval uses a disjoint seed so held-out problems aren't
    # drawn from the same RNG sequence as training.
    loaded["planning"] = BlocksworldPlanningStream(seed=1)
    eval_loaded["planning"] = list(
        itertools.islice(BlocksworldPlanningStream(seed=999_983), cfg.eval_docs)
    )
    print("  loaded planning   <- synthetic Blocksworld STRIPS CoT (procedural)")

    print("Building held-out eval blocks...")
    eval_blocks_by_name = build_eval_blocks(
        eval_loaded,
        formatters,
        tokenizer,
        cfg.seq_len,
        cfg.eval_blocks_per,
        cfg.chat_format,
    )
    total_blocks = sum(len(v) for v in eval_blocks_by_name.values())
    print(f"  {total_blocks} eval blocks held out across {list(eval_blocks_by_name)}")

    # Bloom-style progression: an explicit-language foundation (grammar drills +
    # simple correct stories + general prose) BEFORE the reasoning phases, then
    # the logic->math->physics->philosophy curriculum. 'language' (fineweb-edu)
    # stays on as a fluency backbone throughout. A phase may set "steps" to run
    # shorter than the global default (Phase 0 is a brief primer so the small
    # grammar sets don't over-repeat). Weights are auto-renormalised.
    phases = [
        {
            "name": "Phase_0_Language_Rules",
            "desc": "Explicit grammar correction + general prose",
            "datasets": ["grammar", "language"],
            "mixtures": [[0.20, 0.80]],
            # Short primer. CoEdIT is tiny (~4M tok); at 0.2 weight this keeps it
            # near ~4 epochs (the data-constrained sweet spot) instead of the
            # ~20 epochs it was hitting at 0.4 weight x 800 steps x batch 128.
            "steps": 300,
        },
        {
            "name": "Phase_1_Foundation",
            "desc": "Language grammar + foundational logic + symbolic planning",
            # 'planning' (synthetic Blocksworld CoT, see BlocksworldPlanningStream)
            # sits alongside 'logic' as another formal, exactly-checkable
            # reasoning source -- see the PDDL-Instruct discussion above.
            "datasets": ["language", "logic", "planning"],
            "mixtures": [[0.35, 0.50, 0.15]],
        },
        {
            "name": "Phase_2_Math_Introduction",
            "desc": "Language, logic, symbolic planning and math",
            "datasets": ["language", "logic", "math", "planning"],
            "mixtures": [[0.20, 0.20, 0.45, 0.15], [0.20, 0.30, 0.35, 0.15]],
        },
        {
            "name": "Phase_3_Physics_Application",
            "desc": "Language, logic, math, symbolic planning and physics",
            "datasets": ["language", "logic", "math", "physics", "planning"],
            "mixtures": [
                [0.20, 0.10, 0.15, 0.45, 0.10],
                [0.20, 0.10, 0.20, 0.40, 0.10],
                [0.20, 0.15, 0.25, 0.30, 0.10],
            ],
        },
        {
            "name": "Phase_4_Integration",
            "desc": "Language, reasoning, science, philosophy & humanities",
            "datasets": [
                "language",
                "logic",
                "math",
                "physics",
                "philosophy",
                "humanities",
            ],
            "mixtures": [[0.20, 0.10, 0.10, 0.15, 0.20, 0.25]],
        },
    ]

    total_substeps = sum(len(p["mixtures"]) for p in phases)
    total_microsteps = sum(
        len(p["mixtures"]) * p.get("steps", cfg.steps_per_substep) for p in phases
    )
    total_steps = total_microsteps // cfg.grad_accum
    scheduler = make_scheduler(cfg, optimizer, total_steps)

    os.makedirs(cfg.output_dir, exist_ok=True)
    latest = os.path.join(cfg.output_dir, "latest.pt")
    global_step = 0
    completed_substeps = 0
    if continue_stage:
        # Continued pretraining: load only the WEIGHTS from the prior run and
        # train a new stage (new data mix) from scratch — fresh optimizer and
        # LR schedule, no substep skipping. This is how you add datasets to an
        # already-trained model without a full restart.
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location=device)
            getattr(model, "_orig_mod", model).load_state_dict(
                ckpt["model"]
            )  # architecture must match
            print(
                f"CONTINUE: loaded weights from {latest} (step {ckpt.get('step', '?')}); "
                "starting a fresh stage on the current data mix."
            )
        else:
            print(
                f"CONTINUE requested but no checkpoint at {latest} — training from scratch."
            )
    elif cfg.resume and os.path.exists(latest):
        global_step, completed_substeps = load_checkpoint(
            latest, model, optimizer, scheduler, device
        )
        print(
            f"Resumed from {latest}: step {global_step}, {completed_substeps} substeps done"
        )

    if cfg.torch_compile and device == "cuda":
        # See the long comment on Config.torch_compile: this is opt-in, not
        # the default, because it has crashed on at least one Colab
        # torch/Inductor build even after walling off the complex-valued SSM
        # math from dynamo. If this crashes for you with an InductorError
        # mentioning 'complex', set cfg.torch_compile = False.
        print("Compiling model with torch.compile() (cfg.torch_compile=True)...")
        model = torch.compile(model)

    print(
        f"\nStarting curriculum training ({total_substeps} total substeps, {total_steps} total steps)..."
    )
    substep_global = 0  # flat index across all phases, for resume skip-ahead
    tokens_seen = 0

    def sample(seed_text="The physical principles governing the universe state that"):
        ids = tokenizer.encode(seed_text, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=40, temperature=0.8, top_k=50)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    for phase in phases:
        # Keep only datasets that actually loaded; renormalise the mixture.
        avail = [d for d in phase["datasets"] if d in loaded]
        if not avail:
            print(f"Skipping {phase['name']} — no datasets available.")
            continue
        header_shown = False

        for substep_idx, probs in enumerate(phase["mixtures"]):
            # Skip substeps already finished in a previous run.
            if substep_global < completed_substeps:
                substep_global += 1
                continue
            if not header_shown:
                print(f"\n{'=' * 60}\n{phase['name']}\n{phase['desc']}\n{'=' * 60}")
                header_shown = True

            kept = [
                (d, p)
                for d, p in zip(phase["datasets"], probs, strict=False)
                if d in loaded
            ]
            names = [d for d, _ in kept]
            weights = [p for _, p in kept]
            weights = [w / sum(weights) for w in weights]  # renormalise
            print(
                f"\n  Substep {substep_idx + 1}/{len(phase['mixtures'])} - "
                f"{dict(zip(names, [round(w, 3) for w in weights], strict=False))}"
            )

            # Weighted multiplex instead of interleave_datasets (which can't
            # align differently-typed columns across these datasets).
            # NOTE: streaming iterators restart from the top on resume (their
            # position isn't checkpointed); with shuffled multi-GB pools this is
            # acceptable for a research run.
            mux = WeightedMultiplex(
                [loaded[n] for n in names], weights, names, seed=3407 + substep_idx
            )
            stream = PackedTokenStream(
                mux,
                formatters,
                tokenizer,
                cfg.seq_len,
                cfg.mask_prompt_loss,
                cfg.chat_format,
            )
            streams = [stream]  # single packed stream; batch draws multiple blocks
            prefetcher = BatchPrefetcher(
                streams, cfg.batch_size
            )  # overlap data + compute

            phase_steps = phase.get("steps", cfg.steps_per_substep)
            optimizer.zero_grad(set_to_none=True)
            ema = None
            t_log, tok_log = time.time(), tokens_seen  # throughput window
            for step in range(phase_steps):
                model.train()
                x, y = prefetcher.next(device)
                if x is None:
                    print("  Stream exhausted early.")
                    break

                # Guard: if prompt-masking left every target ignored, cross
                # entropy would be NaN — skip this (degenerate) batch.
                if (y != -100).sum() == 0:
                    continue

                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        hidden = model(x, return_hidden=True)
                        loss = fused_lm_loss(
                            hidden, model.lm_head.weight, y, cfg.z_loss_weight
                        )
                else:
                    hidden = model(x, return_hidden=True)
                    loss = fused_lm_loss(
                        hidden, model.lm_head.weight, y, cfg.z_loss_weight
                    )

                (loss / cfg.grad_accum).backward()
                tokens_seen += x.numel()
                lv = loss.item()
                ema = lv if ema is None else 0.95 * ema + 0.05 * lv

                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if cfg.save_every and global_step % cfg.save_every == 0:
                        save_checkpoint(
                            latest,
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            substep_global,
                            cfg,
                        )
                    # Mid-phase push to HF so a Colab disconnect during a long
                    # phase doesn't lose hours (not only at substep boundaries).
                    if (
                        cfg.hf_repo
                        and cfg.hf_push_every
                        and global_step % cfg.hf_push_every == 0
                    ):
                        try:
                            hf_push_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
                        except Exception as exc:
                            print(f"  HF push failed (non-fatal): {exc}")

                if step % cfg.log_every == 0 or step == phase_steps - 1:
                    dt = max(1e-6, time.time() - t_log)
                    tps = (tokens_seen - tok_log) / dt  # tokens/sec since last log
                    t_log, tok_log = time.time(), tokens_seen
                    print(
                        f"  [{phase['name']} | sub {substep_idx + 1}] step {step:04d} "
                        f"| loss {lv:.4f} (ema {ema:.4f}) | ppl {math.exp(min(20, ema)):8.1f} "
                        f"| lr {scheduler.get_last_lr()[0]:.2e} | {tokens_seen / 1e6:.1f}M tok "
                        f"| {tps / 1e3:.1f}K tok/s"
                    )

            prefetcher.close()  # stop the background worker for this substep

            # Flush trailing accumulated gradients only if the last
            # accumulation cycle was incomplete (avoids a spurious optimizer
            # step with stale/partial grads that corrupts the checkpoint).
            if step % cfg.grad_accum != cfg.grad_accum - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            substep_global += 1  # this substep is now complete

            out_dir = os.path.join(
                cfg.output_dir, phase["name"], f"substep_{substep_idx + 1}"
            )
            os.makedirs(out_dir, exist_ok=True)
            save_checkpoint(
                os.path.join(out_dir, "checkpoint.pt"),
                model,
                optimizer,
                scheduler,
                global_step,
                substep_global,
                cfg,
            )
            save_checkpoint(
                latest, model, optimizer, scheduler, global_step, substep_global, cfg
            )
            tokenizer.save_pretrained(out_dir)

            # Per-dataset eval: report each domain + active-only average.
            active = [d for d in phase["datasets"] if d in loaded]
            per_ds, ev = evaluate_per_dataset(
                model, eval_blocks_by_name, vocab_size, device, active
            )
            print(f"  Saved checkpoint -> {out_dir}")
            parts = " | ".join(
                f"{n} {per_ds[n]:.2f}" for n in sorted(per_ds) if n in per_ds
            )
            print(f"  eval per-dataset: {parts}")
            print(
                f"  eval (active avg) loss {ev:.4f} | ppl {math.exp(min(20, ev)):.1f}"
            )
            print("  ", end="")
            print_ssm_diagnostics(model)  # how the resonators have evolved
            print(f"  sample: {sample()[:200]!r}")

            # Push to HF Hub so checkpoints survive Colab restarts.
            if cfg.hf_repo:
                try:
                    hf_push_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
                except Exception as exc:
                    print(f"  HF push failed (non-fatal): {exc}")

    print("\nTraining complete.")

    print("\n--- Generation test ---")
    seed_text = "The physical principles governing the universe state that"
    seed_idx = tokenizer.encode(seed_text, return_tensors="pt").to(device)
    generated = model.generate(seed_idx, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f"Seed: {seed_text!r}")
    print("Generated:\n" + tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    import sys

    # Colab/Jupyter injects arguments like `-f /root/.../kernel.json`. Parse only
    # the tokens we recognise: a command word and/or a size preset, in any order.
    # e.g.  `... 110m`  `... overwrite 110m`  `... 1b continue`  `... clean 500m`
    args = [a.lower() for a in sys.argv[1:]]
    preset = next((a for a in args if a in MODEL_PRESETS), None)
    commands = {
        "pull",
        "push",
        "clean",
        "overwrite",
        "fresh",
        "restart",
        "resume",
        "continue",
        "extend",
        "diagnose",
    }
    cmd = next((a for a in args if a in commands), "")

    if cmd == "diagnose":
        # Print the SSM resonator timescale readout for a size (loads its
        # checkpoint weights if present; otherwise shows the init spread).
        cfg = Config(preset=preset) if preset else Config()
        m = VectorizedDendriticLM(
            vocab_size=50257,
            d_model=cfg.d_model,
            depth=cfg.depth,
            n_states=cfg.n_states,
            num_branches=cfg.num_branches,
            branch_dim=cfg.branch_dim,
            use_checkpoint=False,
            dendrite_variant=cfg.dendrite_variant,
        )
        latest = os.path.join(cfg.output_dir, "latest.pt")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location="cpu")
            # A checkpoint trained under a different dendrite variant has a
            # different parameter set -- rebuild to match it rather than
            # failing on a state_dict mismatch.
            saved_variant = ckpt.get("config", {}).get(
                "dendrite_variant", cfg.dendrite_variant
            )
            if saved_variant != cfg.dendrite_variant:
                print(
                    f"[{cfg.preset}] checkpoint was trained with "
                    f"dendrite_variant={saved_variant!r}; rebuilding to match."
                )
                m = VectorizedDendriticLM(
                    vocab_size=50257,
                    d_model=cfg.d_model,
                    depth=cfg.depth,
                    n_states=cfg.n_states,
                    num_branches=cfg.num_branches,
                    branch_dim=cfg.branch_dim,
                    use_checkpoint=False,
                    dendrite_variant=saved_variant,
                )
            m.load_state_dict(ckpt["model"])
            print(f"[{cfg.preset}] diagnostics from {latest}:")
        else:
            print(f"[{cfg.preset}] no checkpoint; diagnostics at init:")
        print_ssm_diagnostics(m)
        print_dendrite_diagnostics(m)
        sys.exit(0)

    if cmd in ["pull", "push", "clean"]:
        cfg = Config(preset=preset) if preset else Config()
        if cmd == "pull":
            if not cfg.hf_repo:
                print("Error: set hf_repo in Config first.")
                sys.exit(1)
            hf_pull_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
        elif cmd == "clean":
            # Wipe THIS size's local checkpoints + its remote Hub folder.
            hf_clean(cfg.hf_repo, cfg.output_dir, cfg.preset)
        elif cmd == "push":
            if not cfg.hf_repo:
                print("Error: set hf_repo in Config first.")
                sys.exit(1)
            hf_push_checkpoint(cfg.hf_repo, cfg.output_dir, cfg.preset)
    elif cmd in ["overwrite", "fresh", "restart"]:
        main(resume_override=False, preset=preset)  # fresh run
    elif cmd == "resume":
        main(resume_override=True, preset=preset)  # force-resume
    elif cmd in ["continue", "extend"]:
        # Continued pretraining: keep the trained WEIGHTS, run a new stage on the
        # current data mix with a fresh schedule. Architecture must match.
        main(continue_stage=True, preset=preset)
    else:
        main(preset=preset)  # default: resume if a checkpoint exists
