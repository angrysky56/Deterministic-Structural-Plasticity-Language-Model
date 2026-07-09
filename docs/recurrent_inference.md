# Recurrent Inference — Design Plan

Status: **planned, not yet implemented.** This is a design note, not a spec of
existing code. Generation today uses the FFT/parallel path, which is correct and
unbounded but recomputes the whole prefix on every step.

## Why

The temporal mixer is a diagonal state-space model (`ResonatorSSM`). An SSM has
two mathematically equivalent execution modes:

- **Convolutional (what we use now).** Materialise the kernel `K` of length `L`
  and convolve the whole sequence with an FFT. Great for *training* (fully
  parallel over time). For *autoregressive generation* it is wasteful: each new
  token reprocesses the entire prefix, so producing `T` tokens costs
  `O(T · L · log L)` and memory grows with the prefix.
- **Recurrent.** Carry a hidden state `h ∈ ℂ^{H×N}` and advance one step per
  token: `O(1)` work and `O(H·N)` memory per step, *independent of sequence
  length*. This is the mode that makes "no context limit" cheap at inference.

For chat-length outputs the FFT path is fine. Recurrent inference matters for
long generations and for a responsive, low-latency CLI/server.

## The math (must match the FFT path exactly)

`ResonatorSSMKernel` already defines, per channel `h` and state `n`:

- discrete pole `ā = exp(Δ · A)`, where `A = -exp(log_A_real) + i·A_imag`
- readout `C̃ = C · (exp(Δ·A) − 1) / A`  (the `B≈1`, ZOH-style discretisation)

The convolutional kernel is `K[h, ℓ] = 2·Re( Σ_n C̃[h,n] · ā[h,n]^ℓ )`.

The **equivalent recurrence** for input `u_t` (the post-norm channel value) is:

```
h_t[h,n] = ā[h,n] · h_{t-1}[h,n] + u_t[h]          # state update (B = 1)
y_t[h]   = 2·Re( Σ_n C̃[h,n] · h_t[h,n] ) + D[h]·u_t[h]   # readout + skip
```

with `h_0 = 0`. Expanding the recurrence reproduces the convolution
`y_t = Σ_{ℓ≥0} K[·,ℓ]·u_{t-ℓ} + D·u_t` term for term — so the two paths are
identical up to floating-point error. Keep the state and kernel math in
`float32` (complex ops are unsupported in bf16), exactly as the FFT path does.

## Proposed API

Add to `ResonatorSSM` (non-breaking; training path untouched):

- `init_state(batch, device) -> Tensor`  # zeros, shape (B, H, N) complex64
- `step(u_t, state) -> (y_t, state)`      # one timestep; u_t: (B, H)

Add to `VectorizedDendriticLM`:

- `generate_recurrent(start_tokens, max_new_tokens, temperature, top_k, stop_ids)`
  1. **Prefill:** run the prompt once (FFT path) but also fold it into each
     block's SSM state (either by running `step` over the prompt, or by a
     chunked scan). Simplest first version: run `step` token-by-token over the
     prompt to build state, skipping the FFT entirely.
  2. **Decode:** for each new token, embed it, run each block's `step`, project
     to logits, sample, feed back. Per block we keep one `(B,H,N)` state; the
     dendrite (channel mixer) is already stateless/per-token.

The LayerNorms and the dendritic MLP need no state — only the SSM does — so the
per-layer state is just the SSM recurrence.

## Validation (do this before trusting it)

1. **Numerical parity.** For a trained checkpoint and a fixed prompt, assert
   `logits_recurrent ≈ logits_fft` at every position to ~1e-3 (bf16) / ~1e-5
   (fp32). This is the acceptance test; write it before wiring it into `chat.py`.
2. **Determinism.** Same seed → same tokens from both paths under greedy
   decoding (`temperature→0`).
3. **Throughput.** Benchmark tokens/sec vs the FFT path at prefix lengths
   256/1k/4k/16k; the recurrent path should be flat while the FFT path grows.

## Milestones / effort

- M1 — `step()` + `init_state()` on `ResonatorSSM`, plus a parity unit test
  against the FFT path on random weights. (~half a day)
- M2 — `generate_recurrent()` with token-by-token prefill; wire an opt-in
  `--recurrent` flag in `chat.py`. (~half a day)
- M3 — chunked/parallel prefill (FFT for the prompt, recurrent for decode) so
  long prompts prefill fast. (optimization; later)

## Why it's deferred

Recurrent inference is a speed/memory optimization, not a correctness fix — the
FFT path already generates correct, unbounded text. It's best built *after* a
trained checkpoint exists, because the parity test in step 1 needs real weights
to be meaningful; validating against random weights catches shape bugs but not
subtle discretisation mismatches.
