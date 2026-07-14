"""DSP-LM autoresearch notebook -- real-scale architecture research on an A100.

Source of truth for the Colab notebook; regenerate after editing with:
    jupytext --to notebook research_harness/autoresearch_notebook.py

Do NOT edit the .ipynb by hand -- it's generated from this file (same
convention as colab_trainable_dendritic_lm.ipynb). After regenerating, the
first two cells (`!nvidia-smi` and `!pip install ...`) need to be re-inserted
manually if jupytext strips them -- see the notebook-build note at the bottom
of this file.

WHY THIS EXISTS (read this before changing the budget numbers)
----------------------------------------------------------------
The local research_harness/{prepare_data,train_harness}.py pair borrows
Karpathy's `autoresearch` methodology: a *fixed* 5-minute wall-clock training
budget. That number was tuned for fast iteration on an H100 -- it is not a
property of DSP-LM, and it is not the hardware DSP-LM actually trains on.
Locally, on an RTX 3060, keeping runs short is still a real constraint (slow
card, broken fan), so that harness stays as-is for quick smoke tests. But
this notebook runs on an A100 (Colab), which is DSP-LM's real target hardware
per the project's own README/CLAUDE.md -- so the rules here are deliberately
different:

  - No fixed 5-minute wall clock. The stopping condition is a TOKEN budget
    (a meaningful chunk of real training, not a toy number), with a generous
    wall-clock safety net just so a Colab session can't run away unbounded.
  - Model size, sequence length and batch size default to something that
    actually uses an A100's memory, not whatever fit in a 3060's 12GB.
  - This notebook owns its OWN copy of the DendriticResonatorBlock /
    ResonatorSSM / DendriticMLP classes (inlined below), deliberately
    duplicated from colab_trainable_dendritic_lm.py rather than imported.
    That file is DSP-LM's production training script -- the source of truth
    for real runs -- and should stay stable. This notebook is the free
    scratchpad for architecture experiments (same idea as autoresearch's
    prepare.py-is-fixed / train.py-is-free split). If an experiment here
    wins, port the change back into colab_trainable_dendritic_lm.py by hand;
    don't let this notebook silently diverge into being "the" model.

Uses the same reference dataset/tokenizer/val_bpb methodology as the local
harness (Karpathy's climbmix-400b-shuffle + a from-scratch BPE tokenizer),
so results are still vocab-size-independent and comparable to the local
3060 runs in shape, even though the budget philosophy is different.
"""

import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = torch.cuda.is_available()
except ImportError:
    _HAS_TRITON = False

if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _dynamo_disable = torch.compiler.disable
else:

    def _dynamo_disable(fn):
        return fn


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==========================================================================
# Data: same climbmix reference dataset/tokenizer as the local harness
# (research_harness/prepare_data.py), inlined here for a self-contained
# notebook. Cache lives on Colab's local disk by default -- mount Drive and
# point CACHE_DIR there if you want it to survive a runtime restart.
# ==========================================================================

CACHE_DIR = "/content/dsp_lm_harness_cache"  # change to a Drive path to persist across sessions
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192
# climbmix-400b-shuffle is ~400B tokens across 6542 shards (~61M tokens/shard).
# 40 shards is ~2.4B tokens of raw text -- comfortably more than TOKEN_BUDGET
# below even if you bump it up for a longer real session, so the dataloader
# won't start cycling back over already-seen shards mid-run.
NUM_TRAIN_SHARDS = 40

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def download_single_shard(index):
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f"{BASE_URL}/{filename}"
    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/5 failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < 5:
                time.sleep(2**attempt)
    return False


def download_data(num_shards, download_workers=16):
    """Parallel shard download.

    Uses a thread pool, not multiprocessing.Pool. This is I/O-bound (waiting
    on HTTP requests), so threads are the right tool anyway -- no GIL
    contention to speak of -- but it's also a correctness fix, not just a
    style choice: multiprocessing.Pool needs to pickle download_single_shard
    and re-import this module in each worker process, which breaks inside a
    Colab/Jupyter notebook (there's no real .py file backing the running
    cells for a spawned worker to import), and separately, forking a process
    after CUDA has already been initialised in the parent (which happens
    above, in the nvidia-smi/torch.cuda calls) is explicitly unsafe per
    PyTorch's own docs. Threads share the parent process, so neither problem
    applies.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")
    workers = max(1, min(download_workers, needed))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(download_single_shard, ids))
    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens
    )
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s, saved to {tokenizer_pkl}")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    test = "Hello world! Numbers: 123. Unicode: 你好"
    assert enc.decode(enc.encode_ordinary(test)) == test, "Tokenizer roundtrip failed"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    with open(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found -- run the data prep cell first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """BOS-aligned dataloader with best-fit packing -- 100% utilization, no padding."""
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        doc_buffer.extend(tokenizer.encode(doc_batch, prepend=bos_token))

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx, best_len = i, doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, seq_len, eval_tokens):
    """Bits per byte -- vocab-size independent, comparable across configs."""
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val")
    steps = max(1, eval_tokens // (batch_size * seq_len))
    total_nats, total_bytes = 0.0, 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ==========================================================================
# Model: DSP-LM's ResonatorSSM + DendriticMLP, inlined (this notebook's own
# freely-editable copy -- see the module docstring for why).
# ==========================================================================

if _HAS_TRITON:

    @triton.jit
    def _vandermonde_fwd_kernel(
        neg_alpha_ptr, theta_ptr, cm_real_ptr, cm_imag_ptr, out_ptr, N, L, BLOCK_L: tl.constexpr, BLOCK_N: tl.constexpr
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
        decay = tl.exp(neg_alpha[None, :] * pos[:, None])
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
            g2 = tl.where(l_mask, 2.0 * g, 0.0)[:, None]
            decay = tl.exp(neg_alpha[None, :] * pos[:, None])
            ang = theta[None, :] * pos[:, None]
            cos_a = tl.cos(ang)
            sin_a = tl.sin(ang)
            term = cm_real[None, :] * cos_a - cm_imag[None, :] * sin_a
            acc_cm_real += tl.sum(g2 * decay * cos_a, axis=0)
            acc_cm_imag += tl.sum(-g2 * decay * sin_a, axis=0)
            acc_neg_alpha += tl.sum(g2 * pos[:, None] * decay * term, axis=0)
            acc_theta += tl.sum(
                g2 * decay * pos[:, None] * (-cm_real[None, :] * sin_a - cm_imag[None, :] * cos_a), axis=0
            )
        tl.store(d_cm_real_ptr + base_hn, acc_cm_real, mask=n_mask)
        tl.store(d_cm_imag_ptr + base_hn, acc_cm_imag, mask=n_mask)
        tl.store(d_neg_alpha_ptr + base_hn, acc_neg_alpha, mask=n_mask)
        tl.store(d_theta_ptr + base_hn, acc_theta, mask=n_mask)

    class _VandermondeKernelFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, neg_alpha, theta, cm_real, cm_imag, length):
            h, n = neg_alpha.shape
            block_n = triton.next_power_of_2(max(n, 1))
            block_l = 256
            out = torch.empty((h, length), device=neg_alpha.device, dtype=torch.float32)
            grid = (h, triton.cdiv(length, block_l))
            _vandermonde_fwd_kernel[grid](
                neg_alpha, theta, cm_real, cm_imag, out, n, length, BLOCK_L=block_l, BLOCK_N=block_n
            )
            ctx.save_for_backward(neg_alpha, theta, cm_real, cm_imag)
            ctx.length, ctx.block_n, ctx.block_l = length, block_n, block_l
            return out

        @staticmethod
        def backward(ctx, grad_out):
            neg_alpha, theta, cm_real, cm_imag = ctx.saved_tensors
            h, n = neg_alpha.shape
            grad_out = grad_out.contiguous()
            d_neg_alpha, d_theta = torch.empty_like(neg_alpha), torch.empty_like(theta)
            d_cm_real, d_cm_imag = torch.empty_like(cm_real), torch.empty_like(cm_imag)
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


class ResonatorSSMKernel(nn.Module):
    """S4D-style diagonal damped-complex-pole SSM kernel (see colab_trainable_dendritic_lm.py for full derivation notes)."""

    def __init__(self, d_model, n_states=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        half = n_states // 2
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        c = torch.randn(d_model, half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, half)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(half).repeat(d_model, 1).float())

    def _discretise(self):
        dt = torch.exp(self.log_dt)
        c = torch.view_as_complex(self.C)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt_a = a * dt.unsqueeze(-1)
        a_bar = torch.exp(dt_a)
        b_bar = (a_bar - 1.0) / a
        return a_bar, b_bar, c, dt_a

    def forward(self, length):
        a_bar, b_bar, c, dt_a = self._discretise()
        c_mod = c * b_bar
        if _HAS_TRITON and dt_a.is_cuda:
            kernel = _VandermondeKernelFn.apply(
                dt_a.real.contiguous(), dt_a.imag.contiguous(), c_mod.real.contiguous(), c_mod.imag.contiguous(), length
            )
        else:
            arange = torch.arange(length, device=a_bar.device)
            powers = torch.exp(dt_a.unsqueeze(-1) * arange)
            kernel = 2.0 * torch.einsum("hn,hnl->hl", c_mod, powers).real
        return kernel

    def initial_state(self, batch_size, device):
        half, h = self.log_A_real.shape[1], self.log_A_real.shape[0]
        return torch.zeros(batch_size, h, half, dtype=torch.cfloat, device=device)

    def step(self, x_t, h):
        a_bar, b_bar, c, _ = self._discretise()
        h_new = a_bar * h + b_bar * x_t.to(torch.cfloat).unsqueeze(-1)
        y_t = 2.0 * torch.einsum("hn,bhn->bh", c, h_new).real
        return y_t, h_new


class ResonatorSSM(nn.Module):
    def __init__(self, d_model, n_states=64):
        super().__init__()
        self.d_model = d_model
        self.kernel = ResonatorSSMKernel(d_model, n_states=n_states)
        self.D = nn.Parameter(torch.randn(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)

    @_dynamo_disable
    def _conv_mix(self, x):
        length = x.size(1)
        u = x.transpose(1, 2)
        kernel = self.kernel(length).to(torch.float32)
        u32 = u.to(torch.float32)
        n_fft = 2 * length
        k_f = torch.fft.rfft(kernel, n=n_fft)
        u_f = torch.fft.rfft(u32, n=n_fft)
        y = torch.fft.irfft(u_f * k_f, n=n_fft)[..., :length]
        y = y + u32 * self.D.unsqueeze(-1)
        return y.transpose(1, 2).to(x.dtype)

    def forward(self, x):
        y = self._conv_mix(x)
        return self.out_proj(y) * F.silu(self.gate_proj(x))

    def initial_state(self, batch_size, device):
        return self.kernel.initial_state(batch_size, device)

    def step(self, x_t, h):
        x32 = x_t.to(torch.float32)
        y_raw, h_new = self.kernel.step(x32, h)
        y_raw = (y_raw + x32 * self.D).to(x_t.dtype)
        return self.out_proj(y_raw) * F.silu(self.gate_proj(x_t)), h_new


class DendriticMLP(nn.Module):
    def __init__(self, d_model, num_branches=8, branch_dim=256, threshold=0.1, gate_steepness=10.0):
        super().__init__()
        self.d_model = d_model
        self.num_branches = num_branches
        self.branch_dim = branch_dim
        self.d_ff = num_branches * branch_dim
        self.threshold = threshold
        self.gate_steepness = gate_steepness
        self.value_proj = nn.Linear(d_model, self.d_ff)
        self.branch_gate = nn.Linear(d_model, num_branches)
        self.out_proj = nn.Linear(self.d_ff, d_model)

    def forward(self, x):
        b, t, _ = x.shape
        value = F.silu(self.value_proj(x)).view(b, t, self.num_branches, self.branch_dim)
        logit = self.branch_gate(x)
        gate = torch.sigmoid(self.gate_steepness * (torch.sigmoid(logit) - self.threshold))
        gated = value * gate.unsqueeze(-1)
        return self.out_proj(gated.reshape(b, t, self.d_ff))


class DendriticResonatorBlock(nn.Module):
    def __init__(self, d_model, n_states=64, num_branches=8, branch_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = ResonatorSSM(d_model, n_states=n_states)
        self.norm2 = nn.LayerNorm(d_model)
        self.dendrite = DendriticMLP(d_model, num_branches=num_branches, branch_dim=branch_dim)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.dendrite(self.norm2(x))
        return x

    def initial_state(self, batch_size, device):
        return self.ssm.initial_state(batch_size, device)

    def step(self, x_t, h):
        ssm_out, h_new = self.ssm.step(self.norm1(x_t), h)
        x_t = x_t + ssm_out
        x_t = x_t + self.dendrite(self.norm2(x_t).unsqueeze(1)).squeeze(1)
        return x_t, h_new


@dataclass
class DSPLMConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    d_model: int = 512
    depth: int = 6
    n_states: int = 64
    num_branches: int = 8
    branch_dim: int = 256
    use_checkpoint: bool = False


class DSPLM(nn.Module):
    def __init__(self, config: DSPLMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                DendriticResonatorBlock(
                    config.d_model, n_states=config.n_states, num_branches=config.num_branches, branch_dim=config.branch_dim
                )
                for _ in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.depth)
        for pname, p in self.named_parameters():
            if pname.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        blocks = sum(p.numel() for p in self.blocks.parameters())
        norm_out = sum(p.numel() for p in self.norm_out.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"embedding": embedding, "blocks": blocks, "norm_out": norm_out, "total": total}

    def estimate_flops(self):
        """Rough per-token FLOPs (diagnostic only, not the scored metric)."""
        nparams = sum(p.numel() for p in self.parameters())
        embed_params = self.embedding.weight.numel()
        dense_flops = 6 * (nparams - embed_params)
        t = self.config.sequence_len
        half_states = self.config.n_states // 2
        fft_flops_per_token = sum(10 * self.config.d_model * half_states * max(1, math.log2(2 * t)) for _ in self.blocks)
        return dense_flops + fft_flops_per_token

    def setup_optimizer(self, lr, weight_decay, betas):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or ".kernel." in name or name.endswith(".D"):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            dict(kind="adamw", params=decay, lr=lr, betas=betas, weight_decay=weight_decay),
            dict(kind="adamw", params=no_decay, lr=lr, betas=betas, weight_decay=0.0),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        x = self.embedding(idx)
        for block in self.blocks:
            if self.config.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm_out(x)
        logits = self.lm_head(x).float()
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)
        return logits


# ==========================================================================
# Data prep -- run once per Colab session (or point CACHE_DIR at Drive to
# skip this after the first run).
# ==========================================================================

download_data(NUM_TRAIN_SHARDS)
train_tokenizer()

# ==========================================================================
# Hyperparameters -- sized for an A100, not a 3060. These are starting
# points, not tuned values; the local 3060 harness found lr=4e-3 helped a
# lot at a tiny 4.8M-param/seq_len=1024 scale, but that doesn't necessarily
# transfer to a bigger model here -- expect to re-tune.
# ==========================================================================

# Architecture -- this defaults close to the project's own "42m" preset
# (colab_trainable_dendritic_lm.py MODEL_PRESETS), since an A100 has the
# headroom the 3060 never did. Bump toward "110m" if you want to go bigger.
D_MODEL = 512
DEPTH = 6
N_STATES = 64
NUM_BRANCHES = 8
BRANCH_DIM = 256
USE_CHECKPOINT = False  # flip on if you go big enough to OOM

# Batch -- A100 40GB has ~3.3x a 3060's 12GB (80GB has ~6.6x). Scale
# DEVICE_BATCH_SIZE to whatever fits; TOTAL_BATCH_SIZE is the real
# optimizer-step batch (grad-accumulated if DEVICE_BATCH_SIZE doesn't divide it).
MAX_SEQ_LEN = 2048  # DSP-LM's own project default -- no reason to cap this low anymore
DEVICE_BATCH_SIZE = 64  # 64*2048 = 131072 = 2**17 tokens/fwdbwd -- divides TOTAL_BATCH_SIZE cleanly (grad_accum=2).
                        # Reduce this (or flip on USE_CHECKPOINT) if you OOM.
TOTAL_BATCH_SIZE = 2**18  # ~262K tokens/optimizer-step -- a real pretraining-scale batch, not a toy one

LR = 4e-3  # 3060-harness's best local LR -- treat as a starting guess here, not gospel
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.0

# THE RULE CHANGE: no fixed 5-minute wall clock. Stop when either the token
# budget is hit or the wall-clock safety cap trips, whichever comes first.
# Size TOKEN_BUDGET to something that's actually a meaningful training chunk
# for your Colab session, not a toy smoke-test number.
TOKEN_BUDGET = 200_000_000  # 200M tokens -- a real chunk, not 5-minutes-on-an-H100's ~500M/300s toy number
WALL_CLOCK_SAFETY_SECONDS = 4 * 3600  # 4h safety net so a bad config can't eat your whole Colab session
EVAL_TOKENS = 20 * 52428  # bigger eval sample than the local harness (statistically firmer bpb reading)

# ==========================================================================
# Setup
# ==========================================================================

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = DSPLMConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    d_model=D_MODEL,
    depth=DEPTH,
    n_states=N_STATES,
    num_branches=NUM_BRANCHES,
    branch_dim=BRANCH_DIM,
    use_checkpoint=USE_CHECKPOINT,
)
print(f"Model config: {asdict(config)}")

model = DSPLM(config).to(device)
param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, (
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be a multiple of DEVICE_BATCH_SIZE*MAX_SEQ_LEN ({tokens_per_fwdbwd})"
)
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
effective_total_batch = grad_accum_steps * tokens_per_fwdbwd
print(f"Tokens/fwdbwd: {tokens_per_fwdbwd:,}  grad_accum: {grad_accum_steps}  effective batch: {effective_total_batch:,}")

optimizer = model.setup_optimizer(lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

expected_steps = TOKEN_BUDGET // effective_total_batch
print(f"Token budget: {TOKEN_BUDGET:,} (~{expected_steps:,} optimizer steps at this batch size)")
print(f"Wall-clock safety cap: {WALL_CLOCK_SAFETY_SECONDS}s")


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ==========================================================================
# Training loop -- stops on token budget OR wall-clock safety cap
# ==========================================================================

t_start_training = time.time()
smooth_train_loss = 0
step = 0
tokens_seen = 0

while True:
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(tokens_seen / TOKEN_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL: loss diverged")
        break

    dt = time.time() - t0
    tokens_seen += effective_total_batch
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    tok_per_sec = int(effective_total_batch / dt)
    elapsed = time.time() - t_start_training

    if step % 20 == 0:
        print(
            f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
            f"tok/sec: {tok_per_sec:,} | tokens: {tokens_seen/1e6:.1f}M | elapsed: {elapsed/60:.1f}min    ",
            end="",
            flush=True,
        )

    step += 1
    if tokens_seen >= TOKEN_BUDGET:
        print("\nStopping: token budget reached.")
        break
    if elapsed >= WALL_CLOCK_SAFETY_SECONDS:
        print("\nStopping: wall-clock safety cap reached (token budget not hit -- consider a smaller model/budget).")
        break

print()
total_training_time = time.time() - t_start_training

# ==========================================================================
# Eval + summary
# ==========================================================================

model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, EVAL_TOKENS)

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_minutes: {total_training_time/60:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {tokens_seen / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"seq_len:          {MAX_SEQ_LEN}")
print(f"depth:            {DEPTH}")

# ==========================================================================
# Results log -- append a row here after each run (edit the hyperparameters
# above, re-run the Setup + Training + Eval cells, then run this cell to log
# the outcome). Survives only for the current Colab session unless you save
# `experiment_log` to Drive/CSV.
# ==========================================================================

try:
    experiment_log
except NameError:
    experiment_log = []

experiment_log.append(
    {
        "val_bpb": val_bpb,
        "params_M": round(num_params / 1e6, 1),
        "seq_len": MAX_SEQ_LEN,
        "d_model": D_MODEL,
        "depth": DEPTH,
        "n_states": N_STATES,
        "branch_dim": BRANCH_DIM,
        "lr": LR,
        "tokens_M": round(tokens_seen / 1e6, 1),
        "peak_vram_gb": round(peak_vram_mb / 1024, 2),
    }
)

import pandas as pd

pd.DataFrame(experiment_log)

# NOTEBOOK-BUILD NOTE (not executed): after `jupytext --to notebook
# research_harness/autoresearch_notebook.py`, manually add two cells at the
# very top of the generated .ipynb, in order:
#   1. a code cell containing:  !nvidia-smi
#      (confirms you actually got the A100 you requested)
#   2. a code cell containing:  !pip install -q rustbpe tiktoken pyarrow requests
#      (torch and triton ship preinstalled on Colab's GPU runtimes)
# jupytext will not clobber hand-added cells on regeneration as long as you
# don't re-run it with --update against a stale copy that predates them --
# safest is to just re-insert these two cells after every regeneration.
