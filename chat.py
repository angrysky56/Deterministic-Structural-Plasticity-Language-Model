# -*- coding: utf-8 -*-
"""Interactive CLI for the DSP-LM.

Two modes:
    * completion (default): you type text, the model continues it.
    * --chat: wraps turns in a simple User/Assistant template and stops at EOS.

Examples:
    uv run python chat.py                                   # REPL, latest checkpoint
    uv run python chat.py --chat                            # chat REPL
    uv run python chat.py --prompt "The proof begins" -n 40 # one-shot completion
    uv run python chat.py --checkpoint checkpoints/DSP_LM/latest.pt

If no checkpoint exists it runs an UNTRAINED model so you can test the plumbing
(the output will be gibberish — that is expected until you have real weights).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from colab_trainable_dendritic_lm import VectorizedDendriticLM

# Model kwargs we need to reconstruct the network from a saved config dict.
_MODEL_KEYS = ("d_model", "depth", "n_states", "num_branches", "branch_dim")


def load_model(checkpoint: str | None, vocab_size: int, device: str):
    """Rebuild the model from a checkpoint's saved config, or a default."""
    kwargs: dict = {}
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        saved = ckpt.get("config", {})
        kwargs = {k: saved[k] for k in _MODEL_KEYS if k in saved}
        model = VectorizedDendriticLM(vocab_size=vocab_size, use_checkpoint=False, **kwargs)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step", "?")
        print(f"Loaded checkpoint {checkpoint} (optimizer step {step}) config={kwargs}")
    else:
        model = VectorizedDendriticLM(vocab_size=vocab_size, use_checkpoint=False)
        print("No checkpoint found — using an UNTRAINED model (output will be gibberish).")
    return model.to(device).eval()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    max_context: int,
    stop_ids: set[int],
):
    """Yield generated token ids one at a time, stopping on any stop id."""
    idx = prompt_ids
    for _ in range(max_new_tokens):
        # The SSM has no architectural context limit. max_context<=0 means feed
        # the full prefix (truly unbounded); a positive value is only a compute
        # guard for the parallel/FFT path, which reprocesses the prefix each step.
        window = idx if max_context <= 0 else idx[:, -max_context:]
        logits = model(window)[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        token = int(nxt.item())
        if token in stop_ids:
            break
        idx = torch.cat((idx, nxt), dim=1)
        yield token


def stream_text(token_iter, tokenizer) -> str:
    """Decode tokens as they arrive, printing incrementally."""
    produced: list[int] = []
    for token in token_iter:
        produced.append(token)
        # Decode the tail so multi-token characters render correctly.
        text = tokenizer.decode(produced[-8:])
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    return tokenizer.decode(produced) if produced else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="DSP-LM interactive CLI")
    ap.add_argument("--checkpoint", default="checkpoints/DSP_LM/latest.pt")
    ap.add_argument("--chat", action="store_true", help="use User/Assistant template")
    ap.add_argument("--prompt", default=None, help="one-shot prompt (skips the REPL)")
    ap.add_argument("-n", "--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-context", type=int, default=0,
                    help="0 = unbounded (SSM has no context limit); >0 caps the "
                         "FFT-path prefix per step as a compute guard")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e12)  # no context cap (see training script)
    model = load_model(args.checkpoint, len(tokenizer), device)
    eos = tokenizer.eos_token_id
    # In chat mode also stop if the model starts a new "User" turn.
    stop_ids = {eos}

    def run(text: str) -> None:
        prompt = f"User: {text}\nAssistant:" if args.chat else text
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        tokens = generate(
            model, tokenizer, ids, args.max_new, args.temperature,
            args.top_k, args.max_context, stop_ids,
        )
        print("Assistant:" if args.chat else "", end=" " if args.chat else "")
        stream_text(tokens, tokenizer)

    if args.prompt is not None:
        run(args.prompt)
        return

    print(f"DSP-LM CLI on {device} — {'chat' if args.chat else 'completion'} mode. "
          "Ctrl-C or empty line to quit.\n")
    try:
        while True:
            text = input("You: " if args.chat else "> ").strip()
            if not text:
                break
            run(text)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
