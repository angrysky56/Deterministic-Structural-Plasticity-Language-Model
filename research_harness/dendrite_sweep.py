#!/usr/bin/env python3
"""Ablation over DendriticMLP variants at matched parameter count.

Runs `train_harness.py` once per (variant, seed), parses its summary, and
writes `dendrite_sweep_results.tsv`. Every run shares the harness's fixed
wall-clock budget, dataset, tokenizer and eval metric, so val_bpb differences
should be attributable to the dendrite mechanism -- see THERMAL CONFOUND below
for the one way that assumption breaks on this machine.

The nested ablation isolates one factor per step:

    baseline  historical layer (its soma gate is measurably dead -- see
              tests/test_dendrite.py; it trains as a plain SiLU FFN)
    nmda      + self-gated supralinear branch threshold, learnable steepness k
    compart   + tail-up exp(-h/lambda) masking of value_proj, learnable lambda
    tree      + nonlinear confluence gate at each bifurcation

Multiple seeds are the point, not a luxury: with a fixed time budget the
run-to-run spread is often comparable to the effect being chased, so a
single-seed win is not evidence. The summary reports each variant's spread
and refuses to call anything a win inside the noise.

THERMAL CONFOUND (the reason this script is more than a for-loop)
-----------------------------------------------------------------
This RTX 3060 reports `GPU Max Operating Temp: 93 C`, and NVIDIA's boost
algorithm *targets* that number: the card raises clocks until it reaches 93C
and then holds there by backing them off. Sitting at 93C under load is the
card working as designed, not a fault -- it did this when new with a healthy
fan. Hardware slowdown is 95C and shutdown is 98C, so the margin is thin but
the steady state is normal.

The danger is not the card, it is the EXPERIMENT. The harness gives every run
the same 300 wall-clock seconds. A cold first run boosts high and completes
many steps; the eighth back-to-back run starts heat-soaked, throttles from
step zero, and completes fewer steps on the same clock. Fewer steps means
fewer tokens means worse val_bpb -- an effect with nothing to do with
dendrites. Run in the naive order (all baselines, then all nmda, ...), that
bias lands entirely on whichever variant went last and would fabricate a
result.

Three countermeasures, all on by default:

  1. Runs are INTERLEAVED (round-robin over variants within each seed round)
     so thermal drift is spread evenly across variants instead of loading
     onto the last one.
  2. Every run is PRE-HEATED to the same steady state (--warmup). Note this
     equalises HOT, not cold: measured cooldown on this card is 93->76C in 4
     minutes and 93->65C in 9, asymptotic after, so waiting for a cold start
     would add hours per sweep and never converge. Heating takes ~90s.
     Uniformly throttled runs are comparable, which is all the ablation needs.
  3. Throttle reasons, clocks, power limit and tokens processed are logged per
     run, and the summary REFUSES to interpret val_bpb if tokens varied more
     than --token-tolerance across runs, or if the power limit changed
     mid-sweep -- either means the runs did unequal work.

Measured cost of `-pl 130` vs stock 170W on this card (tree variant, same
seed): roughly 12% fewer tokens/sec, and it still reaches 93C once heat-soaked
-- the cap slows the climb, it does not prevent equilibrium. Keep it for
longevity and noise if you like, but do not expect it to stop throttling.

To reduce throttling outright, cap the board power (needs sudo):
    sudo nvidia-smi -pl 130          # 170W default; ~5% slower, ~10C cooler
or cap clocks, which also drops voltage:
    sudo nvidia-smi -lgc 210,1700
Ampere is well past its efficiency knee at stock, so this usually costs less
performance than it saves in throttling. Undo with `sudo nvidia-smi -rgc` /
`sudo nvidia-smi -pl 170`.

Usage:
    uv run research_harness/dendrite_sweep.py --seeds 2
    uv run research_harness/dendrite_sweep.py --variants baseline,tree --seeds 3
    uv run research_harness/dendrite_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HARNESS = HERE / "train_harness.py"
RESULTS_TSV = HERE / "dendrite_sweep_results.tsv"
LOG_DIR = HERE / "sweep_logs"

ALL_VARIANTS = ("baseline", "nmda", "nmda_t", "compart", "tree")

# This card's own reported thresholds (nvidia-smi -q -d TEMPERATURE):
#   Max Operating 93C  <- boost algorithm's target, normal under load
#   Slowdown      95C  <- hardware starts cutting clocks; a real problem
#   Shutdown      98C
# So the abort threshold is set from Slowdown, not from a guess about what
# "hot" means. Aborting at 93 would abort every run on a healthy card.
DEFAULT_MAX_TEMP = 95
# Warm-up TARGET, not a cooldown ceiling -- see warm_to_steady_state for why
# equalising cold is impossible on this card (93->65C takes 9 minutes).
DEFAULT_START_TEMP = 90

_SUMMARY_FIELDS = (
    "val_bpb",
    "num_params_M",
    "total_tokens_M",
    "num_steps",
    "peak_vram_mb",
    "mfu_percent",
    "gate_k_mean",
    "lambda_mean",
    "tau_rise_mean",
    "tau_decay_mean",
)


def permutation_pvalue(treat: list[float], control: list[float]) -> float:
    """Exact one-tailed permutation p-value that `treat` < `control`.

    The "is the gap bigger than the largest within-variant spread?" heuristic
    used elsewhere in this report is conservative and throws away rank
    information: three treatment runs that ALL beat every control run are
    strong evidence even when each group's internal spread is wide. This
    enumerates every way to split the pooled runs into two groups of the
    observed sizes and asks how often chance alone would produce a mean
    difference at least this favourable.

    With n=3 vs n=3 the smallest achievable p is 1/C(6,3) = 0.05, so treat
    0.05 here as "as strong as this design can show" rather than as proof --
    the fix is more seeds, not a different test.
    """
    from itertools import combinations

    pooled = treat + control
    n = len(treat)
    observed = statistics.mean(treat) - statistics.mean(control)
    idx = range(len(pooled))
    at_least_as_extreme = 0
    total = 0
    for combo in combinations(idx, n):
        a = [pooled[i] for i in combo]
        b = [pooled[i] for i in idx if i not in combo]
        total += 1
        if statistics.mean(a) - statistics.mean(b) <= observed:
            at_least_as_extreme += 1
    return at_least_as_extreme / total if total else 1.0


def _smi(query: str) -> list[str] | None:
    """Run an nvidia-smi --query-gpu and return the first row's fields."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return [f.strip() for f in out.stdout.strip().splitlines()[0].split(",")]
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):
        return None


def gpu_temperature() -> int | None:
    """Current GPU temperature in Celsius, or None if nvidia-smi is absent."""
    row = _smi("temperature.gpu")
    try:
        return int(row[0]) if row else None
    except (ValueError, IndexError):
        return None


def gpu_status() -> dict[str, str]:
    """Temperature, clocks, power draw and active throttle reasons.

    `clocks_event_reasons.active` is a bitmask; the two that matter here are
    sw_thermal_slowdown and hw_thermal_slowdown, which say the card is being
    held back by heat rather than by the power limit or by being idle.
    """
    row = _smi(
        "temperature.gpu,clocks.current.graphics,power.draw,power.limit,"
        "clocks_event_reasons.active"
    )
    if not row or len(row) < 5:
        return {}
    status = {
        "temp": row[0],
        "clock_mhz": row[1],
        "power_w": row[2],
        # Tracked per run because `nvidia-smi -pl` does NOT survive a driver
        # unload while persistence mode is disabled. A limit that silently
        # reverts to 170W mid-sweep would make some runs faster than others --
        # the exact heterogeneity the interleaving is there to prevent.
        "power_limit_w": row[3],
        "throttle_hex": row[4],
    }
    flags = _smi(
        "clocks_event_reasons.sw_thermal_slowdown,"
        "clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_power_cap"
    )
    if flags and len(flags) >= 3:
        active = []
        for name, value in zip(
            ("sw_thermal", "hw_thermal", "power_cap"), flags, strict=False
        ):
            if value.strip().lower() == "active":
                active.append(name)
        status["throttling"] = "+".join(active) if active else "none"
    return status


def warm_to_steady_state(seconds: int, target: int = 90) -> int | None:
    """Pre-heat the GPU so every measured run starts from the same thermal state.

    EQUALISE HOT, NOT COLD. The intuitive approach -- cool down between runs so
    each starts fresh -- does not work on this card. Measured cooldown from its
    93C load equilibrium, fan broken, machine powered:

        t+0min 93C | t+4min 76C | t+9min 65C | asymptotic thereafter

    Sub-60C takes 12+ minutes and ~42C is effectively unreachable, so a
    cool-down gate would add hours to a sweep and still not converge. Heating
    is the opposite: the card reaches its 93C equilibrium within ~90s of load.

    So instead of chasing an unreachable cold start, we deliberately drive
    every run to the SAME hot steady state before measuring. Uniformly
    throttled runs are comparable; differentially throttled runs are not, and
    comparability is the only property the ablation actually needs.

    Args:
        seconds: How long to hold the warm-up load (0 disables warm-up).
        target: Stop early once this temperature is reached.

    Returns:
        Temperature reached, or None if CUDA/nvidia-smi is unavailable.
    """
    if seconds <= 0:
        return gpu_temperature()
    try:
        import torch
    except ImportError:
        return gpu_temperature()
    if not torch.cuda.is_available():
        return gpu_temperature()

    print(f"  warming to {target}C (cap {seconds}s)...", flush=True)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    start = time.time()
    temp = gpu_temperature()
    # Wait for the TARGET, not for the clock. Stopping at a fixed duration
    # undershoots from a cold start: in the first sweep run 1 began at 45C and
    # only reached 82C in 90s while runs 2-8 all started at 92-93C, handing
    # run 1 ~20% more tokens in the same budget and biasing its variant.
    while temp is None or temp < target:
        for _ in range(50):
            a = (a @ b).mul_(0.001)
        torch.cuda.synchronize()
        temp = gpu_temperature()
        if time.time() - start > seconds:
            print(
                f"  !! only reached {temp}C in {seconds}s (target {target}C). "
                "Raise --warmup; this run is not thermally matched.",
                flush=True,
            )
            break
    del a, b
    torch.cuda.empty_cache()
    print(f"  warmed to {temp}C in {time.time() - start:.0f}s", flush=True)
    return temp


def parse_summary(stdout: str) -> dict[str, str]:
    """Pull the harness's `key: value` summary lines out of its stdout."""
    found: dict[str, str] = {}
    for field in _SUMMARY_FIELDS:
        m = re.search(rf"^{re.escape(field)}:\s*(\S+)", stdout, re.MULTILINE)
        if m:
            found[field] = m.group(1)
    return found


def interleave(variants: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    """Round-robin over variants within each seed round.

    (baseline,42) (nmda,42) (compart,42) (tree,42) (baseline,43) ...
    rather than all of baseline, then all of nmda. Thermal drift over the
    sweep then applies roughly equally to every variant instead of penalising
    whichever one happened to run last.
    """
    return [(v, s) for s in seeds for v in variants]


def run_one(
    variant: str, seed: int, env_extra: dict[str, str], dry_run: bool
) -> dict[str, str]:
    """Run the harness once and return its parsed summary."""
    env = {**os.environ, "DENDRITE": variant, "SEED": str(seed), **env_extra}
    cmd = [sys.executable, str(HARNESS)]

    if dry_run:
        print(f"  [dry-run] DENDRITE={variant} SEED={seed} {' '.join(cmd)}")
        return {}

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"dendrite_{variant}_seed{seed}.log"
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        print(f"  FAILED (exit {proc.returncode}) after {elapsed:.0f}s -> {log_path}")
        print(f"  stderr tail:\n{tail}")
        return {"status": "failed"}

    summary = parse_summary(proc.stdout)
    summary["status"] = "ok"
    summary["elapsed_s"] = f"{elapsed:.0f}"
    return summary


def report(rows: list[dict[str, str]], variants: list[str], token_tol: float) -> None:
    """Summarise, but only interpret val_bpb if the runs did equal work."""
    ok = [r for r in rows if r.get("status") == "ok" and "val_bpb" in r]
    if not ok:
        print("No successful runs to summarise.")
        return

    # ---- Confound check FIRST. If the runs did unequal work, val_bpb is not
    # comparable and no amount of averaging fixes it.
    comparable = True
    print("\n" + "=" * 72)
    print("VALIDITY CHECK")
    print("=" * 72)

    # Token spread has TWO causes and they mean opposite things:
    #   (a) different variants run at different speeds -- expected, and an
    #       honest property of the architecture under a fixed time budget;
    #   (b) the SAME variant did different work on different seeds -- that is
    #       thermal drift or another uncontrolled factor, and it is a confound.
    # Only (b) invalidates the comparison. Testing overall spread conflates
    # them and would condemn a sweep whose only "problem" is that one
    # architecture is genuinely slower than another.
    per_variant: dict[str, list[float]] = {}
    for r in ok:
        if r.get("total_tokens_M"):
            per_variant.setdefault(r["variant"], []).append(float(r["total_tokens_M"]))

    print("throughput by variant (fixed time budget => tokens ~ speed):")
    worst_within = 0.0
    for variant, toks in per_variant.items():
        mean = statistics.mean(toks)
        within = (max(toks) - min(toks)) / mean if len(toks) > 1 and mean else 0.0
        worst_within = max(worst_within, within)
        print(
            f"  {variant:<10} {mean:>6.1f}M tokens  "
            f"within-variant spread {within:>5.1%}"
        )
    print(
        f"\nwithin-variant spread (the confound signal): {worst_within:.1%} "
        f"(tolerance {token_tol:.0%})"
    )
    if worst_within > token_tol:
        comparable = False
        print(
            "\n!! THE SAME VARIANT DID UNEQUAL WORK ACROSS SEEDS. That is\n"
            "!! thermal drift or another uncontrolled factor, not architecture.\n"
            "!! val_bpb below is NOT safely interpretable -- check that warm-up\n"
            "!! reached target on every run, then re-run."
        )
    else:
        print("each variant did consistent work across seeds -- no thermal confound")
    if len(per_variant) > 1:
        means = [statistics.mean(t) for t in per_variant.values()]
        print(
            f"\nBetween-variant throughput differs by {(max(means)-min(means))/max(means):.0%}. "
            "Under a fixed TIME budget the slower variants also see fewer\n"
            "tokens, so val_bpb here measures quality-per-second, not\n"
            "quality-per-token. Re-run with MAX_STEPS to separate the two."
        )

    throttled = [r for r in ok if "thermal" in r.get("throttling_after", "")]
    if throttled:
        # Uniform throttling is fine -- it is equal for everyone. Only report
        # it; do not recommend a power cap, which on this card costs ~12%
        # throughput while the workload draws only ~70W and so is thermally,
        # not power, limited.
        print(
            f"\nthermal throttling in {len(throttled)}/{len(ok)} runs "
            "(uniform => comparable; the card is cooling-limited, not "
            "power-limited)"
        )
    # A power limit that changed mid-sweep invalidates the comparison just as
    # thoroughly as thermal drift does.
    limits = {r.get("power_limit_w", "") for r in ok if r.get("power_limit_w")}
    if len(limits) > 1:
        comparable = False
        print(
            f"\n!! POWER LIMIT CHANGED MID-SWEEP: saw {sorted(limits)} W.\n"
            "!! `nvidia-smi -pl` does not survive a driver unload unless\n"
            "!! persistence mode is on. Run `sudo nvidia-smi -pm 1`, reapply\n"
            "!! the limit, and re-run -- these results are not comparable."
        )
    elif limits:
        print(f"power limit: {limits.pop()}W, stable across all runs")

    by_variant: dict[str, list[float]] = {}
    for r in ok:
        by_variant.setdefault(r["variant"], []).append(float(r["val_bpb"]))

    print("\n" + "=" * 72)
    print("RESULTS (lower val_bpb is better)")
    print("=" * 72)
    print(
        f"{'variant':<10} {'n':>2} {'mean':>9} {'min':>9} {'spread':>9} {'vs base':>9}"
    )
    base = by_variant.get("baseline")
    base_mean = statistics.mean(base) if base else None
    spreads = []
    for variant in variants:
        vals = by_variant.get(variant)
        if not vals:
            continue
        mean = statistics.mean(vals)
        spread = max(vals) - min(vals) if len(vals) > 1 else float("nan")
        if len(vals) > 1:
            spreads.append(spread)
        delta = f"{mean - base_mean:+.4f}" if base_mean is not None else "-"
        print(
            f"{variant:<10} {len(vals):>2} {mean:>9.4f} {min(vals):>9.4f} "
            f"{spread:>9.4f} {delta:>9}"
        )

    noise = max(spreads) if spreads else None
    print()
    if noise is None:
        print("Only one seed per variant -- no noise estimate. Re-run with --seeds 2+")
        print("before treating any of these differences as real.")
    elif not comparable:
        print("Verdicts withheld: see VALIDITY CHECK above.")
    else:
        print(f"Seed noise (largest within-variant spread): {noise:.4f} bpb")
        print(f"{'variant':<10} {'delta':>9} {'perm p':>8}  verdict")
        for variant in variants:
            if variant == "baseline" or variant not in by_variant or base_mean is None:
                continue
            vals = by_variant[variant]
            delta = statistics.mean(vals) - base_mean
            p = permutation_pvalue(vals, base) if base else 1.0
            # Rank-based evidence and spread-based evidence can disagree; when
            # they do, the permutation test is the one to trust.
            if p <= 0.05:
                verdict = "significant (exact permutation)"
            elif abs(delta) < noise:
                verdict = "inside seed noise, no call"
            elif delta < 0:
                verdict = "better than noise, p>0.05 -- add seeds"
            else:
                verdict = "worse"
            print(f"  {variant:<8} {delta:>+9.4f} {p:>8.3f}  {verdict}")
        n_min = min(len(v) for v in by_variant.values())
        if n_min < 4:
            print(
                f"\n  note: n={n_min} per variant bounds the smallest reachable\n"
                f"  p at {1 / math.comb(2 * n_min, n_min):.3f}. Add seeds to "
                "strengthen, not a different test."
            )

    print("\nLearned mechanism (init: gate_k=1.0, lambda=4.0):")
    for variant in variants:
        vr = [r for r in rows if r["variant"] == variant and r.get("gate_k_mean")]
        if not vr:
            continue
        ks = [float(r["gate_k_mean"]) for r in vr]
        msg = f"  {variant:<9} gate_k={statistics.mean(ks):.3f}"
        lams = [float(r["lambda_mean"]) for r in vr if r.get("lambda_mean")]
        if lams:
            lam = statistics.mean(lams)
            tag = "compartmentalised" if lam < 4.0 else "reverting to dense"
            msg += f"  lambda={lam:.3f} ({tag})"
        print(msg)
    print(
        "\ngate_k above 1.0 means the layers sharpened the supralinear NMDA\n"
        "transition; near 0 means they linearised it away and the mechanism is\n"
        "inert for language. lambda below 4.0 means they chose compartments.\n"
        "A layer disagreeing with the hypothesis is a result, not a failure."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument(
        "--seeds", type=int, default=2, help="seeds per variant (>=2 to see noise)"
    )
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument(
        "--optimizer", default="muon", help="muon (best known: 1.4755) or adamw"
    )
    ap.add_argument("--muon-lr", default="0.02", help="best from muon_sweep_results")
    ap.add_argument(
        "--max-temp",
        type=int,
        default=DEFAULT_MAX_TEMP,
        help=f"abort above this (default {DEFAULT_MAX_TEMP}C = this card's "
        "hardware Slowdown threshold; 93C under load is NORMAL)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=90,
        help="seconds of pre-heat before each run so all runs start from the "
        "same thermal steady state (0 disables). See warm_to_steady_state: "
        "this card cools far too slowly to equalise cold.",
    )
    ap.add_argument(
        "--start-temp",
        type=int,
        default=DEFAULT_START_TEMP,
        help=f"warm-up target temperature (default {DEFAULT_START_TEMP}C)",
    )
    ap.add_argument(
        "--token-tolerance",
        type=float,
        default=0.05,
        help="max fractional spread in tokens processed before results are "
        "declared non-comparable (default 5%%)",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="fixed-TOKEN budget: give every variant this many steps instead "
        "of a fixed wall-clock budget. Separates per-token quality from "
        "throughput -- the decisive test when variants differ in speed.",
    )
    ap.add_argument("--no-interleave", action="store_true", help="not recommended")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ALL_VARIANTS]
    if bad:
        print(f"Unknown variant(s) {bad}; expected from {ALL_VARIANTS}")
        return 2

    seeds = [args.seed_base + i for i in range(args.seeds)]
    order = (
        [(v, s) for v in variants for s in seeds]
        if args.no_interleave
        else interleave(variants, seeds)
    )

    print("=" * 72)
    print("DENDRITE ABLATION SWEEP")
    print("=" * 72)
    print(f"variants:  {variants}")
    print(f"seeds:     {seeds}")
    print(f"optimizer: {args.optimizer} (muon_lr={args.muon_lr})")
    print(f"order:     {'sequential' if args.no_interleave else 'INTERLEAVED'}")
    print(f"runs:      {len(order)}  (~5 min each => ~{len(order) * 6} min total)")

    if not args.dry_run:
        st = gpu_status()
        print()
        print("GPU: 93C under load is this card's Max Operating Temp and is NORMAL.")
        print(f"     Aborting only above {args.max_temp}C (hardware Slowdown).")
        print(f"     Now: {st.get('temp', '?')}C, {st.get('clock_mhz', '?')}MHz")
        print("     To throttle less: sudo nvidia-smi -pl 130  (undo: -pl 170)")
        if not args.yes:
            try:
                if input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
                    print("Aborted.")
                    return 1
            except EOFError:
                print("\nNon-interactive; re-run with --yes to confirm.")
                return 1

    env_extra = {"OPTIMIZER": args.optimizer, "MUON_LR": args.muon_lr}
    if args.max_steps:
        env_extra["MAX_STEPS"] = str(args.max_steps)
        print(f"budget:    FIXED TOKENS ({args.max_steps} steps/run)")
    rows: list[dict[str, str]] = []

    for i, (variant, seed) in enumerate(order, 1):
        print(f"\n--- [{i}/{len(order)}] {variant} seed={seed} ---", flush=True)
        if not args.dry_run:
            warm_to_steady_state(args.warmup, args.start_temp)
        before = gpu_status()
        temp_before = before.get("temp")
        if temp_before and int(temp_before) > args.max_temp:
            print(f"ABORT: GPU at {temp_before}C, above --max-temp {args.max_temp}C")
            break

        summary = run_one(variant, seed, env_extra, args.dry_run)
        after = gpu_status()
        if args.dry_run:
            continue

        rows.append(
            {
                "variant": variant,
                "seed": str(seed),
                "order": str(i),
                "temp_before": temp_before or "",
                "temp_after": after.get("temp", ""),
                "clock_after": after.get("clock_mhz", ""),
                "power_after": after.get("power_w", ""),
                "power_limit_w": after.get("power_limit_w", ""),
                "throttling_after": after.get("throttling", ""),
                **summary,
            }
        )
        r = rows[-1]
        print(
            f"  val_bpb={r.get('val_bpb', 'n/a')} tokens={r.get('total_tokens_M', '?')}M "
            f"gate_k={r.get('gate_k_mean', '-')} lambda={r.get('lambda_mean', '-')}"
        )
        print(
            f"  thermals: {temp_before}C -> {r['temp_after']}C @ "
            f"{r['clock_after']}MHz {r['power_after']}W "
            f"throttle={r['throttling_after'] or 'n/a'}"
        )
        if r["temp_after"] and int(r["temp_after"]) > args.max_temp:
            print(f"ABORT: GPU hit {r['temp_after']}C after this run.")
            break

    if args.dry_run or not rows:
        return 0

    cols = [
        "order",
        "variant",
        "seed",
        "val_bpb",
        "num_params_M",
        "gate_k_mean",
        "lambda_mean",
        "total_tokens_M",
        "num_steps",
        "mfu_percent",
        "peak_vram_mb",
        "elapsed_s",
        "temp_before",
        "temp_after",
        "clock_after",
        "power_after",
        "power_limit_w",
        "throttling_after",
        "status",
    ]
    with RESULTS_TSV.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"\nWrote {RESULTS_TSV}")

    report(rows, variants, args.token_tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
