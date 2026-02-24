#!/usr/bin/env python3
"""Profile the full FHE pipeline (keygen → encrypt → fhetch → decrypt).

Usage (from project root):
    uv run python scripts/profiler.py [--params 1024|65536] [--mode time|memory|both]

Outputs:
  - Console: top functions by cumulative time, self time, and memory allocations.
  - File:    profile_<ring_dim>.prof  (open with: uv run snakeviz profile_1024.prof)
"""

import argparse
import cProfile
import io
import json
import pstats
import shutil
import tracemalloc
from pathlib import Path

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from client import serialization
from client.context import CryptoContext
from client.crypto import Parameters
from client.keygen import gen_relin_key, gen_sk
from client.utils import find_psi
from fhetch.data import Vector
from fhetch.env import default_global
from fhetch.eval import eval_globals, eval_main
from fhetch.ntt import ROOTS_UNITY, _nb_theory_scratchpad
from fhetch.parser import Program
from fhetch.validate import validate_all

TEMP_DIR = Path("temp")


# ── helpers ──────────────────────────────────────────────────────────────────

def setup_roots(params: Parameters) -> None:
    """Populate the ROOTS_UNITY global for all primes in params."""
    for q in params.moduli:
        ROOTS_UNITY[(params.degree, q)] = find_psi(q, params.degree)


def clear_ntt_caches() -> None:
    """Reset NTT scratchpad so each profiling run starts from the same state."""
    _nb_theory_scratchpad.powers_rou.clear()


def gen_random_msg(size: int) -> list:
    rng = np.random.default_rng()
    return [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(size)]


# ── pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(params: Parameters, ring_dim: int) -> None:
    """Run the complete key-switch pipeline in-process.

    Mirrors the steps in key-switch.sh but without subprocess overhead so
    every Python call is visible to cProfile / tracemalloc.
    """
    msg_size = ring_dim // 2

    # 1. Keygen
    print("  [1/5] Keygen …")
    sk = gen_sk(params)
    relin_key = gen_relin_key(sk, params.q, params.p)

    keys_dir = TEMP_DIR / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    np.save(keys_dir / "sk.npy", sk.value)
    for i, (ksk_0, ksk_1) in enumerate(relin_key):
        serialization.save_mrp(ksk_0, keys_dir / f"relin_d{i}_0.npz")
        serialization.save_mrp(ksk_1, keys_dir / f"relin_d{i}_1.npz")

    # 2. Encrypt two random messages
    print("  [2/5] Encrypting …")
    ctx = CryptoContext(params, sk)
    for name in ("ct_a", "ct_b"):
        ct = ctx.encrypt_msg(gen_random_msg(msg_size))
        d = TEMP_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        serialization.save_mrp(ct[0], d / "ct0.npz")
        serialization.save_mrp(ct[1], d / "ct1.npz")

    # 3. Run the fhetch interpreter (multiply → key-switch → rescale)
    print("  [3/5] Running fhetch interpreter …")
    (TEMP_DIR / "ct_res").mkdir(parents=True, exist_ok=True)
    prog_path = Path(f"examples/key_switch_{ring_dim}.fhetch")
    prog = Program.parse_file(prog_path, parse_all=True).program
    global_env = default_global()
    global_env.update(eval_globals(prog))
    validate_all(prog, global_env)
    eval_main(prog, global_env)

    # 4. Decrypt result
    print("  [4/5] Decrypting …")
    ct0 = serialization.load_mrp(TEMP_DIR / "ct_res" / "ct0.npz")
    ct1 = serialization.load_mrp(TEMP_DIR / "ct_res" / "ct1.npz")
    ctx.decrypt_msg([ct0, ct1])
    print("  [5/5] Done.")


# ── profilers ────────────────────────────────────────────────────────────────

def profile_time(params: Parameters, ring_dim: int) -> None:
    _print_header("TIME PROFILING  (cProfile)")

    clear_ntt_caches()
    pr = cProfile.Profile()
    pr.enable()
    run_pipeline(params, ring_dim)
    pr.disable()

    # --- cumulative time (best for finding call-chain hot-paths) ---
    print("\n── Top 30 by CUMULATIVE time ──────────────────────────────────")
    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(30)
    print(stream.getvalue())

    # --- self time (best for finding the actual compute bottleneck) ---
    print("── Top 20 by SELF time ────────────────────────────────────────")
    stream2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=stream2).sort_stats(pstats.SortKey.TIME)
    ps2.print_stats(20)
    print(stream2.getvalue())

    # Save .prof binary for interactive visualization
    prof_path = Path(f"profile_{ring_dim}.prof")
    pr.dump_stats(prof_path)
    print(f"  → Profile saved to {prof_path}")
    print(f"    Visualize interactively:  uv run snakeviz {prof_path}")
    print(f"    Or with py-spy:           uv run py-spy record -o profile.svg -- python scripts/profile.py")


def profile_memory(params: Parameters, ring_dim: int) -> None:
    _print_header("MEMORY PROFILING  (tracemalloc)")

    clear_ntt_caches()
    tracemalloc.start(25)  # capture up to 25 stack frames
    run_pipeline(params, ring_dim)
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Filter out stdlib / numpy internals noise
    filters = [
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ]
    snapshot = snapshot.filter_traces(filters)

    print("\n── Top 25 allocations by source line ─────────────────────────")
    for i, stat in enumerate(snapshot.statistics("lineno")[:25], 1):
        print(f"  {i:2d}. {stat}")

    print("\n── Top 10 allocations with full traceback ─────────────────────")
    for stat in snapshot.statistics("traceback")[:10]:
        size_kb = stat.size / 1024
        print(f"\n  {size_kb:8.1f} KiB  ({stat.count} allocation(s))")
        for line in stat.traceback.format():
            print(f"    {line}")

    # Summary by file
    print("\n── Memory grouped by file (top 15) ────────────────────────────")
    by_file: dict[str, int] = {}
    for stat in snapshot.statistics("filename"):
        by_file[stat.traceback[0].filename] = stat.size
    for path, size in sorted(by_file.items(), key=lambda x: -x[1])[:15]:
        print(f"  {size / 1024:8.1f} KiB  {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the fhetch FHE pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--params", choices=["1024", "65536"], default="1024",
        help="Ring dimension to use (default: 1024)",
    )
    parser.add_argument(
        "--mode", choices=["time", "memory", "both"], default="both",
        help="What to profile (default: both)",
    )
    args = parser.parse_args()

    ring_dim = int(args.params)
    params_file = Path(f"params_{ring_dim}.json")
    if not params_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_file}. Run from the project root.")

    with open(params_file) as f:
        params = Parameters(**json.load(f))

    # Pre-compute roots of unity outside the profiled sections so that
    # SymPy's primitive_root() cost is not included in pipeline timings.
    print(f"Setting up roots of unity for ring_dim={ring_dim} (may take a moment)…")
    setup_roots(params)
    print("Done.\n")

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    try:
        if args.mode in ("time", "both"):
            profile_time(params, ring_dim)

        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

        if args.mode in ("memory", "both"):
            profile_memory(params, ring_dim)

    finally:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        print("\nTemp directory cleaned up.")


if __name__ == "__main__":
    main()
