#!/usr/bin/env python3
"""
Benchmark and profile find_merges: V3 (Python) vs Rust (bpe_rs).

Usage (from project root):

  # Compare wall-clock time (same freq_table, multiple runs)
  python scripts/benchmark_find_merges_v3_vs_rust.py

  # Rust internal timing breakdown (set BPE_RS_PROFILE=1)
  BPE_RS_PROFILE=1 python scripts/benchmark_find_merges_v3_vs_rust.py

  # Profile Rust with py-spy (native stack frames)
  py-spy record -o rust_profile.svg --native -- python scripts/benchmark_find_merges_v3_vs_rust.py rust_only

  # Profile with cargo flamegraph (if installed)
  cd bpe_rs && cargo flamegraph -- python ../scripts/benchmark_find_merges_v3_vs_rust.py rust_only
"""
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from tests.common import FIXTURES_PATH
from cs336_basics.bpe import pre_tokenize_file, find_merges, find_merges_rust


def main():
    rust_only = len(sys.argv) > 1 and sys.argv[1] == "rust_only"
    input_path = FIXTURES_PATH / "corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    num_merges = max(0, vocab_size - len(special_tokens) - 256)

    print("Pre-tokenizing once...", file=sys.stderr)
    freq_table = pre_tokenize_file(input_path, desired_num_chunks=100, special_tokens=special_tokens)
    print(f"  tokens={len(freq_table)}, num_merges={num_merges}", file=sys.stderr)

    if not rust_only:
        # V3 (Python)
        n_runs = 5
        times_v3 = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            find_merges(freq_table, num_merges)
            times_v3.append(time.perf_counter() - t0)
        mean_v3 = sum(times_v3) / len(times_v3)
        min_v3 = min(times_v3)
        print(f"\nV3 (Python find_merges): min={min_v3:.3f}s  mean={mean_v3:.3f}s  ({n_runs} runs)", file=sys.stderr)

    if find_merges_rust is None:
        print("Rust (bpe_rs) not installed; skipping.", file=sys.stderr)
        if rust_only:
            sys.exit(1)
        return

    # Rust
    n_runs = 5 if not rust_only else 20  # more runs when only Rust for profiling
    times_rust = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        find_merges_rust(freq_table, num_merges)
        times_rust.append(time.perf_counter() - t0)
    mean_rust = sum(times_rust) / len(times_rust)
    min_rust = min(times_rust)
    print(f"Rust (bpe_rs):           min={min_rust:.3f}s  mean={mean_rust:.3f}s  ({n_runs} runs)", file=sys.stderr)

    if not rust_only:
        speedup = mean_v3 / mean_rust if mean_rust > 0 else 0
        print(f"\nSpeedup (V3/Rust mean): {speedup:.2f}x", file=sys.stderr)


if __name__ == "__main__":
    main()
