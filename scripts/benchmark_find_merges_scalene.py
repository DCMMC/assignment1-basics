#!/usr/bin/env python3
"""
Run one of the three find_merges implementations for Scalene profiling.

Usage (from project root):
  scalene --outfile profile_original.html -- scripts/benchmark_find_merges_scalene.py original
  scalene --outfile profile_linked_list.html -- scripts/benchmark_find_merges_scalene.py linked_list
  scalene --outfile profile_incremental.html -- scripts/benchmark_find_merges_scalene.py incremental

Then open the .html files to compare CPU/memory bottlenecks across the three versions.
"""
import os
import sys

# Run from project root so cs336_basics and tests are importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from tests.common import FIXTURES_PATH
from cs336_basics.bpe import (
    train_bpe,
    find_merges_original,
    find_merges_linked_list,
    find_merges,
)

try:
    from cs336_basics.bpe import find_merges_rust
except ImportError:
    find_merges_rust = None

VERSIONS = {
    "original": find_merges_original,
    "linked_list": find_merges_linked_list,
    "incremental": find_merges,
}
if find_merges_rust is not None:
    VERSIONS["rust"] = find_merges_rust


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VERSIONS:
        opts = "|".join(VERSIONS.keys())
        print(f"Usage: {sys.argv[0]} {{{opts}}}", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    find_merges_fn = VERSIONS[name]
    input_path = FIXTURES_PATH / "corpus.en"
    print(f"Running train_bpe with find_merges={name} on {input_path}", file=sys.stderr)
    train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        find_merges_fn=find_merges_fn,
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
