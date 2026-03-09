# Benchmark and profile find_merges


## cProfile

```bash
uv run python -c "
import cProfile
import pstats
from io import StringIO
from cs336_basics.bpe import train_bpe
from tests.common import FIXTURES_PATH

pr = cProfile.Profile()
pr.enable()
train_bpe(FIXTURES_PATH / 'corpus.en', vocab_size=500, special_tokens=['<|endoftext|>'], desired_num_chunks=100)
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(25)
print(s.getvalue())
"
```

## V3 vs Rust (wall-clock + Rust internal profile)

From project root:

```bash
uv run python scripts/benchmark_find_merges_v3_vs_rust.py
```

Rust is typically ~15x faster. To see where time is spent inside Rust:

```bash
BPE_RS_PROFILE=1 uv run python scripts/benchmark_find_merges_v3_vs_rust.py
```

For native profiling (py-spy / flamegraph), see `bpe_rs/README.md`.

---

## Scalene: Python implementations

Compare bottlenecks of the three Python `find_merges` implementations (and Rust when installed) using [Scalene](https://github.com/plasma-umass/scalene).

Install Scalene (if needed):

```bash
uv pip install scalene
```

From the **project root** run (one per version):

```bash
uv run scalene run --outfile profile_original.json scripts/benchmark_find_merges_scalene.py original
uv run scalene run --outfile profile_linked_list.json scripts/benchmark_find_merges_scalene.py linked_list
uv run scalene run --outfile profile_incremental.json scripts/benchmark_find_merges_scalene.py incremental
uv run scalene run --outfile profile_rust.json scripts/benchmark_find_merges_scalene.py rust   # if bpe_rs installed
```

Then open the generated `profile_*.html` files in a browser to compare:

- **CPU %** and **Python vs native** time per line
- **Memory** allocation hotspots
- Which functions/lines dominate in each version (e.g. `_merge_pair_in_list` in original, pair counting loop in linked_list, incremental update in incremental)

Same dataset as `test_train_bpe_speed`: `tests/fixtures/corpus.en`, `vocab_size=500`.
