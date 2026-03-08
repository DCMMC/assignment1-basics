# Benchmark find_merges with Scalene

Compare bottlenecks of the three `find_merges` implementations using [Scalene](https://github.com/plasma-umass/scalene).

Install Scalene (if needed):

```bash
uv pip install scalene
```

From the **project root** run (one per version):

```bash
uv run scalene run --outfile profile_original.json scripts/benchmark_find_merges_scalene.py original
uv run scalene run --outfile profile_linked_list.json scripts/benchmark_find_merges_scalene.py linked_list
uv run scalene run --outfile profile_incremental.json scripts/benchmark_find_merges_scalene.py incremental
```

Then open the generated `profile_*.html` files in a browser to compare:

- **CPU %** and **Python vs native** time per line
- **Memory** allocation hotspots
- Which functions/lines dominate in each version (e.g. `_merge_pair_in_list` in original, pair counting loop in linked_list, incremental update in incremental)

Same dataset as `test_train_bpe_speed`: `tests/fixtures/corpus.en`, `vocab_size=500`.
