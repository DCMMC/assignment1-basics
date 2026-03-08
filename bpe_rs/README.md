# bpe_rs — high-performance find_merges in Rust

Same algorithm as Python V3 (linked list + incremental pair counts), with token interning (u32 ids) and `FxHashMap` for fast pair counts.

## Build (requires Rust + maturin)

From **project root** (so the venv is active):

```bash
uv pip install maturin
cd bpe_rs && uv run maturin develop --release
cd .. && uv run python -c "from bpe_rs import find_merges; print('OK')"
```

If your default Python is 3.14 and PyO3 does not support it yet:

```bash
cd bpe_rs && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

## Run tests (Rust vs V3)

With the extension installed:

```bash
uv run pytest tests/test_train_bpe.py::test_find_merges_speed_compare tests/test_train_bpe.py::test_find_merges_rust_example -v -s
```

You should see timings for original, optimization 1, optimization 2, and **Rust (bpe_rs)**.

## V3 vs Rust benchmark

From project root (same `freq_table`, multiple runs):

```bash
BPE_RS_PROFILE=1 uv run python scripts/benchmark_find_merges_v3_vs_rust.py
```

Typical result on `corpus.en` (vocab_size=500, 243 merges): **Rust ~15x faster** than Python V3 (e.g. V3 ~0.33s, Rust ~0.025s per run).

## Profile the Rust implementation

### 1. Internal timing (no extra tools)

Set `BPE_RS_PROFILE=1` to print a breakdown to stderr:

```bash
BPE_RS_PROFILE=1 uv run python scripts/benchmark_find_merges_v3_vs_rust.py
```

Example: almost all time is in `merge_loop`; convert/build/initial_pair_counts/to_python are negligible.

### 2. py-spy (flamegraph of Python process)

```bash
uv pip install py-spy
sudo uv run py-spy record -o rust_profile.svg -- `pwd`/../.venv/bin/python ../scripts/benchmark_find_merges_v3_vs_rust.py rust_only
```

Open `rust_profile.svg`. On **Linux**, add `--native` to include Rust/native stack frames; on **macOS** `--native` is not supported by py-spy, so you’ll see Python frames and time attributed to native code without symbol names. For Rust-level detail on macOS, use **Instruments** or **BPE_RS_PROFILE** (section 1).

### 3. perf + FlameGraph (Linux only)

`bpe_rs` is a library (PyO3 extension), so there is no binary to pass to `cargo flamegraph`. From **project root**, profile the Python process that calls into Rust:

```bash
perf record -F 99 -g -o perf.data -- python scripts/benchmark_find_merges_v3_vs_rust.py rust_only
```

Then generate a flamegraph using [FlameGraph](https://github.com/brendangregg/FlameGraph) scripts (clone the repo or use the ones from `cargo install flamegraph`’s vendored copy):

```bash
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

Open `flamegraph.svg` to see Rust and Python stacks.

**macOS native stacks:** use Xcode **Instruments** (Time Profiler) attached to the Python process, or rely on **BPE_RS_PROFILE** (section 1) for a timing breakdown.

### 4. perf report (Linux, quick view)

```bash
perf record --call-graph=dwarf -o perf.data -- python scripts/benchmark_find_merges_v3_vs_rust.py rust_only
perf report -i perf.data
```

Further optimization ideas (within the merge loop): reduce hash lookups in `pair_counts`, cache-friendly arena traversal, and avoiding full scan for tie-break when many pairs share the same count.
