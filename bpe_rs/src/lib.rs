//! High-performance BPE find_merges: linked list in arena + incremental pair counts.
//! Token interning (u32 ids) for fast hashing; arena for cache-friendly nodes.
//!
//! Set env BPE_RS_PROFILE=1 to print timing breakdown to stderr.

use std::env;
use std::time::{Duration, Instant};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHashMap;

type TokenId = u32;
type Pair = (TokenId, TokenId);

/// Arena node: (token_id, next node index). None means end of list.
#[derive(Clone, Copy)]
struct Node {
    token_id: TokenId,
    next: Option<usize>,
}

/// In-place BPE merge: same algorithm as Python V3 (incremental pair counts).
#[pyfunction]
#[pyo3(signature = (freq_table, num_merges))]
fn find_merges(
    py: Python<'_>,
    freq_table: PyObject,
    num_merges: usize,
) -> PyResult<PyObject> {
    let profile = env::var_os("BPE_RS_PROFILE").is_some();
    let mut t0 = if profile { Some(Instant::now()) } else { None };

    // 1. Build token table: 0..256 for single bytes, then append merged tokens.
    let mut tokens: Vec<Vec<u8>> = (0u32..=255).map(|i| vec![i as u8]).collect();
    // 2. Build words as (start index in arena, freq). Arena holds all nodes.
    let mut arena: Vec<Node> = Vec::new();
    let mut words: Vec<(usize, u64)> = Vec::new();

    let mut t_convert = if profile { Some(Instant::now()) } else { None };
    let freq_items: Vec<(Vec<u8>, u64)> = py_freq_table_to_rust(py, &freq_table)?;
    let d_convert = if profile { t_convert.take().unwrap().elapsed() } else { Duration::ZERO };

    let mut t_build = if profile { Some(Instant::now()) } else { None };
    for (token_bytes, count) in freq_items {
        if token_bytes.is_empty() {
            continue;
        }
        let mut head: Option<usize> = None;
        for &b in token_bytes.iter().rev() {
            let id = b as TokenId;
            let idx = arena.len();
            arena.push(Node {
                token_id: id,
                next: head,
            });
            head = Some(idx);
        }
        if let Some(h) = head {
            words.push((h, count));
        }
    }
    let d_build = if profile { t_build.take().unwrap().elapsed() } else { Duration::ZERO };

    // 3. FxHashMap for pair counts (faster than default for integer pairs).
    let mut t_initial = if profile { Some(Instant::now()) } else { None };
    let mut pair_counts: FxHashMap<Pair, u64> = FxHashMap::default();
    for &(start, freq) in &words {
        let mut idx = Some(start);
        while let Some(i) = idx {
            let node = arena[i];
            if let Some(next_idx) = node.next {
                let pair = (node.token_id, arena[next_idx].token_id);
                *pair_counts.entry(pair).or_insert(0) += freq;
                idx = Some(next_idx);
            } else {
                break;
            }
        }
    }
    let d_initial = if profile { t_initial.take().unwrap().elapsed() } else { Duration::ZERO };

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(num_merges);

    let mut t_merge = if profile { Some(Instant::now()) } else { None };
    for _ in 0..num_merges {
        let max_count = *pair_counts.values().max().ok_or_else(|| PyValueError::new_err("no pairs"))?;
        // Tie-break: lexicographically greater pair (same as Python max key=(p[0], p[1]))
        let (left_id, right_id) = pair_counts
            .iter()
            .filter(|(_, &c)| c == max_count)
            .map(|(p, _)| *p)
            .max_by_key(|&(a, b)| (tokens[a as usize].as_slice(), tokens[b as usize].as_slice()))
            .unwrap();

        let left_bytes = tokens[left_id as usize].clone();
        let right_bytes = tokens[right_id as usize].clone();
        let mut merged = left_bytes.clone();
        merged.extend_from_slice(&right_bytes);
        let merged_id = tokens.len() as TokenId;
        tokens.push(merged);

        merges.push((left_bytes, right_bytes));

        // Incremental update + merge in place
        for &(start, freq) in &words {
            let mut idx = Some(start);
            let mut prev: Option<usize> = None;
            while let Some(i) = idx {
                let next_idx = arena[i].next;
                if let Some(j) = next_idx {
                    if arena[i].token_id == left_id && arena[j].token_id == right_id {
                        let next_next = arena[j].next;
                        if let Some(pi) = prev {
                            let prev_id = arena[pi].token_id;
                            dec_pair(&mut pair_counts, (prev_id, left_id), freq);
                            inc_pair(&mut pair_counts, (prev_id, merged_id), freq);
                        }
                        dec_pair(&mut pair_counts, (left_id, right_id), freq);
                        if let Some(nn) = next_next {
                            let next_id = arena[nn].token_id;
                            dec_pair(&mut pair_counts, (right_id, next_id), freq);
                            inc_pair(&mut pair_counts, (merged_id, next_id), freq);
                        }
                        arena[i].token_id = merged_id;
                        arena[i].next = next_next;
                        // Keep prev pointing at merged node (i) so next occurrence uses (merged, next) for prev
                        prev = Some(i);
                        idx = next_next;
                        continue;
                    }
                }
                prev = Some(i);
                idx = next_idx;
            }
        }
    }
    let d_merge = if profile { t_merge.take().unwrap().elapsed() } else { Duration::ZERO };

    if profile {
        let total = t0.take().unwrap().elapsed();
        eprintln!("[bpe_rs profile] total={:.3}s | convert(py->rust)={:.3}s build(arena+words)={:.3}s initial_pair_counts={:.3}s merge_loop={:.3}s",
            total.as_secs_f64(),
            d_convert.as_secs_f64(),
            d_build.as_secs_f64(),
            d_initial.as_secs_f64(),
            d_merge.as_secs_f64(),
        );
    }

    // Convert to Python list of (bytes, bytes)
    let mut t_to_py = if profile { Some(Instant::now()) } else { None };
    let py_merges = pyo3::types::PyList::empty(py);
    for (a, b) in merges {
        let py_a = PyBytes::new(py, &a);
        let py_b = PyBytes::new(py, &b);
        py_merges.append((py_a, py_b))?;
    }
    if profile {
        let d_to_py = t_to_py.take().unwrap().elapsed();
        eprintln!("[bpe_rs profile] to_python={:.3}s", d_to_py.as_secs_f64());
    }
    Ok(py_merges.into_py(py))
}

#[inline]
fn inc_pair(map: &mut FxHashMap<Pair, u64>, k: Pair, v: u64) {
    *map.entry(k).or_insert(0) += v;
}

#[inline]
fn dec_pair(map: &mut FxHashMap<Pair, u64>, k: Pair, v: u64) {
    if let Some(c) = map.get_mut(&k) {
        *c -= v;
        if *c == 0 {
            map.remove(&k);
        }
    }
}

fn py_freq_table_to_rust(
    py: Python<'_>,
    freq_table: &PyObject,
) -> PyResult<Vec<(Vec<u8>, u64)>> {
    let dict = freq_table.downcast_bound::<pyo3::types::PyDict>(py)?;
    let mut out = Vec::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let bytes = k.downcast::<PyBytes>()?;
        let count: u64 = v.extract()?;
        out.push((bytes.as_bytes().to_vec(), count));
    }
    Ok(out)
}

#[pymodule]
fn bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_merges, m)?)?;
    Ok(())
}
