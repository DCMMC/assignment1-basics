# ----------------------------------------------------------
# BPE implementation @ CS336 Assignment 1
# @author: Wentao XIAO <xwt97294597@gmail.com>
# ----------------------------------------------------------
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
import argparse
import json
import os
import re
import resource
import time
import onigurumacffi
from typing import BinaryIO, Iterable, Self

try:
    from bpe_rs import apply_bpe_encode_batch as _apply_bpe_encode_batch_rust
    from bpe_rs import find_merges as _find_merges_rust_impl

    def find_merges_rust(freq_table: Counter, num_merges: int):
        return _find_merges_rust_impl(dict(freq_table), num_merges)
except ImportError:
    _apply_bpe_encode_batch_rust = None
    find_merges_rust = None

# GPT-2 pre-tokenizer pattern
PRE_TOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    desired_num_chunks: int = 100,
    find_merges_fn=None,
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input corpus.
    find_merges_fn: optional (freq_table, num_merges) -> merges; default is find_merges_rust when bpe_rs is installed, else find_merges (V3).
    """
    if find_merges_fn is None:
        find_merges_fn = find_merges_rust if find_merges_rust is not None else find_merges
    # initialize the vocab with the special tokens
    vocab: dict[int, bytes] = {i : token.encode("utf-8") for i, token in enumerate(special_tokens)}
    vocab.update({i + len(special_tokens) : bytes([i]) for i in range(256)})
    # pre-tokenize the input corpuse
    freq_table = pre_tokenize_file(input_path, desired_num_chunks, special_tokens)
    num_merges = max(0, vocab_size - len(special_tokens) - 256)
    merges = find_merges_fn(freq_table, num_merges)
    for left, right in merges:
        vocab[len(vocab)] = left + right
    return vocab, merges


def pre_tokenize_file(input_path: str | os.PathLike, desired_num_chunks: int = 100,
                      special_tokens: list[str] = []) -> Counter:
    """
    Pre-tokenize the input file into chunks in parallel.
    """
    freq_table = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks,
            special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>",
        )
        # Optimization 0: Use multiprocessing only for large files
        if os.path.getsize(input_path) < 1024 * 1024:
            # Small file: run in main process to avoid multiprocessing overhead
            init_worker(input_path, PRE_TOKENIZER_PATTERN, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                freq_table.update(pre_tokenize_chunk(start, end))
        else:
            with Pool(
                processes=os.cpu_count(),
                initializer=init_worker,
                initargs=(input_path, PRE_TOKENIZER_PATTERN, special_tokens),
            ) as pool:
                results = pool.starmap(pre_tokenize_chunk, zip(
                    boundaries[:-1], boundaries[1:]))
                for result in results:
                    freq_table.update(result)
    return freq_table


def find_iter(regex: onigurumacffi._Pattern, chunk: bytes,
    special_tokens_regex: onigurumacffi._Pattern, start: int = 0
) -> Iterable[onigurumacffi._Match]:
    """
    Find all matches of the regex in the chunk, skipping special tokens.
    NOTE: use the underlying onigurumacffi API to avoid calling encode() and decode()
          and reuse region.
    """
    region = onigurumacffi._region()
    pos = start
    while pos < len(chunk):
        # Find next special token from pos
        ret = onigurumacffi._lib.onigcffi_search(
            special_tokens_regex._regex_t, chunk, len(chunk), pos, region,
            onigurumacffi.OnigSearchOption.NONE
        )
        spec_match = onigurumacffi._match_ret(ret, chunk, region)

        if spec_match is None:
            segment_end = len(chunk)
        else:
            segment_end = spec_match._begs[0]

        # Tokenize segment [pos, segment_end) with regex
        seg_start = pos
        while seg_start < segment_end:
            ret = onigurumacffi._lib.onigcffi_search(
                regex._regex_t, chunk, segment_end, seg_start, region,
                onigurumacffi.OnigSearchOption.NONE
            )
            match = onigurumacffi._match_ret(ret, chunk, region)
            if match:
                yield match
                seg_start = match._ends[0]
            else:
                break

        if spec_match is None:
            break
        pos = spec_match._ends[0]


def iter_encode_segments(
    chunk: bytes,
    regex: onigurumacffi._Pattern,
    special_tokens_regex: onigurumacffi._Pattern,
    start: int = 0,
) -> Iterable[tuple[bytes, bool]]:
    """
    Yield (segment_bytes, is_special) in order for encode().
    Like find_iter but also yields special-token segments as (bytes, True).
    """
    region = onigurumacffi._region()
    pos = start
    while pos < len(chunk):
        ret = onigurumacffi._lib.onigcffi_search(
            special_tokens_regex._regex_t, chunk, len(chunk), pos, region,
            onigurumacffi.OnigSearchOption.NONE,
        )
        spec_match = onigurumacffi._match_ret(ret, chunk, region)
        if spec_match is None:
            segment_end = len(chunk)
        else:
            segment_end = spec_match._begs[0]

        seg_start = pos
        while seg_start < segment_end:
            ret = onigurumacffi._lib.onigcffi_search(
                regex._regex_t, chunk, segment_end, seg_start, region,
                onigurumacffi.OnigSearchOption.NONE,
            )
            match = onigurumacffi._match_ret(ret, chunk, region)
            if match:
                yield (chunk[match._begs[0]:match._ends[0]], False)
                seg_start = match._ends[0]
            else:
                break

        if spec_match is None:
            break
        yield (chunk[spec_match._begs[0]:spec_match._ends[0]], True)
        pos = spec_match._ends[0]


# Worker state (set by init_worker for each process); used by pre_tokenize_chunk.
_worker_file = None
_worker_regex = None
_worker_special_tokens_regex = None


def init_worker(input_path, pattern, special_tokens):
    """Initialize each worker process with its own file handle and compiled regex."""
    global _worker_file, _worker_regex, _worker_special_tokens_regex
    _worker_file = open(input_path, "rb")
    _worker_regex = onigurumacffi.compile(pattern)
    _worker_special_tokens_regex = onigurumacffi.compile(
        "|".join(re.escape(t) for t in special_tokens) or r"(?!)"
    )


def pre_tokenize_chunk(start, end):
    _worker_file.seek(start)
    chunk = _worker_file.read(end - start)
    return Counter(
        chunk[m._begs[0]:m._ends[0]] for m in find_iter(_worker_regex, chunk, _worker_special_tokens_regex)
    )


class _Node:
    """Single node of a linked list of token bytes (for in-place BPE merge)."""
    __slots__ = ("token", "next")
    token: bytes
    next: "_Node | None"

    def __init__(self, token: bytes, next: "_Node | None" = None) -> None:
        self.token = token
        self.next = next


class _EncodeNode:
    """Single node of a linked list of token ids (for Rust-style encode merge)."""
    __slots__ = ("token_id", "next")
    token_id: int
    next: "_EncodeNode | None"

    def __init__(self, token_id: int, next: "_EncodeNode | None" = None) -> None:
        self.token_id = token_id
        self.next = next


# ---------------------------------------------------------------------------
# Version 1: Original — list-based, Counter, full pair_counts every round
# ---------------------------------------------------------------------------

def _merge_pair_in_list(
    tokens: list[bytes], left: bytes, right: bytes, merged: bytes
) -> list[bytes]:
    """Replace every consecutive (left, right) in tokens with merged."""
    if not left or not right:
        return tokens
    n = len(tokens)
    if n < 2:
        return tokens
    out = []
    i = 0
    while i < n:
        if i < n - 1 and tokens[i] == left and tokens[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def find_merges_original(freq_table: Counter, num_merges: int) -> list[tuple[bytes, bytes]]:
    """
    Original: words as list[bytes], Counter for pair counts, rebuild pair_counts
    and new token lists every round.
    """
    words: list[tuple[list[bytes], int]] = [
        ([bytes([b]) for b in token_bytes], count)
        for token_bytes, count in freq_table.items()
        if token_bytes
    ]
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for tokens, freq in words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        left, right = max(
            (p for p, c in pair_counts.items() if c == max_count),
            key=lambda p: (p[0], p[1]),
        )
        merges.append((left, right))
        merged = left + right
        new_words = []
        for tokens, freq in words:
            new_tokens = _merge_pair_in_list(tokens, left, right, merged)
            new_words.append((new_tokens, freq))
        words = new_words
    return merges


# ---------------------------------------------------------------------------
# Version 2: Optimization 1 — linked list, plain dict, full pair_counts every round
# ---------------------------------------------------------------------------

def find_merges_linked_list(freq_table: Counter, num_merges: int) -> list[tuple[bytes, bytes]]:
    """
    Optimization 1: linked list per word (merge in place), plain dict for pair
    counts (no Counter). Still recomputes pair_counts from scratch every round.
    """
    words: list[tuple[_Node, int]] = []
    for token_bytes, count in freq_table.items():
        if not token_bytes:
            continue
        head: _Node | None = None
        for b in reversed(token_bytes):
            head = _Node(bytes([b]), head)
        assert head is not None
        words.append((head, count))
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for head, freq in words:
            node = head
            while node and node.next:
                pair = (node.token, node.next.token)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                node = node.next
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        left, right = max(
            (p for p, c in pair_counts.items() if c == max_count),
            key=lambda p: (p[0], p[1]),
        )
        merges.append((left, right))
        merged = left + right
        for head, _ in words:
            node = head
            while node and node.next:
                if node.token == left and node.next.token == right:
                    node.token = merged
                    node.next = node.next.next
                else:
                    node = node.next
    return merges


# ---------------------------------------------------------------------------
# Version 3: Optimization 2 — linked list + incremental pair_counts update
# ---------------------------------------------------------------------------

def find_merges(freq_table: Counter, num_merges: int) -> list[tuple[bytes, bytes]]:
    """
    BPE merge algorithm using linked lists: words are (head, freq), pairs counted
    by traversing once (plain dict, no Counter). Merge by mutating nodes in place.
    """
    # Optimization 1: Build linked list for each word, avoid reallocating lists each merge
    words: list[tuple[_Node, int]] = []
    for token_bytes, count in freq_table.items():
        head: _Node | None = None
        for b in reversed(token_bytes):
            head = _Node(bytes([b]), head)
        words.append((head, count))

    merges: list[tuple[bytes, bytes]] = []
    def _dec(d: dict[tuple[bytes, bytes], int], k: tuple[bytes, bytes], v: int) -> None:
        d[k] = d.get(k, 0) - v
        if d[k] <= 0:
            del d[k]

    def _inc(d: dict[tuple[bytes, bytes], int], k: tuple[bytes, bytes], v: int) -> None:
        d[k] = d.get(k, 0) + v

    # Optimization 2: Count pairs with just one full pass in the first time
    # and incrementally update only affected pairs in other passes.
    # Initial pair counts: one full pass
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for head, freq in words:
        node = head
        while node and node.next:
            pair = (node.token, node.next.token)
            _inc(pair_counts, pair, freq)
            node = node.next

    for _ in range(num_merges):
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        left, right = max(
            (p for p, c in pair_counts.items() if c == max_count),
            key=lambda p: (p[0], p[1]),
        )
        merges.append((left, right))
        merged = left + right

        # One pass: update pair_counts for affected pairs and merge in place
        for head, freq in words:
            node = head
            prev: _Node | None = None
            while node and node.next:
                if node.token == left and node.next.token == right:
                    next_next = node.next.next
                    if prev:
                        _dec(pair_counts, (prev.token, left), freq)
                        _inc(pair_counts, (prev.token, merged), freq)
                    _dec(pair_counts, (left, right), freq)
                    if next_next:
                        _dec(pair_counts, (right, next_next.token), freq)
                        _inc(pair_counts, (merged, next_next.token), freq)
                    node.token = merged
                    node.next = next_next
                else:
                    prev = node
                    node = node.next
    return merges


def _merge_pair_in_ids_in_place(
    head: "_EncodeNode | None",
    left_id: int,
    right_id: int,
    merged_id: int,
) -> "_EncodeNode | None":
    """Merge one pair in a linked list of token ids in place (no extra list alloc)."""
    node = head
    while node is not None and node.next is not None:
        if node.token_id == left_id and node.next.token_id == right_id:
            node.token_id = merged_id
            node.next = node.next.next
        else:
            node = node.next
    return head


def _apply_merges_to_word_python_style(
    word_bytes: bytes,
    encoder: dict[bytes, int],
    merges_ids: list[tuple[int, int, int]],
) -> list[int]:
    """
    Apply BPE merges using a linked list and in-place merge (Python V3–style),
    operating purely in id space.

    `merges_ids` contains triples of (left_id, right_id, merged_id).
    """
    if not word_bytes:
        return []
    # Build initial linked list of token ids (one node per input byte)
    head: _EncodeNode | None = None
    for b in reversed(word_bytes):
        head = _EncodeNode(encoder[bytes([b])], head)
    # Apply each merge in id space
    for left_id, right_id, merged_id in merges_ids:
        head = _merge_pair_in_ids_in_place(head, left_id, right_id, merged_id)
    # Collect back to a Python list of ids
    out: list[int] = []
    while head is not None:
        out.append(head.token_id)
        head = head.next
    return out


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self._encoder: dict[bytes, int] = {v: k for k, v in vocab.items()}
        # Precompute merges in id space: (left_id, right_id, merged_id)
        merges_ids: list[tuple[int, int, int]] = []
        for left, right in merges:
            try:
                left_id = self._encoder[left]
                right_id = self._encoder[right]
                merged_id = self._encoder[left + right]
            except KeyError:
                # Skip any merge that doesn't have a corresponding token in vocab
                continue
            merges_ids.append((left_id, right_id, merged_id))
        self._merges_ids = merges_ids
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = [token.encode("utf-8") for token in special_tokens]
        # Match longest special token first so overlapping tokens
        # (e.g. "<|endoftext|><|endoftext|>") are one token
        special_ordered = sorted(special_tokens, key=len, reverse=True)
        self.special_tokens_regex = onigurumacffi.compile(
            "|".join(re.escape(t) for t in special_ordered) or r"(?!)"
        )
        self.regex = onigurumacffi.compile(PRE_TOKENIZER_PATTERN)

    def encode(
        self,
        text: str,
        *,
        use_rust_style: bool = True,
    ) -> list[int]:
        """
        Encode text to token ids. Segments from pre-tokenizer (regex + special tokens)
        are processed in order; each normal word is BPE-merged.

        use_rust_style=True:  Rust batch (bpe_rs.apply_bpe_encode_batch) if available, else Python linked list.
        use_rust_style=False: Python linked list + in-place merge.
        """
        chunk = text.encode("utf-8")
        use_rust = use_rust_style and _apply_bpe_encode_batch_rust is not None

        # We batch consecutive normal segments so that the Rust path can process
        # many words in a single call, and the Python path can also amortize overhead.
        result: list[int] = []
        pending_words: list[bytes] = []

        def flush_pending() -> None:
            nonlocal pending_words, result
            if not pending_words:
                return
            if use_rust and _apply_bpe_encode_batch_rust is not None:
                # Batch path: convert each word's bytes into initial byte-level ids
                words_ids: list[list[int]] = [
                    [self._encoder[bytes([b])] for b in word_bytes]
                    for word_bytes in pending_words
                ]
                encoded_batch = _apply_bpe_encode_batch_rust(
                    words_ids, self._merges_ids
                )
                for word_ids in encoded_batch:
                    result.extend(word_ids)
            else:
                # Pure Python path using linked-list merges in id space
                for word_bytes in pending_words:
                    result.extend(
                        _apply_merges_to_word_python_style(
                            word_bytes, self._encoder, self._merges_ids
                        )
                    )
            pending_words = []

        for segment_bytes, is_special in iter_encode_segments(
            chunk, self.regex, self.special_tokens_regex
        ):
            if is_special:
                flush_pending()
                result.append(self._encoder[segment_bytes])
            else:
                pending_words.append(segment_bytes)

        flush_pending()
        return result

    def decode(self, ids: list[int] | list[list[int]]) -> str:
        # Decode the concatenated token bytes as UTF-8 so multi-byte sequences (e.g. emoji) decode correctly.
        # Replace malformed bytes with the official Unicode replacement character (U+FFFD).
        if ids and isinstance(ids[0], list):
            ids = [tid for row in ids for tid in row]
        return b"".join(self.vocab[tid] for tid in ids).decode("utf-8", errors="replace")

    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None
    ) -> Self:
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {c: b for b, c in byte_encoder.items()}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            encoder = json.load(f)
        vocab: dict[int, bytes] = {
            token_id: _gpt2_token_to_bytes(token_str, byte_decoder)
            for token_str, token_id in encoder.items()
        }
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                left_str, right_str = line.split(" ", 1)
                merges.append((
                    _gpt2_token_to_bytes(left_str, byte_decoder),
                    _gpt2_token_to_bytes(right_str, byte_decoder),
                ))
        return cls(vocab, merges, special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        return chain.from_iterable(self.encode(text) for text in iterable)


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation.
    This is adapted from the GPT-2 code and duplicated here so that we can
    serialize BPE vocabularies and merges in a human-readable format without
    depending on the test utilities.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # characters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def _bytes_to_gpt2_token(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    """Convert a sequence of bytes into the printable GPT-2 unicode representation."""
    return "".join(byte_encoder[b] for b in token_bytes)


def _gpt2_token_to_bytes(token_str: str, byte_decoder: dict[str, int]) -> bytes:
    """Convert GPT-2 unicode token representation back to bytes."""
    return bytes(byte_decoder[c] for c in token_str)


def _serialize_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_prefix: Path,
) -> tuple[Path, Path, int, str]:
    """
    Serialize vocab and merges in a GPT-2-compatible, human-readable format.

    Returns:
        vocab_path, merges_path, longest_token_id, longest_token_repr
    """
    byte_encoder = gpt2_bytes_to_unicode()

    # Invert vocab to GPT-2-style encoder.json: token_string -> token_id
    encoder: dict[str, int] = {}
    for token_id, token_bytes in vocab.items():
        token_str = _bytes_to_gpt2_token(token_bytes, byte_encoder)
        encoder[token_str] = token_id

    vocab_path = out_prefix.with_suffix(".vocab.json")
    merges_path = out_prefix.with_suffix(".merges.txt")

    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(encoder, f, ensure_ascii=False, indent=2)

    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = _bytes_to_gpt2_token(left, byte_encoder)
            right_str = _bytes_to_gpt2_token(right, byte_encoder)
            f.write(f"{left_str} {right_str}\n")

    # Find the longest token in terms of underlying byte length
    longest_token_id, longest_token_bytes = max(
        vocab.items(), key=lambda kv: len(kv[1])
    )
    longest_token_repr = _bytes_to_gpt2_token(longest_token_bytes, byte_encoder)

    return vocab_path, merges_path, longest_token_id, longest_token_repr


def _run_experiment(
    name: str,
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    out_prefix: Path,
) -> None:
    """
    Train BPE on the given corpus, serialize vocab/merges, and report time, memory, and longest token.
    """
    print(f"Running experiment '{name}'")
    print(f"  Input path   : {input_path}")
    print(f"  Vocab size   : {vocab_size}")
    print(f"  Special tokens: {special_tokens}")

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_hours = elapsed_seconds / 3600.0

    # Peak memory usage (best-effort; units are platform-dependent but typically kilobytes on Unix).
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss_kb = usage.ru_maxrss
        max_rss_gb = max_rss_kb / (1024**2)
    except Exception:
        max_rss_kb = None
        max_rss_gb = None

    vocab_path, merges_path, longest_token_id, longest_token_repr = _serialize_vocab_and_merges(
        vocab, merges, out_prefix
    )

    longest_token_bytes = vocab[longest_token_id]

    print("\n=== Training summary ===")
    print(f"Experiment    : {name}")
    print(f"Elapsed time  : {elapsed_seconds:.2f} seconds (~{elapsed_hours:.2f} hours)")
    if max_rss_kb is not None:
        print(f"Peak RSS      : {max_rss_kb} KB (~{max_rss_gb:.3f} GB)")
    else:
        print("Peak RSS      : <unavailable on this platform>")
    print(f"Vocab path    : {vocab_path}")
    print(f"Merges path   : {merges_path}")
    print(f"Vocab size    : {len(vocab)}")
    print(f"Num merges    : {len(merges)}")
    print(
        f"Longest token : id={longest_token_id}, byte_length={len(longest_token_bytes)}, "
        f"token={repr(longest_token_repr)}"
    )
    print("========================\n")


def _run_encode_profile() -> None:
    """Profile tokenizer encode: compare Rust batch vs Python and print bottlenecks."""
    import cProfile
    import pstats

    project_root = Path(__file__).resolve().parent.parent
    fixtures = project_root / "tests" / "fixtures"
    vocab_path = fixtures / "gpt2_vocab.json"
    merges_path = fixtures / "gpt2_merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        print("Profile requires tests/fixtures/gpt2_vocab.json and gpt2_merges.txt")
        return
    tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, special_tokens=["<|endoftext|>"]
    )
    sample_path = fixtures / "tinystories_sample.txt"
    if sample_path.exists():
        full = sample_path.read_text(encoding="utf-8")
        text = full[: max(1, len(full) // 8)]
    else:
        fallback = "The quick brown fox jumps over the lazy dog. " * 200
        text = fallback[: len(fallback) // 8]

    n_runs = 5
    use_rust = _apply_bpe_encode_batch_rust is not None

    # --- Timing: Python ---
    print("\n=== Rust (batch) vs Python (sample size 1/8, {} runs) ===\n".format(n_runs))
    t0 = time.perf_counter()
    for _ in range(n_runs):
        tokenizer.encode(text, use_rust_style=False)
    t_py = time.perf_counter() - t0
    print("Python: {:.4f}s total  ({:.4f}s per run)".format(t_py, t_py / n_runs))

    # --- Timing: Rust batch ---
    if use_rust:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            tokenizer.encode(text, use_rust_style=True)
        t_rust_batch = time.perf_counter() - t0
        print(
            "Rust (batch):     {:.4f}s total  ({:.4f}s per run)".format(
                t_rust_batch, t_rust_batch / n_runs
            )
        )
        if t_rust_batch > 0:
            print(
                "  Speedup (Python / Rust batch): {:.2f}x\n".format(
                    t_py / t_rust_batch
                )
            )
    else:
        t_rust_batch = None
        print("Rust batch encode: not available (bpe_rs not installed)")

    # --- cProfile: Python implementation ---
    prof_py = cProfile.Profile()
    prof_py.enable()
    for _ in range(n_runs):
        tokenizer.encode(text, use_rust_style=False)
    prof_py.disable()
    stats_py = pstats.Stats(prof_py)
    stats_py.sort_stats(pstats.SortKey.CUMULATIVE)
    print("=== Python implementation (top 20 cumulative) ===\n")
    stats_py.print_stats(20)

    if use_rust:
        # --- cProfile: Rust implementation (overhead is in iter_encode_segments + Rust call) ---
        prof_rust = cProfile.Profile()
        prof_rust.enable()
        for _ in range(n_runs):
            tokenizer.encode(text, use_rust_style=True)
        prof_rust.disable()
        stats_rust = pstats.Stats(prof_rust)
        stats_rust.sort_stats(pstats.SortKey.CUMULATIVE)
        print("=== Rust implementation (top 20 cumulative) ===\n")
        stats_rust.print_stats(20)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser(
        description="BPE utilities for CS336 Assignment 1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile tokenizer encode and print bottlenecks.",
    )

    # Train on TinyStories
    tinystories_parser = subparsers.add_parser(
        "train_bpe_tinystories",
        help="Train a BPE tokenizer on the TinyStories training corpus.",
    )
    tinystories_parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Total vocabulary size (including special tokens). Default: 10000.",
    )
    tinystories_parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to prepend to the vocabulary. Default: <|endoftext|>.",
    )

    # Train on OpenWebText sample (expts_owt)
    owt_parser = subparsers.add_parser(
        "train_bpe_expts_owt",
        help="Train a BPE tokenizer on the OpenWebText (expts_owt) training corpus.",
    )
    owt_parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Total vocabulary size (including special tokens). Default: 32000.",
    )
    owt_parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to prepend to the vocabulary. Default: <|endoftext|>.",
    )

    args = parser.parse_args()

    if args.command == "profile":
        _run_encode_profile()
    elif args.command == "train_bpe_tinystories":
        input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
        out_prefix = data_dir / "tinystories_bpe"
        _run_experiment(
            name="TinyStories",
            input_path=input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            out_prefix=out_prefix,
        )
    elif args.command == "train_bpe_expts_owt":
        input_path = data_dir / "owt_train.txt"
        out_prefix = data_dir / "expts_owt_bpe"
        _run_experiment(
            name="expts_owt",
            input_path=input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            out_prefix=out_prefix,
        )