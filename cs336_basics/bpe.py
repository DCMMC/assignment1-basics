# ----------------------------------------------------------
# BPE implementation @ CS336 Assignment 1
# @author: Wentao XIAO <xwt97294597@gmail.com>
# ----------------------------------------------------------
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import argparse
import json
import os
import re
import resource
import time
import onigurumacffi
from typing import BinaryIO

try:
    from bpe_rs import find_merges as _find_merges_rust_impl
    def find_merges_rust(freq_table: Counter, num_merges: int):
        return _find_merges_rust_impl(dict(freq_table), num_merges)
except ImportError:
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


def find_iter(regex, chunk, special_tokens_regex, start=0):
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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser(
        description="BPE utilities for CS336 Assignment 1.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    if args.command == "train_bpe_tinystories":
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