from collections import Counter
from multiprocessing import Pool
import os
import re
import onigurumacffi
from typing import BinaryIO

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
    find_merges_fn: optional (freq_table, num_merges) -> merges; default is find_merges.
    """
    if find_merges_fn is None:
        find_merges_fn = find_merges
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