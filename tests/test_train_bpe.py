import json
import time

from collections import Counter

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_pre_tokenize_file(tmp_path):
    """
    pre_tokenize_file counts token frequencies. Tokens may include leading space
    from the pre-tokenizer pattern; we normalize by stripping leading space for comparison.
    """
    from cs336_basics.bpe import pre_tokenize_file

    text = (
        "low low low low low\n"
        "lower lower widest widest widest\n"
        "newest newest newest newest newest newest\n"
        "<|endoftext|>"
        "low low low low low\n"
        "lower lower widest widest widest\n"
        "newest newest newest newest newest newest\n"
        "<|endoftext|>"
    )
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(text)

    freq_table = pre_tokenize_file(
        corpus_file,
        desired_num_chunks=1,
        special_tokens=["<|endoftext|>"],
    )

    # Normalize: GPT-2 pattern may produce " word" (leading space), count as "word"
    normalized = Counter()
    for token_bytes, count in freq_table.items():
        key = token_bytes.lstrip(b" ")
        if key:
            normalized[key] += count

    expected = {
        b"\n": 6,
        b"low": 10,
        b"lower": 4,
        b"widest": 6,
        b"newest": 12,
    }
    assert dict(normalized) == expected


def test_find_merges_example():
    """
    Example from the spec: low(5), lower(2), widest(3), newest(6).
    First round pair counts: lo:7, ow:7, we:8, er:2, wi:3, id:3, de:3, es:9, st:9, ne:6, ew:6.
    Tie (es, st) -> take lexicographically greater (st). First 6 merges: s t, e st, o w, l ow, w est, n e.
    """
    from cs336_basics.bpe import find_merges

    freq_table = Counter({
        b"low": 5,
        b"lower": 2,
        b"widest": 3,
        b"newest": 6,
    })
    merges = find_merges(freq_table, num_merges=6)
    expected_first_6 = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]
    assert merges == expected_first_6


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    cost = end_time - start_time
    assert cost < 1.5


def test_find_merges_speed_compare():
    """
    Compare performance of the three find_merges implementations using the same
    setup as test_train_bpe_speed (corpus.en, vocab_size=500). All three must
    produce the same merges; the default (optimization 2) must stay under 1.5s.
    """
    from cs336_basics.bpe import (
        train_bpe,
        find_merges_original,
        find_merges_linked_list,
        find_merges,
    )
    input_path = FIXTURES_PATH / "corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    versions = [
        ("original", find_merges_original),
        ("optimization 1 (linked list)", find_merges_linked_list),
        ("optimization 2 (incremental)", find_merges),
    ]
    results = []
    for name, find_merges_fn in versions:
        start = time.time()
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            find_merges_fn=find_merges_fn,
        )
        elapsed = time.time() - start
        results.append((name, merges, elapsed))
    ref_merges = results[0][1]
    for name, merges, elapsed in results:
        assert merges == ref_merges, f"{name} produced different merges"
    assert results[2][2] < 1.5, f"optimization 2 took {results[2][2]:.2f}s (limit 1.5s)"
    for name, _, elapsed in results:
        print(f"  find_merges: {name}: {elapsed:.3f}s")


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
