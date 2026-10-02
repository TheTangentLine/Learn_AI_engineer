"""Tests for Week 9 Day 2: the BPE trainer, encoder and decoder, and parity with tiktoken (GPT-2) and Hugging Face (Qwen)."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day2_solution as d2  # noqa: E402
from bpe import BPE, END_OF_TEXT, GPT2_PATTERN  # noqa: E402

CORPUS = (
    "the cat sat on the mat. the dog sat on the log. the cat and the dog sat together. " * 20
    + "numbers 12 345 6789 and code: x = f(y)\n" * 5
)


@pytest.fixture(scope="module")
def tok() -> BPE:
    return BPE.train(CORPUS, 320, specials=[END_OF_TEXT])


# ----------------------------------------------------------------------------- the trainer


def test_the_textbook_example_merges_in_the_expected_order():
    t = BPE.train("aaabdaaabac", 258, pattern=r".+")
    assert t.ranks[b"aa"] == 256 and t.ranks[b"ab"] == 257, (
        "most frequent pair first; ties go to the smaller pair"
    )
    assert t.vocab_size == 258


def test_training_is_deterministic():
    a, b = BPE.train(CORPUS, 300), BPE.train(CORPUS, 300)
    assert a.ranks == b.ranks


def test_a_vocabulary_of_256_is_just_the_bytes_and_still_encodes_everything():
    t = BPE.train(CORPUS, 256)
    assert t.vocab_size == 256 and t.encode("é") == [195, 169]
    assert t.decode(t.encode("héllo")) == "héllo"


def test_the_vocabulary_must_hold_the_bytes_and_the_specials():
    with pytest.raises(ValueError, match="256 bytes"):
        BPE.train(CORPUS, 255)
    with pytest.raises(ValueError, match="256 bytes"):
        BPE.train(CORPUS, 256, specials=["<a>"])


def test_training_stops_when_there_is_nothing_left_to_merge():
    t = BPE.train("ab", 1000, pattern=r".+")
    assert t.vocab_size == 257, "one merge exists (a,b); after it the word is a single token"


def test_every_learned_token_is_a_new_string_and_ids_are_dense():
    t = BPE.train(CORPUS, 400)
    assert len(set(t.ranks)) == len(t.ranks)
    assert sorted(t.ranks.values()) == list(range(len(t.ranks)))


def test_the_encoder_reproduces_the_trainers_own_segmentation_of_every_training_word():
    """Training and encoding are two implementations of one idea: they must agree on the data the merges were learned from."""
    for text in (CORPUS, d2.course_text(8, 8)[:30000]):
        t, chunks, words = BPE._train(text, 600, GPT2_PATTERN, None)
        assert all(
            tuple(t._encode_chunk_uncached(c)) == tuple(w)
            for c, w in zip(chunks, words, strict=True)
        )


def test_merge_order_follows_frequency():
    t = BPE.train(CORPUS, 300)
    first = t.decode([256])
    assert first in (" t", "th", "he", " s", "at", " the"), (
        first
    )  # one of the most frequent pairs in CORPUS
    assert t.ranks[b" t"] < t.ranks.get(b" together", 10**9)


def test_more_merges_means_fewer_tokens_on_unseen_text():
    held = d2.course_text(8, 8)[:20000]
    train = d2.course_text(1, 7)[:60000]
    counts = [len(BPE.train(train, v).encode(held)) for v in (256, 400, 800)]
    assert counts[0] > counts[1] > counts[2]


# ----------------------------------------------------------------------------- encoding and decoding


def test_every_string_round_trips_including_ones_never_seen(tok):
    for t in d2.parity_texts() + d2.fuzz_texts(200, seed=1):
        assert tok.decode(tok.encode(t)) == t


def test_a_word_is_tokenised_the_same_wherever_it_appears(tok):
    alone = tok.encode(" cat")
    assert tok.encode("the cat sat")[1 : 1 + len(alone)] == alone
    assert tok.encode("a cat!")[1 : 1 + len(alone)] == alone


def test_the_leading_space_belongs_to_the_following_word(tok):
    assert tok.token_strings("the cat")[1].startswith(" ")


def test_decoding_ids_that_stop_inside_a_character_does_not_crash(tok):
    ids = tok.encode("é")
    assert len(ids) == 2
    assert tok.decode(ids[:1]) == "�"
    assert tok.decode_bytes(ids) == "é".encode()


def test_an_unknown_id_is_an_error(tok):
    with pytest.raises(ValueError, match="unknown token id"):
        tok.decode([10**6])


def test_special_tokens_must_be_allowed_explicitly(tok):
    with pytest.raises(ValueError, match="not allowed"):
        tok.encode(f"hello {END_OF_TEXT} world")
    ids = tok.encode(f"hello {END_OF_TEXT} world", allowed_special={END_OF_TEXT})
    assert tok.specials[END_OF_TEXT] in ids
    assert tok.decode(ids) == f"hello {END_OF_TEXT} world"
    assert tok.encode(END_OF_TEXT, allowed_special={END_OF_TEXT}) == [tok.specials[END_OF_TEXT]]


def test_empty_text_is_no_tokens(tok):
    assert tok.encode("") == [] and tok.decode([]) == ""


def test_a_long_run_of_one_character_is_encoded_without_trouble(tok):
    text = "a" * 5000
    assert tok.decode(tok.encode(text)) == text


def test_save_and_load_give_the_same_tokenizer(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    again = BPE.load(tmp_path / "t.json")
    assert (
        again.ranks == tok.ranks and again.specials == tok.specials and again.pattern == tok.pattern
    )
    assert again.encode("the cat sat") == tok.encode("the cat sat")


def test_construction_rejects_broken_vocabularies():
    good = {bytes([i]): i for i in range(256)}
    with pytest.raises(ValueError, match="256 single bytes"):
        BPE({b"a": 0})
    with pytest.raises(ValueError, match="share an id"):
        BPE(good | {b"ab": 5})
    with pytest.raises(ValueError, match="special token shares"):
        BPE(good, specials={"<x>": 7})
    with pytest.raises(ValueError, match="empty byte string"):
        BPE(good | {b"": 999})


# ----------------------------------------------------------------------------- parity with real tokenisers


def _tiktoken_gpt2():
    try:
        import tiktoken

        return tiktoken.get_encoding("gpt2")
    except Exception as exc:  # noqa: BLE001 - offline machine without the cached vocabulary
        pytest.skip(f"tiktoken's GPT-2 vocabulary is not available: {exc}")


def test_gpt2_vocabulary_gives_tiktokens_ids_on_every_hand_built_text_and_on_random_strings():
    ref = _tiktoken_gpt2()
    mine = BPE.from_tiktoken("gpt2")
    texts = d2.parity_texts() + d2.fuzz_texts(300, seed=0) + d2.fuzz_texts(300, seed=99)
    assert (
        d2.parity(
            mine, lambda t: ref.encode(t, allowed_special={END_OF_TEXT}), texts, {END_OF_TEXT}
        )
        == []
    )
    assert mine.vocab_size == ref.n_vocab == 50257


def test_the_classic_gpt2_pattern_agrees_with_tiktokens_current_one_on_the_test_set():
    ref = _tiktoken_gpt2()
    classic = BPE(BPE.from_tiktoken("gpt2").ranks, GPT2_PATTERN)
    assert d2.parity(classic, ref.encode, d2.parity_texts() + d2.fuzz_texts(300, seed=5)) == []


def test_the_qwen_vocabulary_with_its_own_regex_gives_hugging_faces_ids():
    paths = glob.glob(str(Path.home() / ".cache/huggingface/hub" / d2.QWEN_JSON))
    if not paths:
        pytest.skip("Qwen's tokenizer is not in the local Hugging Face cache")
    from transformers import AutoTokenizer

    qwen = BPE.from_hf_tokenizer_json(paths[0])
    hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    assert d2.parity(qwen, hf.encode, d2.parity_texts() + d2.fuzz_texts(300, seed=2)) == []
    assert qwen.encode("1234567") == [16, 17, 18, 19, 20, 21, 22], (
        "Qwen splits digits one by one: its regex says so, and ours agrees"
    )


def test_parity_detects_a_wrong_encoder():
    """A parity check that cannot fail proves nothing: change one merge priority and it must notice."""
    ref = _tiktoken_gpt2()
    mine = BPE.from_tiktoken("gpt2")
    ranks = dict(mine.ranks)
    a, b = ranks[b" the"], ranks[b" and"]
    ranks[b" the"], ranks[b" and"] = b, a
    wrong = BPE(ranks, mine.pattern, mine.specials)
    assert d2.parity(wrong, ref.encode, ["the cat and the dog", "and then the end"]) != []


def test_gpt2_tokens_for_known_strings():
    ref = _tiktoken_gpt2()
    mine = BPE.from_tiktoken("gpt2")
    assert mine.encode("Hello, world!") == ref.encode("Hello, world!") == [15496, 11, 995, 0]
    assert mine.token_strings("Hello, world!") == ["Hello", ",", " world", "!"]


# ----------------------------------------------------------------------------- the experiments' helpers


def test_fuzz_texts_are_deterministic_varied_and_encodable():
    a, b = d2.fuzz_texts(50, seed=3), d2.fuzz_texts(50, seed=3)
    assert a == b and a != d2.fuzz_texts(50, seed=4)
    assert (
        len({len(t) for t in a}) > 10
        and any(not t.isascii() for t in a)
        and any("\n" in t for t in a)
    )
    for t in a:
        t.encode("utf-8")


def test_the_regex_makes_words_tokenise_the_same_in_context_and_no_regex_does_not():
    train = d2.course_text(1, 7)[:120000]
    held = d2.course_text(8, 8)[:20000]
    with_regex = BPE.train(train, 600)
    flat = BPE.train(train, 600, pattern=r"[^\n]+|\n+")
    assert d2.boundary_alignment(with_regex, held) == 1.0
    assert d2.boundary_alignment(flat, held) < 0.5
    assert d2.merges_crossing_words(with_regex) == [] and len(d2.merges_crossing_words(flat)) > 20


def test_chars_per_token_and_the_random_alphabet_covers_every_script_it_claims():
    assert d2.chars_per_token(lambda t: [1, 2], "abcd") == 2.0
    assert d2.chars_per_token(lambda t: [], "abc") == 3.0, (
        "an empty encoding must not divide by zero"
    )
    s = "".join(d2.fuzz_texts(300, seed=0))
    for ch in "é日🙂ă\t\r":
        assert ch in s
