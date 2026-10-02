"""Week 9 Day 2 - Solution: a byte-level BPE tokenizer, validated against two real ones.

1. PARITY    the encoder in ``bpe.py``, loaded with GPT-2's vocabulary, must give tiktoken's ids on a hand-built set and on random strings; loaded
             with Qwen's vocabulary it must give Hugging Face's ids (a second, independent reference with a different pre-tokenising regex)
2. TRAIN     learn merges from this course's own text (weeks 1-7), measure compression on text it has not seen (week 8), and sweep the vocabulary size
3. ABLATE    train once without the pre-tokenising regex: merges then cross word boundaries
4. ARTEFACTS what the tokenizer makes of leading spaces, digits, other languages and code, for four different vocabularies

  uv run python weeks/week09_transformers-from-scratch/solutions/day2_solution.py
"""

from __future__ import annotations

import glob
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import regex  # noqa: E402
from bpe import BPE, END_OF_TEXT, GPT2_PATTERN  # noqa: E402

from common.corpus import load_course_docs  # noqa: E402

QWEN_JSON = "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/tokenizer.json"


def course_text(first_week: int, last_week: int) -> str:
    docs = [
        d for d in load_course_docs() if first_week <= int(d.short.split("/")[0][4:]) <= last_week
    ]
    return "\n\n".join(d.text for d in sorted(docs, key=lambda d: d.path))


def parity_texts() -> list[str]:
    """Hand-built cases for the places tokenisers differ: whitespace, contractions, digits, punctuation runs, code, other scripts, emoji."""
    return [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "don't stop; I'll go; they've said we're done; she'd say it's fine; y'all",
        "DON'T SHOUT, I'LL GO",
        "  two  spaces and\ttabs\tand trailing  ",
        "line one\nline two\r\nline three\n\n\nlast",
        "   leading spaces",
        "trailing spaces    ",
        "1234567890 3.14159 1,000,000 -42 +7 0xFF 1e-9",
        "!!! ??? ... --- === ### $$$ %%% &&& *** (((x))) [[y]] {{z}}",
        "def f(x):\n    if x > 0:\n        return x ** 2  # square\n    return -x",
        '<html><body class="a">&amp; &lt;tag&gt;</body></html>',
        "https://example.com/path?query=1&other=two#fragment",
        "user@example.com and path/to/file.py and C:\\Windows\\System32",
        "Xin chào Việt Nam! Tôi là một mô hình ngôn ngữ.",
        "日本語のテキスト、中文文本、한국어 텍스트",
        "Привет, мир! Ελληνικά, עברית, العربية",
        "emoji 🙂🚀👨‍👩‍👧‍👦 and flags 🇻🇳 and ❤️",
        "café naïve résumé Zürich Ångström",
        "zero\u200bwidth and non\u00a0breaking and \u2028 line separator",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "a" * 200 + " " + "b" * 200,
        "SolidGoldMagikarp",
        " ",
        "\n",
        "x",
        "I have 3 apples. She has 12. They have 1500!",
        "The retry limit is 3 times with exponential backoff.",
        "mixed: ÀÉÎÕÜ àéîõü ß ñ ç œ æ ø",
        "tab\tseparated\tvalues\t\tdouble",
        "\x00\x01\x02 control characters \x1f",
    ]


def fuzz_texts(n: int = 300, seed: int = 0) -> list[str]:
    """Random strings over a deliberately nasty alphabet: ASCII, digits, punctuation, every kind of whitespace, accents, CJK, emoji."""
    rng = random.Random(seed)
    alphabet = (
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") * 3
        + list("0123456789") * 2
        + list(" " * 8 + "\n\t\r")
        + list("'.,;:!?-_()[]{}<>/\\\"@#$%^&*+=~`|")
        + list("éèüñçßÅøÀ")
        + list("日本語中文한국어")
        + list("🙂🚀🎉")
        + list("ăâêôơưđ")
    )
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(n)]


def parity(
    mine: BPE, reference, texts: list[str], allowed: set[str] | None = None
) -> list[tuple[str, list[int], list[int]]]:
    """Texts where ``mine.encode`` and ``reference(text)`` disagree: (text, mine, reference). Empty means parity."""
    bad = []
    for t in texts:
        a = mine.encode(t, allowed_special=allowed or set())
        b = list(reference(t))
        if a != b:
            bad.append((t, a, b))
    return bad


def chars_per_token(encode, text: str) -> float:
    return len(text) / max(1, len(encode(text)))


def vocab_sweep(train: str, held: str, sizes: list[int]) -> list[dict]:
    rows = []
    for size in sizes:
        tok = BPE.train(train, size)
        ids = tok.encode(held)
        rows.append(
            {
                "vocab": tok.vocab_size,
                "tokens": len(ids),
                "chars_per_token": len(held) / len(ids),
                "roundtrip": tok.decode(ids) == held,
            }
        )
    return rows


def merges_crossing_words(tok: BPE) -> list[bytes]:
    """Learned tokens with a non-space character followed by a space ('e ', ', ', 'the '): a merge across a word boundary. The GPT-2 regex
    attaches a space to the FOLLOWING word, so this cannot happen with it (whitespace-only tokens like '\\n   ' are fine)."""
    return [b for b in tok.ranks if len(b) > 1 and re.search(rb"[^\s]\s", b)]


def boundary_alignment(tok: BPE, text: str, pattern: str = GPT2_PATTERN) -> float:
    """Of the words (regex chunks) in ``text``, the share that are tokenised IN CONTEXT exactly as when the word is encoded alone. 100% means a
    word always gets the same tokens whatever surrounds it, so a model sees one representation per word; below that, 'the' before a comma and
    'the' before a space are different token sequences."""
    ids = tok.encode(text)
    starts, pos = {}, 0
    for k, i in enumerate(ids):
        starts[pos] = k
        pos += len(tok._decoder[i])
    ends = {}
    pos = 0
    for k, i in enumerate(ids):
        pos += len(tok._decoder[i])
        ends[pos] = k
    ok = total = 0
    bpos = 0
    for m in regex.finditer(pattern, text):
        b = m.group().encode("utf-8")
        total += 1
        if (
            bpos in starts
            and bpos + len(b) in ends
            and ids[starts[bpos] : ends[bpos + len(b)] + 1] == tok.encode(m.group())
        ):
            ok += 1
        bpos += len(b)
    return ok / total


def main(argv: list[str]) -> None:
    import tiktoken
    from transformers import AutoTokenizer

    texts = parity_texts() + fuzz_texts(300)
    print("1. PARITY with real tokenisers (hand-built cases + 300 random strings)")
    gpt2 = BPE.from_tiktoken("gpt2")
    ref = tiktoken.get_encoding("gpt2")
    bad = parity(gpt2, lambda t: ref.encode(t, allowed_special={END_OF_TEXT}), texts, {END_OF_TEXT})
    print(
        f"   GPT-2 vocabulary (50,257 ids) vs tiktoken: {len(texts) - len(bad)} of {len(texts)} identical"
    )
    qpath = glob.glob(str(Path.home() / ".cache/huggingface/hub" / QWEN_JSON))
    if qpath:
        qwen = BPE.from_hf_tokenizer_json(qpath[0])
        hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        bad_q = parity(qwen, lambda t: hf.encode(t), texts)
        print(
            f"   Qwen2.5 vocabulary (151,643 ids, a different regex) vs Hugging Face: {len(texts) - len(bad_q)} of {len(texts)} identical"
        )
    round_trips = sum(gpt2.decode(gpt2.encode(t)) == t for t in texts)
    print(
        f"   decode(encode(text)) == text for {round_trips} of {len(texts)} (every string, including lone control characters)\n"
    )

    train, held = course_text(1, 7), course_text(8, 8)
    print(
        f"2. TRAIN on the course text (weeks 1-7: {len(train):,} characters), evaluate on week 8 ({len(held):,} characters, not seen)"
    )
    print(f"   {'vocab':>6}{'tokens on week 8':>18}{'chars/token':>13}")
    for r in vocab_sweep(train, held, [256, 300, 512, 1024, 2048, 4096]):
        print(f"   {r['vocab']:>6}{r['tokens']:>18,}{r['chars_per_token']:>13.2f}")
    mine = BPE.train(train, 1024, specials=[END_OF_TEXT])
    refs = {"GPT-2 (50k)": gpt2.encode, "mine (1,024)": mine.encode}
    refs["o200k (200k)"] = tiktoken.get_encoding("o200k_base").encode
    if qpath:
        refs["Qwen2.5 (152k)"] = qwen.encode
    print(
        "\n   the same week-8 text, other vocabularies:   "
        + "   ".join(f"{k}: {chars_per_token(v, held):.2f}" for k, v in refs.items())
    )
    print(
        "   first tokens of 'The retry limit is 3 times with exponential backoff.': "
        + " | ".join(mine.token_strings("The retry limit is 3 times with exponential backoff."))
    )

    print("\n3. ABLATION: no pre-tokenising regex (split only on newlines)")
    flat = BPE.train(train, 1024, pattern=r"[^\n]+|\n+")
    crossing = merges_crossing_words(flat)
    print(
        f"   {'':<12}{'chars/token':>12}{'tokens spanning a word boundary':>34}{'words tokenised the same in context':>38}"
    )
    print(
        f"   {'GPT-2 regex':<12}{chars_per_token(mine.encode, held):>12.2f}{len(merges_crossing_words(mine)):>34}{boundary_alignment(mine, held):>38.1%}"
    )
    print(
        f"   {'no regex':<12}{chars_per_token(flat.encode, held):>12.2f}{len(crossing):>34}{boundary_alignment(flat, held):>38.1%}"
    )
    print(
        "   tokens the no-regex vocabulary learned: "
        + ", ".join(repr(b.decode()) for b in crossing[:8])
    )

    print("\n4. ARTEFACTS: tokens for a few strings (GPT-2 | Qwen2.5 | mine)")
    for s in [" hello", "hello", "Hello", "1234567", "2024-03-01", "indentation\n        x = 1"]:
        row = [gpt2.token_strings(s), qwen.token_strings(s) if qpath else [], mine.token_strings(s)]
        print(f"   {s!r:<26}" + "   ".join(f"{len(r):>2} {r}" for r in row))
    print("\n   tokens per character (lower is cheaper):")
    samples = {
        "English": "The retry limit is 3 times with exponential backoff, and the limit can be raised to at most 5.",
        "Vietnamese": "Giới hạn thử lại là 3 lần với độ trễ tăng dần, và có thể tăng lên tối đa 5 lần.",
        "Japanese": "再試行の上限は3回で、指数バックオフを使います。最大5回まで増やせます。",
        "Python": "def retry(fn, n=3):\n    for i in range(n):\n        try:\n            return fn()\n        except Exception:\n            pass",
    }
    print(f"   {'':<12}" + "".join(f"{k:>16}" for k in refs))
    for lang, s in samples.items():
        print(f"   {lang:<12}" + "".join(f"{len(v(s)) / len(s):>16.2f}" for v in refs.values()))


if __name__ == "__main__":
    main(sys.argv)
