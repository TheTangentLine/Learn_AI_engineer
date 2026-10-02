"""Byte-level BPE from scratch: a trainer, an encoder, a decoder, and loaders for the vocabularies of real tokenizers.

    tok = BPE.train(text, vocab_size=1024)               # learn merges from a corpus
    ids = tok.encode("Hello, world!")                    # text -> token ids
    tok.decode(ids) == "Hello, world!"                   # always: byte-level BPE can encode ANY string
    ref = BPE.from_tiktoken("gpt2")                      # the same encoder with GPT-2's vocabulary: its ids match tiktoken's

What is the same in every GPT-style tokenizer (and what this file implements):
  1. PRE-TOKENISE   a regular expression cuts the text into chunks (a word with its leading space, a run of digits, punctuation, whitespace);
                    merges never cross a chunk boundary, which is why " the" is one token and "the dog" is never merged into one
  2. BYTES          each chunk becomes its UTF-8 bytes: the base vocabulary is the 256 byte values, so nothing is ever "unknown"
  3. MERGES         a vocabulary is a table  token bytes -> rank.  A chunk is encoded by repeatedly merging the adjacent pair whose
                    concatenation has the LOWEST rank, until no adjacent pair is in the table. Rank = merge order = token id.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import regex

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT = "<|endoftext|>"


class BPE:
    def __init__(
        self,
        ranks: dict[bytes, int],
        pattern: str = GPT2_PATTERN,
        specials: dict[str, int] | None = None,
    ):
        if any(len(b) == 0 for b in ranks):
            raise ValueError("the empty byte string is not a token")
        singles = {b for b in ranks if len(b) == 1}
        if len(singles) != 256:
            raise ValueError(
                "a byte-level vocabulary must contain all 256 single bytes (otherwise some text cannot be encoded)"
            )
        ids = list(ranks.values())
        if len(set(ids)) != len(ids):
            raise ValueError("two tokens share an id")
        self.ranks, self.pattern, self.specials = dict(ranks), pattern, dict(specials or {})
        self._re = regex.compile(pattern)
        self._decoder = {i: b for b, i in ranks.items()} | {
            i: s.encode() for s, i in self.specials.items()
        }
        if len(self._decoder) != len(ranks) + len(self.specials):
            raise ValueError("a special token shares an id with an ordinary token")
        self._special_re = (
            regex.compile(
                "|".join(regex.escape(s) for s in sorted(self.specials, key=len, reverse=True))
            )
            if self.specials
            else None
        )
        self._chunk = lru_cache(maxsize=65536)(self._encode_chunk_uncached)

    @property
    def vocab_size(self) -> int:
        return len(self._decoder)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk_uncached(self, chunk: bytes) -> tuple[int, ...]:
        if chunk in self.ranks:  # the whole chunk is one token
            return (self.ranks[chunk],)
        parts = [bytes([c]) for c in chunk]
        while len(parts) > 1:
            best_rank, best_pair = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get(parts[i] + parts[i + 1])
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, (parts[i], parts[i + 1])
            if best_pair is None:
                break
            merged: list[bytes] = []
            i = 0
            while i < len(
                parts
            ):  # merge every non-overlapping occurrence of that pair, left to right
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return tuple(self.ranks[p] for p in parts)

    def encode(
        self, text: str, allowed_special: set[str] | frozenset[str] = frozenset()
    ) -> list[int]:
        """Text to ids. A special token's text inside ``text`` is an error unless it is allowed: otherwise a user could type ``<|endoftext|>``
        and inject a control token (the same rule tiktoken applies)."""
        if not self.specials:
            return self._encode_ordinary(text)
        out: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.group() not in allowed_special:
                raise ValueError(
                    f"the text contains the special token {m.group()!r}, which is not allowed"
                )
            out += self._encode_ordinary(text[pos : m.start()])
            out.append(self.specials[m.group()])
            pos = m.end()
        return out + self._encode_ordinary(text[pos:])

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for m in self._re.finditer(text):
            out.extend(self._chunk(m.group().encode("utf-8")))
        return out

    # ------------------------------------------------------------------ decoding
    def decode_bytes(self, ids: list[int]) -> bytes:
        try:
            return b"".join(self._decoder[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id {exc.args[0]}") from None

    def decode(self, ids: list[int]) -> str:
        """Bytes to text. A model can emit ids that stop in the middle of a character; ``errors='replace'`` shows U+FFFD instead of crashing."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_strings(self, text: str) -> list[str]:
        """The tokens of ``text`` as readable strings (partial characters shown as their byte escapes)."""
        return [
            self._decoder[i].decode("utf-8", errors="backslashreplace") for i in self.encode(text)
        ]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        pattern: str = GPT2_PATTERN,
        specials: list[str] | None = None,
    ) -> BPE:
        """Learn ``vocab_size - 256 - len(specials)`` merges. Each step merges the most frequent adjacent pair over the whole corpus (counted
        once per distinct chunk, weighted by how often it occurs); ties go to the numerically smaller pair so a run is deterministic."""
        return cls._train(text, vocab_size, pattern, specials)[0]

    @classmethod
    def _train(cls, text: str, vocab_size: int, pattern: str, specials: list[str] | None):
        n_special = len(specials or [])
        if vocab_size < 256 + n_special:
            raise ValueError("the vocabulary must hold the 256 bytes and the special tokens")
        counts = Counter(m.group().encode("utf-8") for m in regex.finditer(pattern, text))
        chunks = list(counts)
        words = [list(c) for c in chunks]  # each chunk as a list of token ids (initially its bytes)
        freqs = [counts[c] for c in chunks]
        token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        ranks: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:], strict=False):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)
        next_id = 256
        while next_id < vocab_size - n_special:
            candidates = ((c, p) for p, c in pair_counts.items() if c > 0)
            best = max(candidates, key=lambda cp: (cp[0], -cp[1][0], -cp[1][1]), default=None)
            if best is None:
                break
            _, (a, b) = best
            merged = token_bytes[a] + token_bytes[b]
            # merges are applied globally in order, so two different splits of one string cannot both exist: `merged` is always a new token
            ranks[merged] = token_bytes_id = next_id
            token_bytes[token_bytes_id] = merged
            next_id += 1
            for wi in list(where[(a, b)]):
                w, f = words[wi], freqs[wi]
                for pair in zip(
                    w, w[1:], strict=False
                ):  # take the word's old pairs out of the counts ...
                    pair_counts[pair] -= f
                    where[pair].discard(wi)
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(token_bytes_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for pair in zip(out, out[1:], strict=False):  # ... and put the new ones in
                    pair_counts[pair] += f
                    where[pair].add(wi)
            pair_counts = Counter({p: c for p, c in pair_counts.items() if c > 0})
        specials_map = {s: next_id + k for k, s in enumerate(specials or [])}
        tok = cls(ranks, pattern, specials_map)
        tok.training_segmentation = {chunks[wi]: tuple(words[wi]) for wi in range(len(chunks))}
        return tok, chunks, words

    # ------------------------------------------------------------------ persistence and real vocabularies
    def save(self, path: str | Path) -> None:
        data = {
            "pattern": self.pattern,
            "tokens": [[b.hex(), i] for b, i in sorted(self.ranks.items(), key=lambda kv: kv[1])],
            "specials": self.specials,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> BPE:
        d = json.loads(Path(path).read_text())
        return cls({bytes.fromhex(h): i for h, i in d["tokens"]}, d["pattern"], d["specials"])

    @classmethod
    def from_tiktoken(cls, name: str = "gpt2") -> BPE:
        """The same algorithm with an existing vocabulary, to check this encoder against the reference implementation."""
        import tiktoken

        enc = tiktoken.get_encoding(name)
        return cls(dict(enc._mergeable_ranks), enc._pat_str, dict(enc._special_tokens))

    @classmethod
    def from_hf_tokenizer_json(cls, path: str | Path) -> BPE:
        """A Hugging Face byte-level BPE (GPT-2 style, such as Qwen's): its vocabulary uses a printable stand-in for every byte."""
        data = json.loads(Path(path).read_text())
        to_byte = {c: b for b, c in _bytes_to_unicode().items()}
        ranks = {bytes(to_byte[ch] for ch in tok): i for tok, i in data["model"]["vocab"].items()}
        pre = data["pre_tokenizer"]
        pattern = (
            next(p["pattern"]["Regex"] for p in pre["pretokenizers"] if p["type"] == "Split")
            if pre["type"] == "Sequence"
            else pre["pattern"]["Regex"]
        )
        specials = {t["content"]: t["id"] for t in data["added_tokens"] if t.get("special")}
        return cls(ranks, pattern, specials)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's table: every byte gets a printable character (so vocab files can be text). Printable bytes map to themselves; the rest are
    shifted into U+0100 and up: space becomes 'Ġ', newline 'Ċ'."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars, extra = printable[:], 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars, strict=True)}
