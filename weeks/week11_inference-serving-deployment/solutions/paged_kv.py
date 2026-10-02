"""PagedAttention's memory manager, from scratch: the KV cache as fixed-size BLOCKS handed out on demand, with a block table per sequence.

The problem it solves (Week 9: a KV cache is the biggest thing in GPU memory when serving). A naive server reserves a contiguous buffer of ``max_len`` tokens for every
request, because it cannot know how long the answer will be. Most requests are far shorter, so most of the reserved memory is wasted ("internal fragmentation"), and
the freed holes are the wrong size for the next request ("external fragmentation"). Paging fixes both: memory is cut into blocks of ``block_size`` tokens, a sequence
owns a list of blocks (its block table) that need not be adjacent, and a block is allocated only when the sequence actually grows into it.

    pool = BlockPool(num_blocks=1024, block_size=16)
    seq = pool.new_sequence()
    pool.append(seq, n_tokens=40)                  # allocates ceil(40 / 16) = 3 blocks
    child = pool.fork(seq)                         # shares those blocks (reference counted); a write to a shared block copies it first
    pool.free(seq)                                 # blocks still referenced by `child` stay allocated

``paged_attention_decode`` shows the other half: attention that reads K and V THROUGH the block table, equal to attention over a contiguous cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


class OutOfBlocks(RuntimeError):
    """No free block is left: the scheduler must preempt or queue a sequence."""


@dataclass
class Sequence:
    id: int
    blocks: list[int] = field(default_factory=list)
    length: int = 0  # tokens stored


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks, self.block_size = num_blocks, block_size
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() hands out block 0 first
        self._ref = [0] * num_blocks
        self._seqs: dict[int, Sequence] = {}
        self._next_id = 0
        self.copies = 0  # copy-on-write events

    # -- accounting
    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def blocks_needed(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_size)

    def can_fit(self, tokens: int, extra_for: int = 0) -> bool:
        """Would a NEW sequence of ``tokens`` fit, keeping ``extra_for`` blocks in reserve for running sequences to grow?"""
        return self.blocks_needed(tokens) + extra_for <= self.free_blocks

    # -- sequences
    def new_sequence(self) -> Sequence:
        seq = Sequence(self._next_id)
        self._next_id += 1
        self._seqs[seq.id] = seq
        return seq

    def _take(self) -> int:
        if not self._free:
            raise OutOfBlocks("the block pool is exhausted")
        b = self._free.pop()
        self._ref[b] = 1
        return b

    def append(self, seq: Sequence, n_tokens: int = 1) -> list[int]:
        """Store ``n_tokens`` more tokens; returns the physical blocks that were newly allocated (or copied). If the sequence's last block is shared with another
        sequence it is COPIED before being written (copy-on-write). All-or-nothing: if the pool runs out, nothing is changed."""
        if n_tokens < 1:
            raise ValueError("n_tokens must be at least 1")
        if seq.id not in self._seqs:
            raise KeyError("unknown or freed sequence")
        bs = self.block_size
        room = (-seq.length) % bs  # free slots in the last block
        cow = 1 if room and seq.blocks and self._ref[seq.blocks[-1]] > 1 else 0
        new_blocks = max(0, math.ceil((n_tokens - room) / bs)) if n_tokens > room else 0
        if cow + new_blocks > self.free_blocks:
            raise OutOfBlocks(f"need {cow + new_blocks} blocks, {self.free_blocks} free")
        allocated: list[int] = []
        if cow:
            old = seq.blocks[-1]
            self._ref[old] -= 1
            seq.blocks[-1] = self._take()
            self.copies += 1
            allocated.append(seq.blocks[-1])
        for _ in range(new_blocks):
            b = self._take()
            seq.blocks.append(b)
            allocated.append(b)
        seq.length += n_tokens
        return allocated

    def fork(self, seq: Sequence) -> Sequence:
        """A child that shares every block of ``seq`` (same prompt, different continuations: beam search, parallel sampling, a shared system prompt)."""
        child = self.new_sequence()
        child.blocks, child.length = list(seq.blocks), seq.length
        for b in child.blocks:
            self._ref[b] += 1
        return child

    def free(self, seq: Sequence) -> None:
        if self._seqs.pop(seq.id, None) is None:
            raise KeyError("unknown or already freed sequence")
        for b in seq.blocks:
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)
        seq.blocks, seq.length = [], 0

    def refcount(self, block: int) -> int:
        return self._ref[block]

    # -- memory efficiency
    def wasted_slots(self) -> int:
        """Reserved but empty token slots: only ever the unused tail of each sequence's last block (internal fragmentation of paging)."""
        return sum(
            len(s.blocks) * self.block_size - s.length for s in self._seqs.values() if s.blocks
        )

    def check(self) -> None:
        """Invariants: every block is free xor referenced; reference counts equal the number of block tables containing the block; no duplicates in the free list."""
        counts = [0] * self.num_blocks
        for s in self._seqs.values():
            for b in s.blocks:
                counts[b] += 1
        free = set(self._free)
        assert len(free) == len(self._free), "duplicate in the free list"
        for b in range(self.num_blocks):
            assert counts[b] == self._ref[b], (
                f"block {b}: ref {self._ref[b]} but {counts[b]} owners"
            )
            assert (b in free) == (self._ref[b] == 0), f"block {b}: free-list and refcount disagree"


def contiguous_waste(lengths: list[int], max_len: int) -> float:
    """Share of reserved token slots that are empty when every request gets a contiguous ``max_len`` buffer."""
    return 1 - sum(lengths) / (len(lengths) * max_len)


def paged_waste(lengths: list[int], block_size: int) -> float:
    """Share of reserved slots that are empty with paging: only the partly-filled last block of each request."""
    reserved = sum(math.ceil(n / block_size) * block_size for n in lengths)
    return 1 - sum(lengths) / reserved


# ----------------------------------------------------------------------------- attention through a block table


def paged_attention_decode(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, block_table: list[int], length: int
) -> torch.Tensor:
    """One decoding step for ONE sequence: ``q`` (H, d) attends over the ``length`` cached tokens that live in the physical blocks of ``block_table``.

    k_pool, v_pool: (num_blocks, block_size, H, d), the physical memory. The logical token t is in block ``block_table[t // block_size]`` at slot ``t % block_size``.
    """
    bs = k_pool.shape[1]
    idx = torch.arange(length)
    blocks = torch.tensor(block_table)[idx // bs]
    k = k_pool[blocks, idx % bs]  # (length, H, d): a gather through the table
    v = v_pool[blocks, idx % bs]
    scores = torch.einsum("hd,thd->ht", q, k) / math.sqrt(q.shape[-1])
    return torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), v)
