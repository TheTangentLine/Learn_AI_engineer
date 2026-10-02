"""Chat templates, training examples with a loss mask, and the common dataset formats.

A chat model is an ordinary language model that was trained on text with a fixed layout. SmolLM2 (like Qwen and many others) uses ChatML:

    <|im_start|>system\\nYou are ...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>\\n

Fine-tuning has to reproduce that layout EXACTLY (a wrong template trains the model on a format it will never see at inference) and must compute the
loss only on the tokens the model is meant to produce: the assistant's answer and the end-of-turn token that teaches it to stop.

    text = render(messages, add_generation_prompt=True)            # == tokenizer.apply_chat_template(...)
    ex = encode_example(messages, tok, max_len=512)                # ex.input_ids, ex.labels (-100 on everything but the answer)
    batch = collate([ex1, ex2], pad_id)                            # right-padded tensors
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
IGNORE = -100  # the label cross_entropy skips
ROLES = ("system", "user", "assistant")


def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """The ChatML template of SmolLM2, re-implemented (it is a Jinja template in tokenizer_config.json): a missing system message is filled in with the
    model's default one; every message is ``<|im_start|>role\\ncontent<|im_end|>\\n``."""
    out = []
    if messages and messages[0]["role"] != "system":
        out.append(f"{IM_START}system\n{DEFAULT_SYSTEM}{IM_END}\n")
    for m in messages:
        out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


def validate_messages(messages: list[dict], *, require_assistant_last: bool = True) -> None:
    """Reject conversations that would silently train on garbage: unknown roles, empty content, a system message that is not first, two turns by the same
    role in a row, or (for SFT) no final assistant answer to learn from."""
    if not messages:
        raise ValueError("no messages")
    for i, m in enumerate(messages):
        if set(m) - {"role", "content"} or "role" not in m or "content" not in m:
            raise ValueError(f"message {i} must have exactly 'role' and 'content'")
        if m["role"] not in ROLES:
            raise ValueError(f"message {i}: unknown role {m['role']!r}")
        if not isinstance(m["content"], str) or not m["content"].strip():
            raise ValueError(f"message {i}: empty content")
        if m["role"] == "system" and i != 0:
            raise ValueError("a system message must come first")
        if m["role"] == "assistant" and m["content"][0].isspace():
            # the header ends with a newline and BPE would fuse it with the answer's leading whitespace into one token that straddles the boundary:
            # training would see a tokenisation the model never produces at inference (where the prompt ends after the newline)
            raise ValueError(f"message {i}: an assistant answer must not start with whitespace")
        if IM_START in m["content"] or IM_END in m["content"]:
            raise ValueError(
                f"message {i}: the content contains a control token (it would break the template)"
            )
    turns = [m["role"] for m in messages if m["role"] != "system"]
    for a, b in zip(turns, turns[1:], strict=False):
        if a == b:
            raise ValueError("turns must alternate between user and assistant")
    if turns and turns[0] != "user":
        raise ValueError("the first turn after the system message must be the user's")
    if require_assistant_last and messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant's answer")


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    n_answer: int  # tokens the loss is computed on
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.input_ids)


def trainable_spans(
    messages: list[dict], train_on: str = "last"
) -> tuple[str, list[tuple[int, int]]]:
    """The rendered text and the character ranges that count for the loss: each selected assistant message's content plus its <|im_end|> (but not
    the header before it, nor the newline after it)."""
    text, spans, pos = "", [], 0
    if messages and messages[0]["role"] != "system":
        text += f"{IM_START}system\n{DEFAULT_SYSTEM}{IM_END}\n"
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for i, m in enumerate(messages):
        header = f"{IM_START}{m['role']}\n"
        text += header
        start = len(text)
        text += m["content"] + IM_END
        if m["role"] == "assistant" and (train_on == "all" or i == last):
            spans.append((start, len(text)))
        text += "\n"
        pos = len(text)
    _ = pos
    return text, spans


def encode_example(
    messages: list[dict], tok, max_len: int = 512, train_on: str = "last"
) -> Example:
    """Tokenise the WHOLE rendered conversation once (so the ids are exactly what the model sees at inference) and mark the answer tokens using the
    tokenizer's character offsets. If the conversation is longer than ``max_len`` the PROMPT is cut from the left, never the answer; an answer that
    does not fit raises (silently training on half an answer teaches the model to stop mid-JSON)."""
    validate_messages(messages)
    text, spans = trainable_spans(messages, train_on)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    labels = [
        ids[k]
        if any(a <= s < b for a, b in spans for s in [offsets[k][0]])
        and offsets[k][1] > offsets[k][0]
        else IGNORE
        for k in range(len(ids))
    ]
    n_answer = sum(1 for lab in labels if lab != IGNORE)
    truncated = False
    if len(ids) > max_len:
        first = next(k for k, lab in enumerate(labels) if lab != IGNORE)
        if len(ids) - first > max_len:
            raise ValueError(
                f"the answer alone ({len(ids) - first} tokens) does not fit in max_len={max_len}"
            )
        ids, labels, truncated = ids[-max_len:], labels[-max_len:], True
    return Example(ids, labels, n_answer, truncated)


def collate(examples: list[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-padded batch. Padding on the right is safe with a causal model: a real token never attends to a later (padding) position."""
    n = max(len(e) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), n), IGNORE, dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e)] = torch.tensor(e.input_ids)
        labels[i, : len(e)] = torch.tensor(e.labels)
    return {"input_ids": ids, "labels": labels}


def shift_for_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position t predicts token t+1: drop the last logit and the first label."""
    return logits[:, :-1], labels[:, 1:]


# ----------------------------------------------------------------------------- the common dataset formats


def to_alpaca(messages: list[dict]) -> dict:
    """{"instruction", "input", "output"} (Stanford Alpaca): a single-turn task; the system message becomes the instruction, the user message the input."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    answer = next(m["content"] for m in messages if m["role"] == "assistant")
    return {"instruction": system, "input": user, "output": answer}


def from_alpaca(rec: dict) -> list[dict]:
    msgs = [{"role": "system", "content": rec["instruction"]}] if rec.get("instruction") else []
    return msgs + [
        {"role": "user", "content": rec["input"]},
        {"role": "assistant", "content": rec["output"]},
    ]


def to_sharegpt(messages: list[dict]) -> dict:
    """{"conversations": [{"from": "system|human|gpt", "value": ...}]} (ShareGPT / Vicuna)."""
    name = {"system": "system", "user": "human", "assistant": "gpt"}
    return {"conversations": [{"from": name[m["role"]], "value": m["content"]} for m in messages]}


def from_sharegpt(rec: dict) -> list[dict]:
    role = {"system": "system", "human": "user", "gpt": "assistant"}
    return [{"role": role[c["from"]], "content": c["value"]} for c in rec["conversations"]]


def to_prompt_completion(messages: list[dict]) -> dict:
    """{"prompt", "completion"}: the rendered prompt up to the assistant header and the answer: the layout of completion-only SFT."""
    return {
        "prompt": render(messages[:-1], add_generation_prompt=True),
        "completion": messages[-1]["content"] + IM_END,
    }


def to_preference(prompt_messages: list[dict], chosen: str, rejected: str) -> dict:
    """{"prompt": [...messages], "chosen": str, "rejected": str}: the layout of DPO (Day 5)."""
    if chosen == rejected:
        raise ValueError("chosen and rejected are identical: the pair teaches nothing")
    return {"prompt": prompt_messages, "chosen": chosen, "rejected": rejected}


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataset_stats(examples: list[Example]) -> dict:
    """What to look at before training: length distribution, how much of each example the loss sees, truncation."""
    lengths = sorted(len(e) for e in examples)
    q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]  # noqa: E731
    total, answer = sum(lengths), sum(e.n_answer for e in examples)
    return {
        "n": len(examples),
        "tokens": total,
        "answer_tokens": answer,
        "answer_fraction": answer / total,
        "p50": q(0.5),
        "p95": q(0.95),
        "max": lengths[-1],
        "truncated": sum(e.truncated for e in examples),
    }
