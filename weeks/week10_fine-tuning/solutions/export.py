"""Merging, exporting and publishing a fine-tuned model.

to_hf_state_dict(model)         the from-scratch decoder's weights under the names Hugging Face expects (the inverse of ``load_hf_state_dict``)
save_hf_model(model, tok, dir)  config + safetensors + tokenizer + model card: a directory ``AutoModelForCausalLM.from_pretrained`` can load
ollama_modelfile(...)           the text of an Ollama Modelfile for a ChatML model (NOT RUN: Ollama and llama.cpp are not installed here)
to_openai_jsonl / validate_...  the layout of a hosted fine-tuning job's training file and the rules its API enforces (NOT RUN: needs a key)
size_table(...)                 what the weights occupy in each precision, with the embedding table kept at fp16
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))

import blocks as B  # noqa: E402
import quant as Q  # noqa: E402


def to_hf_state_dict(model: B.Decoder) -> dict[str, torch.Tensor]:
    """``embed_tokens.weight`` -> ``model.embed_tokens.weight`` and so on. A tied output matrix is NOT written (the checkpoint stores the table once, and the
    config says ``tie_word_embeddings``), exactly as the original checkpoint does."""
    sd = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight":
            if model.cfg.tie_embeddings:
                continue
            sd[k] = v.detach().clone()
        else:
            sd[f"model.{k}"] = v.detach().clone()
    return sd


def model_card(base: str, metrics: dict[str, str], data_note: str, extra: str = "") -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    return f"""---
base_model: {base}
library_name: transformers
tags: [fine-tuned, lora-merged, order-extraction]
---
# Order extraction (fine-tuned {base.split("/")[-1]})

Turns a customer email into a JSON order (`is_order`, `customer_name`, `order_id`, `items`, `urgency`, `delivery_date`, `total_amount`, `currency`).
Trained with LoRA (merged into the weights) on synthetic emails; **evaluated on hand-written emails it had never seen**.

## Results
| measure | value |
|---|---|
{rows}

## Training data
{data_note}

## Prompt format
System message `Extract the order from the email as JSON.`, the email as the user message; ChatML template. Greedy decoding. Stops at `<|im_end|>`.

## Limitations
Trained on templated English emails; phrasing far from them is where it fails. Does not handle currencies other than USD, EUR and GBP. Not evaluated for bias or safety.
{extra}"""


def save_hf_model(
    model: B.Decoder, tok, out_dir: str | Path, base: str | object, card: str
) -> list[str]:
    """``base`` is the Hugging Face name of the model that was fine-tuned (its config is copied) or a config object."""
    from safetensors.torch import save_file
    from transformers import AutoConfig

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (AutoConfig.from_pretrained(base) if isinstance(base, str) else base).save_pretrained(out)
    tok.save_pretrained(out)
    save_file(
        {k: v.contiguous() for k, v in to_hf_state_dict(model).items()},
        out / "model.safetensors",
        metadata={"format": "pt"},
    )
    (out / "README.md").write_text(card)
    return sorted(p.name for p in out.iterdir())


def ollama_modelfile(
    model_path: str = "./order-extractor.gguf",
    system: str = "Extract the order from the email as JSON.",
    temperature: float = 0.0,
) -> str:
    """An Ollama Modelfile for a ChatML model. (NOT RUN: it needs the weights converted to GGUF with llama.cpp's converter and Ollama installed.)"""
    template = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>"
    return f'FROM {model_path}\nTEMPLATE """{template}"""\nSYSTEM """{system}"""\nPARAMETER temperature {temperature}\nPARAMETER stop "<|im_end|>"\nPARAMETER stop "<|im_start|>"\n'


# ----------------------------------------------------------------------------- hosted fine-tuning APIs (format and cost only)


def to_openai_jsonl(records: list[dict]) -> list[str]:
    """One JSON object per line: {"messages": [...]}: the chat fine-tuning format of OpenAI-style APIs (the other providers differ in details)."""
    return [json.dumps({"messages": r["messages"]}, ensure_ascii=False) for r in records]


def validate_openai_jsonl(
    lines: list[str], *, min_examples: int = 10, max_tokens_estimate: int = 65536
) -> list[str]:
    """The rules such a service enforces, checked before you pay for a failed upload: valid JSON per line, a messages list, known roles, content strings,
    at least one assistant message, a minimum number of examples. Token counts are estimated at 4 characters per token. Returns the list of problems."""
    problems = []
    if len(lines) < min_examples:
        problems.append(f"only {len(lines)} examples; at least {min_examples} are required")
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except ValueError:
            problems.append(f"line {i}: not valid JSON")
            continue
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            problems.append(f"line {i}: missing messages")
            continue
        if any(
            m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str)
            for m in msgs
        ):
            problems.append(f"line {i}: a message has an unknown role or non-string content")
        if not any(m.get("role") == "assistant" for m in msgs):
            problems.append(f"line {i}: no assistant message to learn from")
        if sum(len(m.get("content", "")) for m in msgs) / 4 > max_tokens_estimate:
            problems.append(f"line {i}: longer than the context limit")
    return problems


def estimate_training_cost(n_tokens: int, epochs: int, price_per_million: float) -> float:
    """Hosted fine-tuning is billed per training token: dataset tokens x epochs x price. (A price card you supply: none is built in.)"""
    return n_tokens * epochs * price_per_million / 1e6


# ----------------------------------------------------------------------------- sizes


def size_table(model: B.Decoder) -> list[dict]:
    """Bytes of the weights per format: the embedding table (and tied output matrix) stays at fp16; the transformer blocks use the format's bits per weight."""
    n_embed = model.embed_tokens.weight.numel()
    n_blocks = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embed_tokens" not in n and "norm" not in n and "lm_head" not in n
    )
    n_other = sum(p.numel() for n, p in model.named_parameters() if "norm" in n)
    rows = []
    for fmt in ("fp32", "fp16", "q8_0", "q4_0", "nf4"):
        bits = Q.BITS_PER_WEIGHT[fmt]
        nbytes = (
            n_blocks * bits
            + n_embed * (32 if fmt == "fp32" else 16)
            + n_other * (32 if fmt == "fp32" else 16)
        ) / 8
        rows.append({"format": fmt, "bytes": nbytes, "bits_per_block_weight": bits})
    return rows


def audit_directory(path: str | Path) -> list[str]:
    """Checks before uploading a model directory: the files a loader needs exist, the model card has its front matter and results, and nothing in the text
    files looks like a credential."""
    p = Path(path)
    problems = [
        f"missing {name}"
        for name in ("config.json", "model.safetensors", "tokenizer.json", "README.md")
        if not (p / name).exists()
    ]
    card = (p / "README.md").read_text() if (p / "README.md").exists() else ""
    if not card.startswith("---") or "base_model:" not in card:
        problems.append("the model card has no base_model front matter")
    if "| measure |" not in card:
        problems.append("the model card reports no results")
    secret = re.compile(r"(?:hf_|sk-|ghp_|AKIA)[A-Za-z0-9]{16,}")
    for f in p.glob("*"):
        if f.suffix in (".json", ".md", ".txt") and secret.search(f.read_text(errors="ignore")):
            problems.append(f"{f.name} looks like it contains a credential")
    return problems
