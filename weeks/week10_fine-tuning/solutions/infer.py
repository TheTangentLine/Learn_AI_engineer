"""Run a model on the extraction task: load SmolLM2 into the Week 9 decoder, generate greedily with the KV cache, score the replies.

model, tok = load_base()                              # SmolLM2-135M-Instruct in the from-scratch decoder
reply = complete(model, tok, messages)                # greedy, stops at <|im_end|>
results = run_task(model, tok, emails, make_messages) # list of (email, gold, reply, Score, prompt_tokens, new_tokens, seconds)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "weeks/week09_transformers-from-scratch/solutions"))
sys.path.insert(0, str(ROOT / "weeks/week09_transformers-from-scratch/solutions/weekly/fastgen"))

import blocks as B  # noqa: E402
import chatfmt as C  # noqa: E402
import orders as O  # noqa: E402
from generate import generate  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_base(name: str = BASE):
    """The Hugging Face checkpoint loaded into the from-scratch decoder (verified equal to the library's logits in the Week 9 tests)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    model = B.Decoder(B.config_from_hf(hf.config)).eval()
    B.load_hf_state_dict(model, hf.state_dict())
    del hf
    return model, tok


@dataclass
class Result:
    email: str
    gold: dict
    reply: str
    score: O.Score
    prompt_tokens: int
    new_tokens: int
    seconds: float


@torch.no_grad()
def complete(
    model: B.Decoder, tok, messages: list[dict], max_new_tokens: int = 160
) -> tuple[str, int, int, float]:
    """Greedy completion of a chat prompt. Returns (reply text without the end-of-turn token, prompt tokens, new tokens, seconds)."""
    prompt_ids = tok(C.render(messages, add_generation_prompt=True), add_special_tokens=False)[
        "input_ids"
    ]
    stop = {tok.convert_tokens_to_ids(C.IM_END)}
    t0 = time.perf_counter()
    gen = generate(model, prompt_ids, max_new_tokens, mode="cache", temperature=0.0, stop_ids=stop)
    dt = time.perf_counter() - t0
    ids = [t for t in gen.tokens if t not in stop]
    return tok.decode(ids), len(prompt_ids), len(gen.tokens), dt


def run_task(
    model,
    tok,
    examples: list[tuple[str, dict]],
    make_messages: Callable[[str], list[dict]],
    max_new_tokens: int = 160,
) -> list[Result]:
    out = []
    for email, gold in examples:
        reply, n_prompt, n_new, dt = complete(model, tok, make_messages(email), max_new_tokens)
        out.append(Result(email, gold, reply, O.score(reply, gold), n_prompt, n_new, dt))
    return out


# ----------------------------------------------------------------------------- prompts


def zero_shot_messages(email: str) -> list[dict]:
    """The prompt a Week 2 style pipeline would send: the full schema description in the system message."""
    return [
        {"role": "system", "content": O.SCHEMA_PROMPT},
        {"role": "user", "content": f"<email>\n{email}\n</email>"},
    ]


def few_shot_messages(shots: list[tuple[str, dict]]) -> Callable[[str], list[dict]]:
    def make(email: str) -> list[dict]:
        msgs = [{"role": "system", "content": O.SCHEMA_PROMPT}]
        for e, g in shots:
            msgs += [
                {"role": "user", "content": f"<email>\n{e}\n</email>"},
                {"role": "assistant", "content": O.order_json(g)},
            ]
        return msgs + [{"role": "user", "content": f"<email>\n{email}\n</email>"}]

    return make


def tuned_messages(email: str) -> list[dict]:
    """The short prompt a fine-tuned model needs: the task is in its weights, not in the prompt."""
    return [{"role": "system", "content": O.SYSTEM_SHORT}, {"role": "user", "content": email}]


def training_messages(email: str, gold: dict) -> list[dict]:
    return tuned_messages(email) + [{"role": "assistant", "content": O.order_json(gold)}]
