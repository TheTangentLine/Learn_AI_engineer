"""Evaluate a base or fine-tuned model on the extraction task and on the forgetting probes, with uncertainty and paired comparisons.

model = load_tuned(path)                                   # SmolLM2 + adapters, merged for speed
res = evaluate_sets(model, tok, {"human": HUMAN, "synthetic test": test})
table = compare(res_a["human"], res_b["human"])            # paired bootstrap on exact-match per email
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import chatfmt as C  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402
import orders as O  # noqa: E402
import probes as P  # noqa: E402

from common.abtest import wilson  # noqa: E402
from common.evalkit import paired_bootstrap  # noqa: E402


def load_tuned(ckpt: str | Path, base=None, *, merge: bool = True):
    """The base model plus the adapters saved in ``ckpt`` (a dict with ``state``, ``r``, ``alpha``, ``targets``), merged into the weights for speed."""
    model, tok = base if base is not None else I.load_base()
    data = torch.load(ckpt, weights_only=False)
    L.add_lora(model, r=data["r"], alpha=data["alpha"], targets=tuple(data["targets"]))
    L.load_lora_state_dict(model, data["state"])
    if merge:
        L.merge_lora(model)
    return model.eval(), tok


def records_to_pairs(recs: list[dict]) -> list[tuple[str, dict]]:
    """(email, gold) from a JSONL record: the gold is recovered by parsing the target JSON back into the Order fields."""
    out = []
    for r in recs:
        email, answer = r["messages"][1]["content"], r["messages"][-1]["content"]
        import json

        d = json.loads(answer)
        d["items"] = [(i["name"], i["quantity"]) for i in d["items"]]
        out.append((email, d))
    return out


def evaluate_sets(
    model,
    tok,
    sets: dict[str, list[tuple[str, dict]]],
    make=I.tuned_messages,
    max_new_tokens: int = 160,
) -> dict[str, list[I.Result]]:
    return {
        name: I.run_task(model, tok, items, make, max_new_tokens) for name, items in sets.items()
    }


def rate_with_ci(successes: int, n: int) -> tuple[float, float, float]:
    lo, hi = wilson(successes, n)
    return successes / n, lo, hi


def summarize_results(results: list[I.Result]) -> dict:
    s = O.summarize([r.score for r in results])
    n = len(results)
    s["exact_ci"] = rate_with_ci(sum(r.score.exact for r in results), n)
    s["valid_order_ci"] = rate_with_ci(sum(r.score.valid_order for r in results), n)
    s["prompt_tokens"] = statistics.fmean(r.prompt_tokens for r in results)
    s["new_tokens"] = statistics.fmean(r.new_tokens for r in results)
    s["seconds"] = statistics.fmean(r.seconds for r in results)
    return s


def compare(a: list[I.Result], b: list[I.Result], metric: str = "exact") -> dict:
    """Paired bootstrap on the per-email metric ('exact': 0/1; 'fields': fraction of the 8 fields right) of A against B on the same emails."""
    f = (lambda r: float(r.score.exact)) if metric == "exact" else (lambda r: r.score.n_correct / 8)
    return paired_bootstrap([f(r) for r in a], [f(r) for r in b])


def run_probes(model, tok, max_new_tokens: int = 40) -> dict:
    """Reply to every probe with the model's ordinary chat prompt; per-kind accuracy and the rate of order-JSON replies to unrelated prompts."""
    outcomes = []
    for p in P.PROBES:
        reply, *_ = I.complete(model, tok, [{"role": "user", "content": p.prompt}], max_new_tokens)
        outcomes.append((p, reply, p.passes(reply), P.emits_order_json(reply)))
    kinds = sorted({p.kind for p in P.PROBES})
    by_kind = {
        k: sum(ok for p, _, ok, _ in outcomes if p.kind == k)
        / sum(p.kind == k for p, *_ in outcomes)
        for k in kinds
    }
    return {
        "accuracy": sum(ok for _, _, ok, _ in outcomes) / len(outcomes),
        "by_kind": by_kind,
        "order_json_rate": sum(o for *_, o in outcomes) / len(outcomes),
        "outcomes": outcomes,
        "n": len(outcomes),
    }


@torch.no_grad()
def text_loss(model, tok, text: str, window: int = 128, max_tokens: int = 12000) -> float:
    """Language-model loss (nats/token) on ordinary prose, with no chat template: has the fine-tune damaged the model's grasp of general text?"""
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"][:max_tokens])
    n = (len(ids) - 1) // window * window
    x, y = ids[:n].view(-1, window), ids[1 : n + 1].view(-1, window)
    total = 0.0
    for i in range(0, len(x), 16):
        logits = model(x[i : i + 16])
        total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y[i : i + 16].reshape(-1), reduction="sum"
        ).item()
    return total / y.numel()


def error_types(results: list[I.Result]) -> dict[str, int]:
    """What kind of wrong: not JSON, invalid order, or a valid order with wrong fields (and which field is wrong most often)."""
    c = {"not_json": 0, "invalid_order": 0, "wrong_fields": 0, "exact": 0}
    per_field: dict[str, int] = {}
    for r in results:
        if not r.score.valid_json:
            c["not_json"] += 1
        elif not r.score.valid_order:
            c["invalid_order"] += 1
        elif r.score.exact:
            c["exact"] += 1
        else:
            c["wrong_fields"] += 1
            for f, ok in r.score.fields.items():
                if not ok:
                    per_field[f] = per_field.get(f, 0) + 1
    c.update({f"wrong:{k}": v for k, v in sorted(per_field.items(), key=lambda kv: -kv[1])})
    return c


def fmt_ci(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.0%} [{ci[1]:.0%}, {ci[2]:.0%}]"


def bytes_of(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


_ = (C, math)
