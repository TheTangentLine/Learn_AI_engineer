"""Week 10 weekly challenge: fine-tune a small model to beat the prompted base on the Week 2 extraction task, at a fraction of the cost.

  uv run python weeks/week10_fine-tuning/solutions/weekly/extract_ft/run_weekly.py [--epochs 3] [--quick] [--retrain]

Data (Day 2) -> LoRA SFT (Day 3) -> evaluation against the prompted baselines (Day 4) -> cost model -> ``outputs/w10_report.md``.
Evaluations are cached (replies are stored, scores are recomputed), so re-running the report does not re-generate; ``--quick`` is a smoke test on a handful of items.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parents[1]
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SOLUTIONS))

import chatfmt as C  # noqa: E402
import costs as K  # noqa: E402
import day2_solution as d2  # noqa: E402
import evalrun as E  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402
import orders as O  # noqa: E402
import report as R  # noqa: E402
import train_sft as T  # noqa: E402

OUT = ROOT / "outputs"
CACHE = OUT / "w10_weekly"
HOSTED_CARD = K.PriceCard(
    input_per_m=0.5,
    output_per_m=1.5,
    label="assumed hosted-style card: $0.50 / $1.50 per million input / output tokens",
)
VM_DOLLARS_PER_HOUR = 0.2  # an assumed price for a small CPU virtual machine


def key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def cached_run(
    name: str, model, tok, items: list[tuple[str, dict]], make, tag: str
) -> list[I.Result]:
    """Generate replies once per (system, email); a re-run re-scores the stored replies without generating."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key(name, tag)}.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    results = []
    for email, gold in items:
        k = key(email)
        if k not in store:
            reply, n_prompt, n_new, dt = I.complete(model, tok, make(email))
            store[k] = {
                "reply": reply,
                "prompt_tokens": n_prompt,
                "new_tokens": n_new,
                "seconds": dt,
            }
            path.write_text(json.dumps(store))
        r = store[k]
        results.append(
            I.Result(
                email,
                gold,
                r["reply"],
                O.score(r["reply"], gold),
                r["prompt_tokens"],
                r["new_tokens"],
                r["seconds"],
            )
        )
    return results


def train_if_needed(tok, epochs: int, limit: int | None, retrain: bool, tag: str) -> Path:
    path = OUT / "w10_ckpt" / f"{tag}_epoch{epochs}.pt"
    if path.exists() and not retrain:
        return path
    model, _ = I.load_base()
    data = OUT / "w10_data"
    train = [
        C.encode_example(r["messages"], tok) for r in C.read_jsonl(data / "train.jsonl")[:limit]
    ]
    dev = [
        C.encode_example(r["messages"], tok)
        for r in C.read_jsonl(data / "dev.jsonl")[: 20 if limit else None]
    ]
    L.add_lora(model, r=16, alpha=32, dropout=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save(epoch: int, run: T.SFTRun) -> None:
        import torch

        torch.save(
            {
                "state": L.lora_state_dict(model),
                "r": 16,
                "alpha": 32,
                "targets": list(L.DEFAULT_TARGETS),
                "epoch": epoch,
            },
            OUT / "w10_ckpt" / f"{tag}_epoch{epoch}.pt",
        )

    run = T.train_sft(
        model,
        train,
        dev,
        T.SFTConfig(epochs=epochs),
        tok.pad_token_id,
        log=lambda s: print("   " + s, flush=True),
        on_epoch=save,
    )
    (OUT / "w10_ckpt" / f"{tag}_run.json").write_text(
        json.dumps(
            {
                "tokens": run.tokens_seen,
                "seconds": run.seconds,
                "epochs": epochs,
                "examples": len(train),
            }
        )
    )
    return path


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else (1 if quick else 3)
    tag = "weekly_quick" if quick else "sft"
    if not (OUT / "w10_data/train.jsonl").exists():
        d2.main(["--out", str(OUT / "w10_data")])
    base, tok = I.load_base()
    ckpt = train_if_needed(tok, epochs, 48 if quick else None, "--retrain" in argv, tag)
    n_h, n_s = (6, 6) if quick else (len(O.HUMAN_EMAILS), 100)
    human = O.HUMAN_EMAILS[:n_h]
    synth = E.records_to_pairs(C.read_jsonl(OUT / "w10_data/test.jsonl"))[:n_s]
    sets = {"human": human, "synthetic": synth}
    suffix = "quick" if quick else "full"
    three_shot = I.few_shot_messages(O.FEW_SHOT_EXAMPLES)
    run: dict[str, dict] = {}
    for name, make, mdl in (
        ("prompted base, zero-shot", I.zero_shot_messages, base),
        ("prompted base, 3-shot", three_shot, base),
    ):
        run[name] = {
            s: cached_run(name, mdl, tok, items, make, f"{suffix}-{s}") for s, items in sets.items()
        }
    tuned, _ = E.load_tuned(ckpt)
    name = "fine-tuned (LoRA, merged)"
    run[name] = {
        s: cached_run(name, tuned, tok, items, I.tuned_messages, f"{suffix}-{ckpt.name}-{s}")
        for s, items in sets.items()
    }

    summaries = {n: {s: E.summarize_results(r) for s, r in res.items()} for n, res in run.items()}
    best_prompt = max(
        ("prompted base, zero-shot", "prompted base, 3-shot"),
        key=lambda n: summaries[n]["human"]["field_accuracy"],
    )
    ex, fl = (
        E.compare(run[name]["human"], run[best_prompt]["human"], "exact"),
        E.compare(run[name]["human"], run[best_prompt]["human"], "fields"),
    )
    costs = {}
    for n, s in summaries.items():
        h = s["human"]
        costs[n] = {
            "prompt_tokens": h["prompt_tokens"],
            "new_tokens": h["new_tokens"],
            "hosted": HOSTED_CARD.per_1000(h["prompt_tokens"], h["new_tokens"]),
            "self_hosted": K.compute_cost_per_1000(h["seconds"], VM_DOLLARS_PER_HOUR),
        }
    meta_path = OUT / "w10_ckpt" / f"{tag}_run.json"
    meta = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else {"tokens": 0, "seconds": 0, "epochs": epochs, "examples": 0}
    )
    one_off = K.training_cost(meta["tokens"], epochs, meta["seconds"], VM_DOLLARS_PER_HOUR)
    saving = costs[best_prompt]["hosted"] / 1000 - costs[name]["hosted"] / 1000
    probes = (
        E.run_probes(tuned, tok)
        if not quick
        else {"accuracy": float("nan"), "order_json_rate": float("nan")}
    )
    base_probes = (
        E.run_probes(base, tok)
        if not quick
        else {"accuracy": float("nan"), "order_json_rate": float("nan")}
    )
    m = {
        "task": "turn a customer email into a typed Order (8 fields) as JSON (the Week 2 Day 3 schema).",
        "model": f"{I.BASE} (135M parameters) in the Week 9 decoder",
        "training": f"LoRA r=16 on all linear layers, {meta['examples']} synthetic examples, {epochs} epochs, {meta['tokens']:,} tokens in {meta['seconds']:.0f} s on a laptop CPU.",
        "systems": summaries,
        "compare": {
            "n": ex["n"],
            "diff_exact": ex["diff"],
            "exact_low": ex["ci_low"],
            "exact_high": ex["ci_high"],
            "p_exact": ex["p"],
            "wins": ex["wins"],
            "losses": ex["losses"],
            "ties": ex["ties"],
            "diff_fields": fl["diff"],
            "fields_low": fl["ci_low"],
            "fields_high": fl["ci_high"],
        },
        "costs": costs,
        "break_even": {"one_off": one_off, "requests": K.break_even_requests(one_off, saving)},
        "forgetting": f"36 unrelated probe prompts: base {base_probes['accuracy']:.0%} correct, fine-tuned {probes['accuracy']:.0%}; replies to unrelated prompts that are an order object: base {base_probes['order_json_rate']:.0%}, fine-tuned {probes['order_json_rate']:.0%}.",
        "assumptions": {
            "card": HOSTED_CARD.label,
            "list": [
                f"hosted-style price card: {HOSTED_CARD.label}",
                f"self-hosting: a small CPU VM at an assumed ${VM_DOLLARS_PER_HOUR}/hour, at full utilisation",
                "training cost: the measured wall-clock seconds on this laptop priced at the same VM rate",
                "latency is the measured CPU time on this machine (Apple M2, 4 threads, float32)",
            ],
        },
        "not_run": [
            "a hosted frontier model, prompted or fine-tuned (no API key was used): its accuracy and real price are unknown",
            "GPU training, QLoRA, Unsloth/TRL (the LoRA and the loop are written from scratch and verified against PEFT)",
            "Ollama/GGUF (not installed)",
        ],
        "limits": [
            "38 hand-written emails give wide intervals; a difference of a few emails is noise",
            "the training data come from templates; accuracy on phrasing unlike them is what the hand-written set measures",
            "one training run, one seed",
            "a 135M model: conclusions about larger models are not supported",
        ],
    }
    text = R.build_report(m)
    out = OUT / ("w10_report_quick.md" if quick else "w10_report.md")
    out.write_text(text)
    print(text)
    print(f"\n[written to {out.relative_to(ROOT)}]")


if __name__ == "__main__":
    main(sys.argv)
