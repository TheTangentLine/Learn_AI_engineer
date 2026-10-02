"""Week 10 Day 4 - Solution: evaluating the fine-tune. Base against tuned, in distribution and out, overfitting, forgetting.

Systems compared (all SmolLM2-135M-Instruct, greedy decoding, evaluated on the SAME emails):
  base, 3-shot prompt      the best prompted baseline of Day 1 (593 prompt tokens)
  LoRA after epoch 1/2/3   the Day 3 checkpoints, with the short prompt (about 85 tokens)
  a hosted frontier model  NOT RUN (no API key was used this week); the column is there so a report has a place for it

Sets: 38 HAND-WRITTEN emails (unseen phrasing: the honest test) and 100 held-out SYNTHETIC emails (same generator as training: the easy test).

  uv run python weeks/week10_fine-tuning/solutions/day4_solution.py [--ckpt-tag sft] [--n-synth 100]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import chatfmt as C  # noqa: E402
import evalrun as E  # noqa: E402
import infer as I  # noqa: E402
import orders as O  # noqa: E402

from common.corpus import load_course_docs  # noqa: E402

CKPT = ROOT / "outputs/w10_ckpt"
DATA = ROOT / "outputs/w10_data"


def prose(n_chars: int = 60000) -> str:
    docs = [d for d in load_course_docs() if d.short.startswith(("week01", "week02", "week03"))]
    return "\n\n".join(d.text for d in docs)[:n_chars]


def main(argv: list[str]) -> None:
    tag = argv[argv.index("--ckpt-tag") + 1] if "--ckpt-tag" in argv else "sft"
    n_synth = int(argv[argv.index("--n-synth") + 1]) if "--n-synth" in argv else 100
    human = O.HUMAN_EMAILS
    synth = E.records_to_pairs(C.read_jsonl(DATA / "test.jsonl"))[:n_synth]
    text = prose()
    base = I.load_base()
    systems: dict[str, dict] = {}

    print("evaluating base model with a 3-shot prompt ...", file=sys.stderr, flush=True)
    three_shot = I.few_shot_messages(O.FEW_SHOT_EXAMPLES)
    res = {
        "human": I.run_task(base[0], base[1], human, three_shot),
        "synthetic": I.run_task(base[0], base[1], synth, three_shot),
    }
    systems["base, 3-shot"] = {
        "res": res,
        "probes": E.run_probes(*base),
        "text_loss": E.text_loss(*base, text),
    }

    for epoch in (1, 2, 3):
        path = CKPT / f"{tag}_epoch{epoch}.pt"
        if not path.exists():
            continue
        print(f"evaluating {path.name} ...", file=sys.stderr, flush=True)
        model, tok = E.load_tuned(path)
        res = E.evaluate_sets(model, tok, {"human": human, "synthetic": synth})
        systems[f"LoRA epoch {epoch}"] = {
            "res": res,
            "probes": E.run_probes(model, tok),
            "text_loss": E.text_loss(model, tok, text),
        }
        del model

    print("1. EXTRACTION QUALITY (95% Wilson intervals; exact = all 8 fields right)")
    print(
        f"   {'system':<16}{'set':<11}{'valid JSON':>11}{'valid order':>13}{'exact':>20}{'field acc':>11}{'prompt tok':>12}{'s / email':>11}"
    )
    for name, s in systems.items():
        for setname in ("human", "synthetic"):
            r = E.summarize_results(s["res"][setname])
            print(
                f"   {name:<16}{setname:<11}{r['valid_json']:>11.0%}{r['valid_order']:>13.0%}{E.fmt_ci(r['exact_ci']):>20}{r['field_accuracy']:>11.0%}{r['prompt_tokens']:>12.0f}{r['seconds']:>11.1f}"
            )
    print(
        "   hosted frontier model, prompted: NOT RUN (no API key); a column for it belongs here\n"
    )

    best = max(
        (n for n in systems if n.startswith("LoRA")),
        key=lambda n: E.summarize_results(systems[n]["res"]["human"])["field_accuracy"],
        default=None,
    )
    if best:
        print(
            f"2. PAIRED COMPARISON on the 38 hand-written emails: {best} against the best prompt (3-shot)"
        )
        for metric in ("exact", "fields"):
            c = E.compare(
                systems[best]["res"]["human"], systems["base, 3-shot"]["res"]["human"], metric
            )
            print(
                f"   {metric:<7} difference {c['diff']:+.3f}  95% interval [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  p = {c['p']:.4f}  (better on {c['wins']}, worse on {c['losses']}, tied on {c['ties']})"
            )
        print("\n3. WHICH FIELDS, AND WHAT KIND OF WRONG (hand-written set)")
        r = E.summarize_results(systems[best]["res"]["human"])
        print("   field accuracy: " + ", ".join(f"{f} {v:.0%}" for f, v in r["fields"].items()))
        print(
            "   error types:    "
            + ", ".join(f"{k} {v}" for k, v in E.error_types(systems[best]["res"]["human"]).items())
        )
        bad = [x for x in systems[best]["res"]["human"] if not x.score.exact][:4]
        for x in bad:
            wrong = [f for f, ok in x.score.fields.items() if not ok]
            print(f"   e.g. {x.email[:60]!r} wrong: {wrong or x.score.error}")

    print("\n4. OVERFITTING: how the checkpoints change (hand-written / synthetic exact match)")
    for name, s in systems.items():
        if name.startswith("LoRA"):
            h, y = (
                E.summarize_results(s["res"]["human"]),
                E.summarize_results(s["res"]["synthetic"]),
            )
            print(
                f"   {name:<14} hand-written {E.fmt_ci(h['exact_ci'])}   synthetic {E.fmt_ci(y['exact_ci'])}   gap {y['exact'] - h['exact']:+.0%}"
            )

    print(
        "\n5. FORGETTING: general probes (36 unrelated prompts, ordinary chat prompt) and loss on ordinary prose"
    )
    print(
        f"   {'system':<16}{'probe acc':>10}{'facts':>7}{'arith':>7}{'format':>8}{'json':>6}{'order JSON for an unrelated prompt':>38}{'prose loss':>12}"
    )
    for name, s in systems.items():
        p = s["probes"]
        k = p["by_kind"]
        print(
            f"   {name:<16}{p['accuracy']:>10.0%}{k['facts']:>7.0%}{k['arithmetic']:>7.0%}{k['format']:>8.0%}{k['json']:>6.0%}{p['order_json_rate']:>38.0%}{s['text_loss']:>12.3f}"
        )


if __name__ == "__main__":
    main(sys.argv)
