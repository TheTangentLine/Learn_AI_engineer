"""The weekly report: a markdown document assembled from measured results (never typed by hand), with every assumption and every thing that was not run stated."""

from __future__ import annotations

from collections.abc import Sequence


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def build_report(m: dict) -> str:
    """``m`` holds: systems (name -> {human, synthetic summaries, cost per 1000}), compare (paired comparison tuned vs baseline), fields, forgetting, training, assumptions, not_run."""
    sysrows = []
    for name, s in m["systems"].items():
        h, y = s["human"], s["synthetic"]
        sysrows.append(
            [
                name,
                f"{h['valid_order']:.0%}",
                f"{h['exact']:.0%} [{h['exact_ci'][1]:.0%}, {h['exact_ci'][2]:.0%}]",
                f"{h['field_accuracy']:.0%}",
                f"{y['exact']:.0%} [{y['exact_ci'][1]:.0%}, {y['exact_ci'][2]:.0%}]",
                f"{h['prompt_tokens']:.0f}",
                f"{h['seconds']:.1f}",
            ]
        )
    c = m["compare"]
    cost_rows = [
        [
            name,
            f"{v['prompt_tokens']:.0f}",
            f"{v['new_tokens']:.0f}",
            f"{v['hosted']:.2f}",
            f"{v['self_hosted']:.3f}",
        ]
        for name, v in m["costs"].items()
    ]
    lines = [
        "# Week 10 report: fine-tuning a small model for order extraction",
        "",
        f"**Task.** {m['task']}",
        f"**Model.** {m['model']}. **Training.** {m['training']}",
        "",
        "## Results (38 hand-written emails never used for training; 100 held-out synthetic emails)",
        md_table(
            [
                "system",
                "valid order (hand)",
                "exact (hand) [95% CI]",
                "field accuracy (hand)",
                "exact (synthetic) [95% CI]",
                "prompt tokens",
                "s / email",
            ],
            sysrows,
        ),
        "",
        f"**Fine-tuned against the best prompt, paired on the same {c['n']} hand-written emails:** exact match {c['diff_exact']:+.2f} [{c['exact_low']:+.2f}, {c['exact_high']:+.2f}] (p {'< 0.0001' if c['p_exact'] < 0.0001 else '= ' + format(c['p_exact'], '.4f')}; better on {c['wins']}, worse on {c['losses']}, tied on {c['ties']}); per-field accuracy {c['diff_fields']:+.2f} [{c['fields_low']:+.2f}, {c['fields_high']:+.2f}].",
        "",
        "## What it costs",
        md_table(
            [
                "system",
                "prompt tokens",
                "new tokens",
                f"hosted-style price per 1,000 ({m['assumptions']['card']})",
                "self-hosted per 1,000",
            ],
            cost_rows,
        ),
        "",
        f"Break-even: the one-off cost of {m['break_even']['one_off']:.2f} (training) is repaid after about {m['break_even']['requests']:,.0f} requests relative to the best prompted baseline under the same price card.",
        "",
        "## Forgetting",
        m["forgetting"],
        "",
        "## Assumptions (all prices are inputs, none were looked up)",
        "\n".join(f"- {a}" for a in m["assumptions"]["list"]),
        "",
        "## Not run",
        "\n".join(f"- {a}" for a in m["not_run"]),
        "",
        "## Limits",
        "\n".join(f"- {a}" for a in m["limits"]),
        "",
    ]
    return "\n".join(lines)
