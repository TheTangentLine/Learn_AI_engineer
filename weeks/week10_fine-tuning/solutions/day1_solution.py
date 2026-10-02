"""Week 10 Day 1 - Solution: should this task be fine-tuned? Evidence first.

1. TEMPLATE   our ChatML renderer against the tokenizer's own ``apply_chat_template``
2. EXAMPLES   training examples with the loss on the answer only; what the dataset statistics say
3. FORMATS    the same conversation in Alpaca, ShareGPT, prompt/completion and preference layouts
4. BASELINE   the untouched SmolLM2-135M-Instruct on 38 hand-written emails, zero-shot and with 3 examples: what does prompting cost, and what does it get?
5. MEMO       the numbers a decision memo needs

  uv run python weeks/week10_fine-tuning/solutions/day1_solution.py [--no-baseline]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import chatfmt as C  # noqa: E402
import infer as I  # noqa: E402
import orders as O  # noqa: E402


def sample_conversations() -> list[list[dict]]:
    email, gold = O.HUMAN_EMAILS[0]
    u = lambda t: {"role": "user", "content": t}  # noqa: E731
    a = lambda t: {"role": "assistant", "content": t}  # noqa: E731
    s = lambda t: {"role": "system", "content": t}  # noqa: E731
    return [
        [u("Hi")],
        [s("Be brief."), u("What is 2+2?")],
        [u("one"), a("two"), u("three")],
        [s("S"), u("q1"), a("a1"), u("q2"), a("a2"), u("q3")],
        [u("  leading and trailing spaces  \n\n")],
        [u("unicode: café 日本語 🙂"), a("ok")],
        [s(O.SYSTEM_SHORT), u(email), a(O.order_json(gold))],
        [u("line1\nline2\n\n<not a control token>")],
    ]


def template_parity(tok) -> tuple[int, int]:
    convs = sample_conversations()
    ok = 0
    for msgs in convs:
        for gen in (False, True):
            if msgs[-1]["role"] == "assistant" and gen:
                continue
            ok += C.render(msgs, add_generation_prompt=gen) == tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=gen
            )
    total = sum(2 if m[-1]["role"] != "assistant" else 1 for m in convs)
    return ok, total


def baseline(model, tok, name: str, make, shots_label: str = "") -> dict:
    res = I.run_task(model, tok, O.HUMAN_EMAILS, make)
    s = O.summarize([r.score for r in res])
    s.update(
        name=name,
        prompt_tokens=statistics.fmean(r.prompt_tokens for r in res),
        new_tokens=statistics.fmean(r.new_tokens for r in res),
        seconds=statistics.fmean(r.seconds for r in res),
        results=res,
    )
    return s


def main(argv: list[str]) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(I.BASE)
    ok, total = template_parity(tok)
    print(
        f"1. TEMPLATE: our renderer equals tokenizer.apply_chat_template on {ok} of {total} cases (with and without the generation prompt)\n"
    )

    email, gold = O.HUMAN_EMAILS[0]
    msgs = I.training_messages(email, gold)
    ex = C.encode_example(msgs, tok)
    print("2. TRAINING EXAMPLE (the first email): the loss sees only the highlighted answer")
    print(
        f"   {len(ex)} tokens, loss on {ex.n_answer} ({ex.n_answer / len(ex):.0%}): {tok.decode([t for t in ex.labels if t != C.IGNORE])!r}"
    )
    exs = [C.encode_example(I.training_messages(e, g), tok) for e, g in O.HUMAN_EMAILS]
    st = C.dataset_stats(exs)
    print(
        f"   over the 38 emails: {st['tokens']:,} tokens, answer fraction {st['answer_fraction']:.0%}, length median {st['p50']} / 95th percentile {st['p95']} / max {st['max']}\n"
    )

    print("3. THE SAME CONVERSATION IN FOUR LAYOUTS")
    print(f"   alpaca     : {sorted(C.to_alpaca(msgs))}")
    print(f"   sharegpt   : {[c['from'] for c in C.to_sharegpt(msgs)['conversations']]}")
    pc = C.to_prompt_completion(msgs)
    print(
        f"   prompt/completion: prompt ends {pc['prompt'][-24:]!r}; completion ends {pc['completion'][-12:]!r}"
    )
    print(f"   preference : {sorted(C.to_preference(msgs[:-1], O.order_json(gold), '{}'))}\n")

    if "--no-baseline" in argv:
        return
    model, _ = I.load_base()
    print("4. BASELINE: untouched SmolLM2-135M-Instruct on 38 hand-written emails (greedy)")
    print(
        f"   {'prompt':<22}{'valid JSON':>11}{'valid order':>13}{'exact':>7}{'field acc':>11}{'prompt tok':>12}{'new tok':>9}{'s / email':>11}"
    )
    runs = [
        baseline(model, tok, "zero-shot + schema", I.zero_shot_messages),
        baseline(model, tok, "3-shot + schema", I.few_shot_messages(O.FEW_SHOT_EXAMPLES)),
    ]
    for r in runs:
        print(
            f"   {r['name']:<22}{r['valid_json']:>11.0%}{r['valid_order']:>13.0%}{r['exact']:>7.0%}{r['field_accuracy']:>11.0%}{r['prompt_tokens']:>12.0f}{r['new_tokens']:>9.0f}{r['seconds']:>11.1f}"
        )
    best = runs[1]
    print(
        "\n   per-field accuracy of the 3-shot prompt: "
        + ", ".join(f"{f} {v:.0%}" for f, v in best["fields"].items())
    )
    print("   what the failures look like (3-shot):")
    seen = set()
    for r in best["results"]:
        key = r.score.error.split(":")[0] + ("" if r.score.valid_order else r.score.error[:40])
        if not r.score.exact and key not in seen and len(seen) < 4:
            seen.add(key)
            print(f"     {r.score.error or 'valid but wrong fields'}  <- {r.reply[:90]!r}")


if __name__ == "__main__":
    main(sys.argv)
