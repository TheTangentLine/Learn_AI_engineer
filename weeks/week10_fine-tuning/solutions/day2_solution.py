"""Week 10 Day 2 - Solution: a 1,000-example synthetic dataset with quality filters, and a measurement of what the filters catch.

1. GENERATE  1,500 emails from templates, 8% of them with a deliberately corrupted LABEL (a wrong id, a wrong quantity, an invented item, ...)
2. FILTER    check every label against the email's own text; remove exact and near duplicates (MinHash + LSH); remove anything that overlaps the evaluation set
3. MEASURE   how many of the known defects each stage caught, how many clean examples it wrongly removed, and what slipped through
4. BALANCE   the label distributions before and after, and the token statistics the training run will see
5. WRITE     train / dev / test as chat-format JSONL and a data card

  uv run python weeks/week10_fine-tuning/solutions/day2_solution.py [--out DIR]
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import chatfmt as C  # noqa: E402
import gen_data as G  # noqa: E402
import infer as I  # noqa: E402
import orders as O  # noqa: E402

N_RAW, DEFECT_RATE, N_DEV, N_TEST, SEED = 1500, 0.08, 100, 150, 0


def eval_texts() -> list[str]:
    """Everything that must stay unseen during training: the hand-written evaluation emails and the few-shot examples of the baseline."""
    return [e for e, _ in O.HUMAN_EMAILS + O.FEW_SHOT_EXAMPLES]


def build(seed: int = SEED, n_raw: int = N_RAW, defect_rate: float = DEFECT_RATE):
    raw = G.generate(n_raw, seed=seed, defect_rate=defect_rate)
    kept, report = G.curate(raw, eval_texts())
    train, dev, test = G.split(kept, N_DEV, N_TEST, seed=seed)
    return raw, kept, report, (train, dev, test)


def to_record(s: G.Sample) -> dict:
    return {
        "messages": I.training_messages(s.email, s.gold),
        "template": s.template,
        "defect": s.defect,
    }


def defect_catch_table(raw: list[G.Sample]) -> list[tuple[str, int, int]]:
    """For each kind of injected defect: how many there were and how many a validity check against the email text caught."""
    rows = []
    for kind in G.DEFECTS:
        bad = [s for s in raw if s.defect == kind]
        rows.append((kind, len(bad), sum(1 for s in bad if G.failures(s))))
    return rows


def lsh_accuracy(texts: list[str], threshold: float = 0.8) -> dict:
    """MinHash + LSH against the exact all-pairs Jaccard on the same texts: how many true near-duplicate pairs it finds, and how much work it saves."""
    sh = [G.shingles(t) for t in texts]
    t0 = time.perf_counter()
    truth = {
        (i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if G.jaccard(sh[i], sh[j]) >= threshold
    }
    t_exact = time.perf_counter() - t0
    t0 = time.perf_counter()
    found = G.near_duplicate_pairs(texts, threshold)
    t_lsh = time.perf_counter() - t0
    return {
        "true_pairs": len(truth),
        "found": len(found & truth),
        "false_pairs": len(found - truth),
        "recall": len(found & truth) / max(1, len(truth)),
        "t_exact": t_exact,
        "t_lsh": t_lsh,
        "comparisons": len(texts) * (len(texts) - 1) // 2,
    }


def lsh_scaling(sizes=(600, 1200, 2400, 4800)) -> list[tuple[int, float, float, float]]:
    """Seconds for exact all-pairs Jaccard and for MinHash + LSH as the number of texts grows: quadratic against roughly linear."""
    texts = [s.email for s in G.generate(max(sizes), seed=3)]
    rows = []
    for n in sizes:
        sh = [G.shingles(t) for t in texts[:n]]
        t0 = time.perf_counter()
        _ = sum(1 for i in range(n) for j in range(i + 1, n) if G.jaccard(sh[i], sh[j]) >= 0.8)
        t_exact = time.perf_counter() - t0
        t0 = time.perf_counter()
        G.near_duplicate_pairs(texts[:n])
        rows.append(
            (n, t_exact, time.perf_counter() - t0, t_exact / max(1e-9, time.perf_counter() - t0))
        )
    return rows


def fmt_dist(d: dict[str, float]) -> str:
    return "  ".join(f"{k}: {v:.0%}" for k, v in d.items())


def main(argv: list[str]) -> None:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "outputs/w10_data"
    raw, kept, report, (train, dev, test) = build()
    print(
        f"1. GENERATED {report.generated} emails; {report.defects_injected} ({report.defects_injected / report.generated:.1%}) have a deliberately wrong LABEL\n"
    )

    print("2. FILTER FUNNEL (each stage sees what the previous one left)")
    left = report.generated
    for name, removed in report.stages:
        left -= removed
        print(f"   removed {removed:>4}  {name:<52} -> {left} left")
    print(
        f"\n3. WHAT THE LABEL CHECKS CAUGHT (known defects only; {len(raw) - report.defects_injected} clean samples produced {report.false_rejections} false rejections)"
    )
    print(f"   {'defect':<20}{'injected':>9}{'caught':>8}")
    for kind, n, caught in defect_catch_table(raw):
        print(
            f"   {kind:<20}{n:>9}{caught:>8}   {'' if caught == n else '<- ' + str(n - caught) + ' slipped through'}"
        )
    print(
        f"   overall: {report.defects_caught} of {report.defects_injected} caught ({report.defects_caught / report.defects_injected:.0%}); {report.defects_missed} defective examples remain in the {report.kept} kept ({report.defects_missed / report.kept:.1%} label noise)"
    )
    print(
        "   which checks fired: "
        + ", ".join(f"{k} {v}" for k, v in report.failure_counts.most_common())
    )
    slipped = [s for s in kept if s.defect]
    for s in slipped[:3]:
        print(
            f"   slipped through ({s.defect}): {s.email[:80]!r} -> label items {s.gold['items'][:2]}, urgency {s.gold['urgency']}"
        )

    print(
        "\n4. NEAR-DUPLICATE DETECTION: MinHash + LSH against exact all-pairs Jaccard (600 emails, threshold 0.8)"
    )
    acc = lsh_accuracy([s.email for s in raw[:600]])
    print(
        f"   true pairs {acc['true_pairs']}: LSH found {acc['found']} (recall {acc['recall']:.0%}), {acc['false_pairs']} false pairs; exact {acc['t_exact']:.2f}s over {acc['comparisons']:,} comparisons, LSH {acc['t_lsh']:.2f}s"
    )

    print(
        "   scaling (seconds):  "
        + "   ".join(f"n={n}: exact {te:.2f}s, LSH {tl:.2f}s" for n, te, tl, _ in lsh_scaling())
    )

    print("\n5. BALANCE (training split)")
    d = G.distribution(train)
    for k in ("is_order", "urgency", "currency", "has_date", "n_items", "template"):
        print(f"   {k:<10}{fmt_dist(d[k])}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(I.BASE)
    exs = [C.encode_example(r["messages"], tok) for r in map(to_record, train)]
    st = C.dataset_stats(exs)
    print(
        f"\n   {len(train)} train / {len(dev)} dev / {len(test)} test; the training split is {st['tokens']:,} tokens ({st['answer_fraction']:.0%} are answer tokens), length median {st['p50']}, 95th percentile {st['p95']}, max {st['max']}, truncated {st['truncated']}"
    )

    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        C.write_jsonl(out / f"{name}.jsonl", (to_record(s) for s in part))
    contaminated = sum(
        any(G.ngram_overlap(s.email, e) for e in eval_texts()) for s in train + dev + test
    )
    (out / "data_card.md").write_text(
        f"# Order-extraction dataset\n\nGenerated {report.generated} emails (seed {SEED}), kept {report.kept} after filtering "
        f"({report.defects_missed} known label defects remain = {report.defects_missed / report.kept:.1%}). Splits: {len(train)} / {len(dev)} / {len(test)}.\n"
        f"Evaluation emails sharing an 8-word sequence with any split: {contaminated}.\nTemplates: {dict(Counter(s.template for s in train))}\n"
    )
    print(
        f"\nwrote {out}/train.jsonl, dev.jsonl, test.jsonl, data_card.md; evaluation emails sharing an 8-word sequence with a split: {contaminated}"
    )
    _ = random


if __name__ == "__main__":
    main(sys.argv)
