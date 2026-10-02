"""Week 11 Day 1 - Solution: the same model at four quantisation levels, on a real runtime: quality against speed against size.

The model is the Week 10 fine-tune (SmolLM2-135M-Instruct + LoRA, merged), converted to GGUF and quantised with llama.cpp (installed with `brew install llama.cpp`).
For each format:
  size         the file on disk and the effective bits per weight
  speed        llama-bench: prompt processing and token generation, on the GPU (Metal) and on the CPU only
  quality      (a) exact match on the 38 hand-written order emails served through llama-server at temperature 0
               (b) perplexity on 40,000 characters of ordinary prose (the Week 1-2 lessons), a model-level damage detector

  uv run python weeks/week11_inference-serving-deployment/solutions/day1_solution.py [--no-bench] [--no-quality]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.append(
    str(ROOT / "weeks/week10_fine-tuning/solutions")
)  # appended: Week 10 has its own day1_solution.py

import llamacpp as L  # noqa: E402
import orders as O  # noqa: E402

from common.abtest import wilson  # noqa: E402
from common.corpus import load_course_docs  # noqa: E402

FORMATS = ["f16", "Q8_0", "Q4_K_M", "Q4_0", "Q2_K"]
PARAMS = 134_515_008  # SmolLM2-135M (tied embedding counted once)


def path_of(fmt: str) -> Path:
    return L.GGUF_DIR / f"{L.STEM}-{fmt}.gguf"


def bits_per_weight(size_bytes: int, params: int = PARAMS) -> float:
    """File size in bits over the number of parameters: includes the scales, the metadata and the tokenizer."""
    return size_bytes * 8 / params


def extraction_quality(model: Path, items=O.HUMAN_EMAILS) -> dict:
    """Exact match and field accuracy of the model served by llama-server (its own ChatML template), greedy."""
    scores = []
    with L.LlamaServer(model, parallel=1, ctx=2048) as srv:
        for email, gold in items:
            reply = L.reply_text(
                srv.chat(
                    [
                        {"role": "system", "content": O.SYSTEM_SHORT},
                        {"role": "user", "content": email},
                    ],
                    max_tokens=200,
                )
            )
            scores.append(O.score(reply, gold))
    s = O.summarize(scores)
    s["exact_ci"] = wilson(sum(x.exact for x in scores), len(scores))
    return s


def prose_file() -> Path:
    f = L.OUT / "ppl_text.txt"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            "\n\n".join(
                d.text for d in load_course_docs() if d.short.startswith(("week01", "week02"))
            )[:40000]
        )
    return f


def main(argv: list[str]) -> None:
    L.ensure_ggufs()
    rows = []
    for fmt in FORMATS:
        p = path_of(fmt)
        row = {"fmt": fmt, "mb": L.file_mb(p), "bpw": bits_per_weight(p.stat().st_size)}
        if "--no-bench" not in argv:
            gpu, cpu = L.bench(p, ngl=99), L.bench(p, ngl=0)
            row |= {
                "pp_gpu": gpu["pp"],
                "tg_gpu": gpu["tg"],
                "pp_cpu": cpu["pp"],
                "tg_cpu": cpu["tg"],
            }
        if "--no-quality" not in argv:
            q = extraction_quality(p)
            ppl, se = L.perplexity(p, prose_file())
            row |= {
                "exact": q["exact"],
                "ci": q["exact_ci"],
                "field": q["field_accuracy"],
                "valid": q["valid_order"],
                "ppl": ppl,
                "se": se,
            }
        rows.append(row)
        print(f"  done {fmt}", file=sys.stderr, flush=True)
    print(f"{'format':<8}{'MB':>7}{'bits/w':>8}", end="")
    if "--no-bench" not in argv:
        print(f"{'pp GPU':>9}{'tg GPU':>9}{'pp CPU':>9}{'tg CPU':>9}", end="")
    if "--no-quality" not in argv:
        print(f"{'exact (hand-written)':>26}{'field':>7}{'valid':>7}{'prose ppl':>12}", end="")
    print()
    for r in rows:
        print(f"{r['fmt']:<8}{r['mb']:>7.0f}{r['bpw']:>8.1f}", end="")
        if "pp_gpu" in r:
            print(
                f"{r['pp_gpu']:>9.0f}{r['tg_gpu']:>9.0f}{r['pp_cpu']:>9.0f}{r['tg_cpu']:>9.0f}",
                end="",
            )
        if "exact" in r:
            print(
                f"{r['exact']:>10.0%} [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]{r['field']:>9.0%}{r['valid']:>7.0%}{r['ppl']:>8.1f} ±{r['se']:.1f}",
                end="",
            )
        print()


if __name__ == "__main__":
    main(sys.argv)
