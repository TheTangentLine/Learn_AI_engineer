"""Week 11 weekly challenge: deploy a RAG API and the fine-tuned model end to end in Docker, evaluate it, load-test it, and write ``outputs/w11_report.md``.

  uv run python weeks/week11_inference-serving-deployment/solutions/weekly/serve_rag/run_weekly.py [--quick] [--report-only] [--keep]

1. RETRIEVAL   (offline, no model) BM25 over the lessons: is the right lesson retrieved? choose a relevance floor on the dev half of the questions
2. DEPLOY      docker compose: two llama.cpp containers (fine-tuned extractor, Qwen2.5-0.5B chat) + the API container built from this repository
3. QUALITY     the labelled questions through /v1/ask: baseline (any BM25 match), a relevance floor, and the floor plus abstaining without calling the model; the 38 hand-written emails through the extractor
4. LOAD        closed-loop sweep and open-loop arrival rates on /v1/ask
5. REPORT      ``outputs/w11_report.md`` from the measured numbers (``--report-only`` re-renders from the cached ``outputs/w11_weekly/results.json``)
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parents[1]
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SOLUTIONS))
sys.path.insert(0, str(SOLUTIONS / "deploy"))
sys.path.insert(0, str(SOLUTIONS / "ui"))
sys.path.append(str(ROOT / "weeks/week10_fine-tuning/solutions"))

import build_context as BC  # noqa: E402
import dockerlab as D  # noqa: E402
import evalrag as E  # noqa: E402
import llamacpp as L  # noqa: E402
import loadtest as LT  # noqa: E402
import orders as O  # noqa: E402
import questions as Q  # noqa: E402
import report as R  # noqa: E402
import stackctl as S  # noqa: E402
from llmapi import rag  # noqa: E402

CACHE = ROOT / "outputs" / "w11_weekly"
REPORT = ROOT / "outputs" / "w11_report.md"
URL = "http://127.0.0.1:8000"


def retrieval_analysis(docs_dir: Path, rows: list[dict]) -> dict:
    index = rag.Bm25Index(rag.load_corpus(docs_dir))
    res = E.retrieval_only(index.search, rows)
    ins = [r for r in res if r["in_scope"]]
    outs = [r for r in res if not r["in_scope"]]
    dev = [(r["top_score"], r["in_scope"]) for r in res if r["split"] == "dev"]
    floor = E.best_floor(dev)
    test_in = [r for r in ins if r["split"] == "test"]
    test_out = [r for r in outs if r["split"] == "test"]
    return {
        "floor": floor,
        "info": {
            "chunks": len(index.chunks),
            "docs": len({c.doc for c in index.chunks}),
            "n_in": len(ins),
            "hit_all": E.wilson(sum(bool(r["hit"]) for r in ins), len(ins)),
            "rank1": sum(r["rank"] == 1 for r in ins),
            "in_min": min(r["top_score"] for r in ins),
            "in_max": max(r["top_score"] for r in ins),
            "in_median": statistics.median(r["top_score"] for r in ins),
            "out_min": min(r["top_score"] for r in outs),
            "out_max": max(r["top_score"] for r in outs),
            "out_median": statistics.median(r["top_score"] for r in outs),
            "test_in": len(test_in),
            "test_in_kept": sum(r["top_score"] > floor for r in test_in),
            "test_out": len(test_out),
            "test_out_dropped": sum(r["top_score"] <= floor for r in test_out),
        },
    }


def extraction_check(url: str, key: str) -> dict:
    """The 38 hand-written emails through the deployed gateway, model='order-extractor', greedy."""
    import httpx

    scores = []
    with httpx.Client(timeout=120) as c:
        for email, gold in O.HUMAN_EMAILS:
            r = c.post(
                f"{url}/v1/chat/completions",
                headers={"authorization": f"Bearer {key}"},
                json={
                    "model": "order-extractor",
                    "messages": [
                        {"role": "system", "content": O.SYSTEM_SHORT},
                        {"role": "user", "content": email},
                    ],
                    "max_tokens": 200,
                    "temperature": 0,
                },
            )
            reply = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
            scores.append(O.score(reply, gold))
    s = O.summarize(scores)
    s["exact"] = E.wilson(sum(x.exact for x in scores), len(scores))
    return s


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "results.json"
    if "--report-only" in argv:
        _write(
            json.loads(cache_file.read_text())
        )  # JSON turned the interval tuples into lists; the report only unpacks them
        return
    if not D.available():
        sys.exit("docker is not available: this challenge deploys with docker compose")
    chat_gguf = L.ensure_chat_gguf()
    extractor = L.GGUF_DIR / f"{L.STEM}-Q8_0.gguf"
    if not extractor.exists():
        sys.exit(f"{extractor} is missing: run Day 1 first")
    rows = Q.questions()
    if quick:
        rows = [r for r in rows if r["id"] in {"in00", "in01", "in02", "in03", "out00", "out01"}]
    ctx = Path(tempfile.mkdtemp(prefix="w11-weekly-"))
    info = BC.build(ctx / "api", limit_weeks=11)
    ra = retrieval_analysis(ctx / "api" / "docs", Q.questions())
    floor = ra["floor"]
    print(
        f"1. RETRIEVAL: floor chosen on dev = {floor:.1f}; hit rate {E.fmt(ra['info']['hit_all'])}",
        flush=True,
    )

    models = L.GGUF_DIR
    assert chat_gguf.parent == models
    env0 = S.compose_env(models, ctx / "api", 0.0)
    res: dict = {
        "floor": floor,
        "retrieval": {**ra["info"], "docs": info["docs"]},
        "n_in": sum(r["in_scope"] for r in rows),
        "n_out": sum(not r["in_scope"] for r in rows),
        "quality": {},
    }
    try:
        res["up_seconds"] = S.up(env0)
        res["ready_seconds"] = S.wait_ready(URL)
        res["image_mb"] = D.image_size_mb(f"{S.PROJECT}-api")
        print(
            f"2. DEPLOYED: up in {res['up_seconds']:.0f} s, ready {res['ready_seconds']:.1f} s later, image {res['image_mb']:.0f} MB",
            flush=True,
        )
        res["extraction"] = extraction_check(URL, S.KEY)
        print(
            f"3a. extractor behind the gateway: {E.fmt(res['extraction']['exact'])} exact",
            flush=True,
        )
        res["quality"]["baseline"] = {
            "outcomes": E.run_questions(URL, S.KEY, rows, model="qwen-chat")
        }
        print("3b. baseline questions done", flush=True)
        env1 = S.compose_env(models, ctx / "api", floor)
        S.up(env1, "api")
        S.wait_ready(URL)
        res["quality"]["floor"] = {"outcomes": E.run_questions(URL, S.KEY, rows, model="qwen-chat")}
        print("3c. floor questions done", flush=True)
        env2 = S.compose_env(models, ctx / "api", floor, abstain=True)
        S.up(env2, "api")
        S.wait_ready(URL)
        res["quality"]["abstain"] = {
            "outcomes": E.run_questions(URL, S.KEY, rows, model="qwen-chat")
        }
        print("3d. floor + abstain questions done", flush=True)
        res["per_level"] = 8 if quick else 32
        res["open_duration"] = 6.0 if quick else 30.0
        res["closed"] = LT.closed_sweep(
            URL, S.KEY, rows, levels=(1, 2) if quick else (1, 2, 4, 8), per_level=res["per_level"]
        )
        print("4a. closed-loop sweep done", flush=True)
        res["open"] = LT.open_runs(
            URL,
            S.KEY,
            rows,
            rates=(0.5, 1.0) if quick else (0.5, 1.0, 2.0, 4.0),
            duration=res["open_duration"],
        )
        res["knee"] = LT.find_knee(res["open"])
        res["container_stats"] = S.container_stats(env2)
        print("4b. open-loop runs done", flush=True)
    finally:
        if "--keep" not in argv:
            S.down(env0)
    cache_file.write_text(json.dumps(res, indent=1, default=list))
    _write(res)


def _write(res: dict) -> None:
    REPORT.write_text(R.render(res))
    print(f"report written to {REPORT}")


if __name__ == "__main__":
    main(sys.argv)
