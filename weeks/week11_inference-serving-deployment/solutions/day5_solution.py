"""Week 11 Day 5 - Solution: the whole product behind one URL, driven the way the chat UI drives it, and the UI itself started for real.

Two llama.cpp servers (the Week 10 fine-tuned order extractor and a general chat model, Qwen2.5-0.5B-Instruct) sit behind the Day 4 gateway, which also serves
``POST /v1/ask`` (BM25 over this repository's Week 9-11 lessons, answers that cite numbered sources). The script then

1. asks questions through the UI's own client (``ui/client.py``): sources before the first token, streaming, citation check, measured cost, feedback
2. calls the extractor through the same gateway (``model="order-extractor"``)
3. starts the REAL Streamlit app (headless) and checks that it serves and reads the API; the page logic is tested separately with AppTest (test_day5.py)

  uv run python weeks/week11_inference-serving-deployment/solutions/day5_solution.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ui"))

import client as UI  # noqa: E402
import llamacpp as L  # noqa: E402
from stack import DEMO_USERS, ApiServer  # noqa: E402

DOCS = (
    ROOT / "weeks"
)  # the lessons of Weeks 9, 10 and 11 are the corpus (any Markdown directory works)
PRICE = UI.PriceCard(
    0.50, 1.50
)  # ASSUMED hosted-style card, an input to the UI's sidebar; no real price was looked up

QUESTIONS = [
    ("in scope", "Why do we divide attention scores by the square root of d?"),
    ("in scope", "What does the rank r control in LoRA?"),
    ("in scope", "How does continuous batching differ from static batching?"),
    ("out of scope", "What is the capital of France?"),
]
EMAIL = "Hi, this is Dana Okoye. Please send 3 x USB-C cable (order ref K-204) to my office. It is urgent, we need them tomorrow."


def run_questions(api_url: str) -> None:
    c = UI.ApiClient(api_url, "sk-demo-alice")
    print(
        "1. ASK THE COURSE through the UI client (retrieval + streamed answer; the price card below is an assumption)"
    )
    for kind, q in QUESTIONS:
        stats = UI.AnswerStats()
        text, sources = "", []
        for ev in c.ask(q, stats, k=4, max_tokens=160):
            if ev.kind == "sources":
                sources = ev.data
            elif ev.kind == "token":
                text += ev.data
        rep = UI.citation_report(text, len(sources))
        cost = UI.format_cost(PRICE, stats, text)
        print(f"\n   [{kind}] {q}")
        listed = ", ".join(
            "[{}] {} > {}".format(s["n"], s["doc"].split("/")[-1], s["heading"]) for s in sources
        )
        print(f"   sources: {listed or 'none retrieved'}")
        print(
            f"   answer ({stats.completion_tokens} tokens, first token {(stats.ttft or 0) * 1000:.0f} ms, {stats.tokens_per_second:.0f} tok/s): {' '.join(text.split())[:300]}"
        )
        print(
            f"   citations: {rep}; request {stats.request_id}; cost at the assumed card ${cost['dollars']:.6f} ({'estimated' if cost.get('estimated') else 'from server usage'})"
        )
        ok = c.feedback(stats.request_id, -1 if kind == "out of scope" else 1, "other")
        print(f"   feedback recorded: {ok}")

    print("\n2. EXTRACT AN ORDER through the same gateway, model='order-extractor'")
    stats = UI.AnswerStats()
    text = "".join(
        ev.data
        for ev in c.chat(
            [
                {"role": "system", "content": "Extract the order from the email as JSON."},
                {"role": "user", "content": EMAIL},
            ],
            stats,
            model="order-extractor",
            max_tokens=200,
        )
        if ev.kind == "token"
    )
    print(f"   {text}")
    try:
        print(f"   parses as JSON with keys: {sorted(json.loads(text))}")
    except ValueError:
        print("   (the reply is not valid JSON)")
    print(f"   usage today: {c.usage()}")


def run_streamlit(api_url: str) -> None:
    port = L.free_port()
    env = {**os.environ, "API_URL": api_url, "API_KEY": "sk-demo-alice"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(HERE / "ui/chat_app.py"),
            "--server.headless=true",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("\n3. THE STREAMLIT APP, started for real")
    try:
        ok = False
        for _ in range(60):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).text == "ok":
                    ok = True
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        page = httpx.get(f"http://127.0.0.1:{port}/", timeout=5) if ok else None
        print(
            f"   health endpoint ok: {ok}; page served: {page is not None and page.status_code == 200} "
            f"(a browser is needed to see it render; the page's behaviour is covered by the AppTest tests)"
        )
    finally:
        proc.terminate()
        proc.wait(10)


def main() -> None:
    gguf = L.ensure_chat_gguf()
    extractor = L.GGUF_DIR / f"{L.STEM}-Q8_0.gguf"
    with (
        L.LlamaServer(extractor, parallel=2, ctx=4096) as ex,
        L.LlamaServer(gguf, parallel=2, ctx=4096) as chat,
    ):
        with ApiServer(
            ex.url,
            users=DEMO_USERS,
            max_inflight=4,
            extra_env={
                "LLMAPI_BACKENDS": json.dumps({"order-extractor": ex.url, "qwen-chat": chat.url}),
                "LLMAPI_DOCS_DIR": str(DOCS),
                "LLMAPI_ASK_MODEL": "qwen-chat",
            },
        ) as api:
            print(f"(gateway {api.url}; corpus {DOCS}; models: order-extractor, qwen-chat)\n")
            run_questions(api.url)
            run_streamlit(api.url)


if __name__ == "__main__":
    main()
