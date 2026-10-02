"""A streaming chat UI (Streamlit) for the API: two modes, citations, feedback buttons, and a per-answer cost and speed readout.

    API_URL=http://127.0.0.1:8000 API_KEY=sk-... streamlit run chat_app.py

  Ask the course   POST /v1/ask: retrieves passages, streams an answer that cites them; the sources appear before the first word of the answer
  Extract an order POST /v1/chat/completions on the fine-tuned extractor; the reply is parsed and shown as a JSON object

All the logic (SSE parsing, citation checks, cost arithmetic) lives in ``client.py`` and is unit-tested; this file only draws it. Tests inject a fake client through
``st.session_state['_client']`` and drive the page with ``streamlit.testing.v1.AppTest``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import (
    AnswerStats,
    ApiClient,
    PriceCard,
    citation_report,
    format_cost,
    render_with_links,
)  # noqa: E402

SYSTEM_SHORT = "Extract the order from the email as JSON."
EXAMPLES = [
    "Why do we divide attention scores by the square root of d?",
    "What does LoRA change, and what is the rank for?",
    "How does continuous batching differ from static batching?",
]


def get_client() -> ApiClient:
    injected = st.session_state.get("_client")
    if injected is not None:
        return injected
    return ApiClient(
        st.session_state.get("api_url", os.environ.get("API_URL", "http://127.0.0.1:8000")),
        st.session_state.get("api_key", os.environ.get("API_KEY", "")),
    )


def send_feedback(index: int, rating: int) -> None:
    msg = st.session_state.messages[index]
    if msg.get("feedback") is None and msg.get("request_id"):
        ok = get_client().feedback(
            msg["request_id"], rating, st.session_state.get(f"reason_{index}", "other")
        )
        msg["feedback"] = rating if ok else "failed"


def draw_answer(msg: dict, index: int, price: PriceCard) -> None:
    n = len(msg.get("sources") or [])
    if msg.get("mode") == "extract":
        try:
            st.json(json.loads(msg["content"]))
        except ValueError:
            st.warning("the model's reply is not valid JSON")
            st.code(msg["content"])
    else:
        st.markdown(render_with_links(msg["content"], n))
    if n:
        rep = citation_report(msg["content"], n)
        if rep["invalid"]:
            st.error(
                f"The answer cites source(s) {rep['invalid']} that do not exist: treat it as unreliable."
            )
        elif rep["uncited_claims"]:
            st.warning("The answer cites no source: it may not be grounded in the documents.")
        with st.expander(f"Sources ({n})", expanded=False):
            for s in msg["sources"]:
                st.markdown(f"**[{s['n']}]** `{s['doc']}` · {s['heading']}  \n{s['snippet']}")
    c = msg.get("cost")
    if c:
        speed = (
            f" · first token {c['ttft'] * 1000:.0f} ms · {c['tokens_per_second']:.0f} tokens/s"
            if c.get("ttft")
            else ""
        )
        est = " (estimated)" if c["estimated"] else ""
        st.caption(
            f"{c['prompt_tokens']} prompt + {c['completion_tokens']} answer tokens{est} · ${c['dollars']:.6f} at the card in the sidebar{speed} · {c['seconds']:.1f} s"
        )
    if msg.get("request_id"):
        cols = st.columns([1, 1, 6])
        done = msg.get("feedback")
        cols[0].button(
            "👍",
            key=f"up_{index}",
            on_click=send_feedback,
            args=(index, 1),
            disabled=done is not None,
        )
        cols[1].button(
            "👎",
            key=f"down_{index}",
            on_click=send_feedback,
            args=(index, -1),
            disabled=done is not None,
        )
        if done in (1, -1):
            cols[2].caption(
                "thanks: recorded" if done == 1 else "thanks: recorded, we will look at this one"
            )
        elif done == "failed":
            cols[2].caption("could not send feedback")


def main() -> None:
    st.set_page_config(page_title="Course assistant", page_icon="💬", layout="centered")
    st.session_state.setdefault("messages", [])
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Ask the course", "Extract an order"], key="mode")
        st.text_input(
            "API URL", value=os.environ.get("API_URL", "http://127.0.0.1:8000"), key="api_url"
        )
        st.text_input(
            "API key", value=os.environ.get("API_KEY", ""), type="password", key="api_key"
        )
        k = st.slider("Passages to retrieve", 1, 8, 4, key="k") if mode == "Ask the course" else 0
        max_tokens = st.slider("Max answer tokens", 16, 400, 200, key="max_tokens")
        st.subheader("Price card (an assumption)")
        pin = st.number_input(
            "$ per million input tokens", min_value=0.0, value=0.0, step=0.1, key="price_in"
        )
        pout = st.number_input(
            "$ per million output tokens", min_value=0.0, value=0.0, step=0.1, key="price_out"
        )
        price = PriceCard(pin, pout)
        usage = get_client().usage()
        if usage:
            st.progress(
                min(1.0, usage["tokens_today"] / max(1, usage["daily_tokens"])),
                text=f"{usage['tokens_today']:,} of {usage['daily_tokens']:,} tokens used today",
            )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("Course assistant")
    st.caption(
        "Answers cite their sources. Thumbs tell us when they are wrong."
        if mode == "Ask the course"
        else "Paste a customer email; the fine-tuned model returns the order as JSON."
    )
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                draw_answer(msg, i, price)

    if not st.session_state.messages and mode == "Ask the course":
        st.write("Try one of these:")
        for ex in EXAMPLES:
            st.caption(f"· {ex}")

    prompt = st.chat_input("Ask a question" if mode == "Ask the course" else "Paste an email")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        client, stats, sources, text = get_client(), AnswerStats(), [], ""
        src_box, out_box = st.container(), st.empty()
        if mode == "Ask the course":
            events = client.ask(prompt, stats, k=k, max_tokens=max_tokens)
        else:
            events = client.chat(
                [{"role": "system", "content": SYSTEM_SHORT}, {"role": "user", "content": prompt}],
                stats,
                model="order-extractor",
                max_tokens=max_tokens,
            )
        error = None
        for ev in events:
            if ev.kind == "sources":
                sources = ev.data
                with src_box:
                    st.caption(
                        f"retrieved {len(sources)} passage(s): "
                        + ", ".join(f"[{s['n']}] {s['doc'].split('/')[-1]}" for s in sources)
                    )
            elif ev.kind == "token":
                text += ev.data
                out_box.markdown(text + "▌")
            elif ev.kind == "error":
                error = ev.data
        out_box.empty()
        if error:
            wait = f" Try again in {error['retry_after']} s." if error.get("retry_after") else ""
            (st.warning if stats.status == 429 else st.error)(
                f"{error.get('message', 'the request failed')}{wait}"
            )
            if text:
                st.markdown(text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text,
                    "mode": "ask" if mode == "Ask the course" else "extract",
                    "error": error,
                    "sources": sources,
                }
            )
            return
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": text,
                "mode": "ask" if mode == "Ask the course" else "extract",
                "sources": sources,
                "request_id": stats.request_id,
                "cost": format_cost(price, stats, text),
                "feedback": None,
            }
        )
    st.rerun()


main()
