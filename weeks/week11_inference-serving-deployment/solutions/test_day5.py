"""Tests for Week 11 Day 5: the retrieval layer, /v1/ask and /v1/feedback, the UI client's SSE parsing and arithmetic, and the Streamlit page driven headlessly with AppTest."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ui"))

import client as UI  # noqa: E402
from llmapi import ApiKeyStore, EchoBackend, User, create_app, make_key, rag  # noqa: E402
from llmapi.app import Settings  # noqa: E402

DOCS = {
    "a/attention.md": "# Attention\n\n## Scaling\n\nWe divide the attention scores by the square root of d so that softmax does not saturate.\n\n## Masks\n\nThe causal mask hides future tokens.\n",
    "b/lora.md": "# LoRA\n\nLoRA freezes the weights and learns a low-rank update with a rank r and a scale alpha over r.\n\n```python\n# a heading inside a fence\nx = 1\n```\n\nMerging folds the update into the weights.\n",
    "c/batching.md": "# Batching\n\n## Continuous batching\n\nNew requests join the batch after every decoding step; finished ones leave at once.\n",
}


@pytest.fixture(scope="module")
def index():
    chunks = []
    for name, text in DOCS.items():
        chunks += rag.chunk_markdown(text, name)
    return rag.Bm25Index(chunks)


# ----------------------------------------------------------------------------- chunking and retrieval


def test_chunks_follow_headings_and_carry_the_heading_path():
    chunks = rag.chunk_markdown(DOCS["a/attention.md"], "a/attention.md")
    assert (
        [c.heading for c in chunks] == ["Attention > Scaling", "Attention > Masks"]
        and chunks[0].id == "a/attention.md#0"
        and "square root" in chunks[0].text
    )


def test_a_hash_inside_a_code_fence_is_not_a_heading_and_fences_are_not_split():
    chunks = rag.chunk_markdown(DOCS["b/lora.md"], "b/lora.md")
    assert (
        len(chunks) == 1
        and "# a heading inside a fence" in chunks[0].text
        and chunks[0].heading == "LoRA"
    )


def test_long_sections_split_at_paragraph_boundaries_and_trivial_ones_are_dropped():
    long = "# T\n\n" + "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(10))
    chunks = rag.chunk_markdown(long, "t.md", max_chars=500)
    assert (
        len(chunks) > 3
        and all(len(c.text) <= 500 + 400 for c in chunks)
        and all(c.heading == "T" for c in chunks)
    )
    assert rag.chunk_markdown("# Only a title\n\n## Empty\n", "x.md") == []


def test_search_ranks_the_matching_chunk_first_and_ignores_stop_words_and_unmatched_queries(index):
    top = index.search("why divide attention scores by sqrt d", 3)
    assert top[0][0].doc == "a/attention.md" and top[0][0].heading.endswith("Scaling")
    assert index.search("continuous batching of requests", 1)[0][0].doc == "c/batching.md"
    assert index.search("the of and", 3) == [] and index.search("zzzxyzzy", 3) == []
    assert rag.tokenize("The LoRA rank, r=8!") == ["lora", "rank", "r"] or "lora" in rag.tokenize(
        "The LoRA rank, r=8!"
    )
    with pytest.raises(ValueError):
        rag.Bm25Index([])


def test_a_relevance_floor_drops_weak_matches_and_a_floor_above_everything_leaves_no_sources(index):
    hits = index.search("continuous batching decoding step", 4)
    assert len(hits) >= 1
    top = hits[0][1]
    assert [c.id for c, _ in index.search("continuous batching decoding step", 4, top - 1e-9)][
        :1
    ] == [hits[0][0].id]
    assert index.search("continuous batching decoding step", 4, top) == []  # strictly above
    messages, sources = rag.make_retriever(index, min_score=1e9)("continuous batching", 4)
    assert sources == [] and "<sources>\n</sources>" in messages[1]["content"]
    _, normal = rag.make_retriever(index)("continuous batching", 4)
    assert normal  # the default floor of 0 still retrieves


def test_the_prompt_numbers_the_sources_wraps_them_and_demands_citations(index):
    hits = index.search("lora rank", 2)
    msgs = rag.build_messages("what is the rank?", hits)
    assert (
        msgs[0]["role"] == "system"
        and "Cite the source number" in msgs[0]["content"]
        and "I don't know based on the provided sources" in msgs[0]["content"]
    )
    user = msgs[1]["content"]
    assert (
        user.startswith("<sources>")
        and '<source id="1" ref="b/lora.md">' in user
        and user.rstrip().endswith("<question>what is the rank?</question>")
    )


def test_the_retriever_returns_messages_and_user_facing_sources_and_survives_no_match(index):
    msgs, sources = rag.make_retriever(index)("how does continuous batching work", 2)
    assert (
        sources[0]["n"] == 1
        and sources[0]["doc"] == "c/batching.md"
        and set(sources[0]) == {"n", "id", "doc", "heading", "snippet", "score"}
        and len(msgs) == 2
    )
    msgs, sources = rag.make_retriever(index)("zzzxyzzy", 3)
    assert sources == [] and "<sources>\n</sources>" in msgs[1]["content"]


def test_citation_helpers():
    assert rag.cited_numbers("as shown [1] and [3], again [1]") == [
        1,
        3,
        1,
    ] and rag.invalid_citations("see [1] [4] [0]", 3) == [0, 4]
    assert UI.used_citations("[2] x [1] [2]") == [1, 2]


# ----------------------------------------------------------------------------- the API's ask and feedback endpoints

KEY = make_key()
HEADERS = {"authorization": f"Bearer {KEY}"}


def make_app(index, backend=None, **kw):
    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_tokens_cap=100))
    backend = backend or EchoBackend()
    return create_app(backend, keys, retriever=rag.make_retriever(index), **kw), backend


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_ask_retrieves_passages_prompts_the_model_with_them_and_returns_sources_with_the_answer(
    index,
):
    app, backend = make_app(index)

    async def go():
        async with client_for(app) as c:
            return await c.post(
                "/v1/ask",
                json={"question": "why divide attention scores by sqrt d", "k": 2},
                headers=HEADERS,
            )

    r = asyncio.run(go())
    j = r.json()
    assert (
        r.status_code == 200
        and j["sources"][0]["doc"] == "a/attention.md"
        and len(j["sources"]) <= 2
        and j["object"] == "chat.completion"
    )
    sent = backend.calls[0]["messages"]
    assert "square root of d" in sent[1]["content"] and sent[0]["role"] == "system"


def test_a_streamed_ask_sends_the_sources_event_before_any_token(index):
    app, _ = make_app(index)

    async def go():
        async with client_for(app) as c:
            return await c.post(
                "/v1/ask", json={"question": "continuous batching", "stream": True}, headers=HEADERS
            )

    text = asyncio.run(go()).text
    assert text.startswith("event: sources\ndata: ") and text.index("event: sources") < text.index(
        '"content"'
    )
    events = list(UI.parse_sse(iter(text.splitlines())))
    assert (
        events[0].kind == "sources"
        and events[0].data[0]["doc"] == "c/batching.md"
        and events[-1].kind == "done"
        and any(e.kind == "usage" for e in events)
    )


def test_ask_needs_a_key_validates_the_question_and_is_limited_like_chat(index):
    app, backend = make_app(index)

    async def go():
        async with client_for(app) as c:
            return (
                (await c.post("/v1/ask", json={"question": "x"})).status_code,
                (await c.post("/v1/ask", json={"question": ""}, headers=HEADERS)).status_code,
                (
                    await c.post(
                        "/v1/ask", json={"question": "x", "max_tokens": 5000}, headers=HEADERS
                    )
                ).status_code,
            )

    assert asyncio.run(go()) == (401, 400, 400) and backend.calls == []


def test_ask_is_absent_when_no_retriever_is_configured():
    keys = ApiKeyStore()
    keys.add(KEY, User("alice"))
    app = create_app(EchoBackend(), keys)

    async def go():
        async with client_for(app) as c:
            return (await c.post("/v1/ask", json={"question": "x"}, headers=HEADERS)).status_code

    assert asyncio.run(go()) == 404


def test_ask_with_several_models_uses_the_configured_ask_model_and_an_explicit_model_wins(index):
    extractor, chat = EchoBackend(), EchoBackend()
    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_tokens_cap=100))
    app = create_app(
        {"order-extractor": extractor, "chat": chat},
        keys,
        retriever=rag.make_retriever(index),
        settings=Settings(ask_model="chat"),
    )

    async def go(body):
        async with client_for(app) as c:
            return (await c.post("/v1/ask", json=body, headers=HEADERS)).status_code

    assert asyncio.run(go({"question": "continuous batching"})) == 200
    assert (len(chat.calls), len(extractor.calls)) == (1, 0)
    assert asyncio.run(go({"question": "continuous batching", "model": "order-extractor"})) == 200
    assert (len(chat.calls), len(extractor.calls)) == (1, 1)
    assert asyncio.run(go({"question": "continuous batching", "model": "nope"})) == 404


def test_with_nothing_retrieved_the_gateway_abstains_without_calling_the_model(index):
    # a retriever whose floor is above every score: no sources at all
    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_tokens_cap=100))
    backend = EchoBackend()
    app = create_app(
        backend,
        keys,
        retriever=rag.make_retriever(index, min_score=1e9),
        settings=Settings(abstain_without_sources=True),
    )

    async def go(stream):
        async with client_for(app) as c:
            r = await c.post(
                "/v1/ask",
                json={"question": "continuous batching", "stream": stream},
                headers=HEADERS,
            )
            m = await c.get("/metrics", headers={"authorization": "Bearer x"})
            return r, m

    r, _ = asyncio.run(go(False))
    j = r.json()
    assert r.status_code == 200 and j["choices"][0]["message"]["content"] == rag.ABSTAIN
    assert j["sources"] == [] and j["usage"]["total_tokens"] == 0
    assert backend.calls == []  # the model was never asked
    r2, _ = asyncio.run(go(True))
    assert (
        r2.status_code == 200 and rag.ABSTAIN.split()[0] in r2.text and "event: sources" in r2.text
    )
    assert backend.calls == []
    assert app.state.metrics.render().count("llmapi_abstained_total 2") == 1


def test_the_abstain_rule_only_applies_when_nothing_was_retrieved(index):
    app, backend = make_app(index, settings=Settings(abstain_without_sources=True))

    async def go():
        async with client_for(app) as c:
            return await c.post(
                "/v1/ask", json={"question": "continuous batching", "k": 2}, headers=HEADERS
            )

    j = asyncio.run(go()).json()
    assert len(backend.calls) == 1 and j["sources"]  # there were sources: the model was asked
    assert j["choices"][0]["message"]["content"] != rag.ABSTAIN


def test_search_returns_at_most_k_hits_even_when_more_chunks_match(index):
    many = index.search("the", 10) or index.search("attention batching weights", 10)
    assert len(index.search("attention batching weights causal", 2)) <= 2
    assert len(index.search("attention batching weights causal", 1)) == 1
    assert len(many) >= 1


def test_without_the_flag_an_empty_retrieval_still_calls_the_model(index):
    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_tokens_cap=100))
    backend = EchoBackend()
    app = create_app(backend, keys, retriever=rag.make_retriever(index, min_score=1e9))

    async def go():
        async with client_for(app) as c:
            return (
                await c.post("/v1/ask", json={"question": "continuous batching"}, headers=HEADERS)
            ).status_code

    assert asyncio.run(go()) == 200 and len(backend.calls) == 1


def test_the_process_entry_point_builds_models_and_retrieval_from_the_environment(
    monkeypatch, tmp_path
):
    from llmapi.__main__ import build_app

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "# Batching\n\nContinuous batching lets new requests join each step.\n"
    )
    monkeypatch.setenv("LLMAPI_KEYS", json.dumps([{"user": "u", "key": KEY}]))
    monkeypatch.setenv("LLMAPI_BACKENDS", json.dumps({"m1": "http://h1:1", "m2": "http://h2:2"}))
    monkeypatch.setenv("LLMAPI_DOCS_DIR", str(docs))
    monkeypatch.setenv("LLMAPI_ASK_MODEL", "m2")
    monkeypatch.setenv("LLMAPI_ABSTAIN_WITHOUT_SOURCES", "1")
    monkeypatch.setenv("LLMAPI_MIN_SCORE", "7.5")
    app = build_app()
    assert set(app.state.backends) == {"m1", "m2"} and app.state.settings.ask_model == "m2"
    assert app.state.settings.abstain_without_sources is True
    assert any(r.path == "/v1/ask" for r in app.routes)
    monkeypatch.delenv("LLMAPI_DOCS_DIR")
    monkeypatch.delenv("LLMAPI_BACKENDS")
    app2 = build_app()
    assert not any(r.path == "/v1/ask" for r in app2.routes) and len(app2.state.backends) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("LLMAPI_DOCS_DIR", str(empty))
    with pytest.raises(SystemExit):
        build_app()


def test_feedback_is_recorded_per_user_with_a_reason_and_never_stores_the_answer(index, tmp_path):
    app, _ = make_app(index, settings=Settings(feedback_path=str(tmp_path / "fb.jsonl")))

    async def go():
        async with client_for(app) as c:
            ok = await c.post(
                "/v1/feedback",
                json={
                    "request_id": "r1",
                    "rating": -1,
                    "reason": "bad_citation",
                    "comment": "cites the wrong page",
                },
                headers=HEADERS,
            )
            bad_rating = await c.post(
                "/v1/feedback", json={"request_id": "r1", "rating": 3}, headers=HEADERS
            )
            bad_reason = await c.post(
                "/v1/feedback",
                json={"request_id": "r1", "rating": 1, "reason": "because"},
                headers=HEADERS,
            )
            anon = await c.post("/v1/feedback", json={"request_id": "r1", "rating": 1})
            long_comment = await c.post(
                "/v1/feedback",
                json={"request_id": "r1", "rating": 1, "comment": "x" * 501},
                headers=HEADERS,
            )
            return ok, bad_rating, bad_reason, anon, long_comment, await c.get("/metrics")

    ok, bad_rating, bad_reason, anon, long_comment, metrics = asyncio.run(go())
    assert (
        ok.status_code == 200
        and bad_rating.status_code == 400
        and bad_reason.status_code == 400
        and anon.status_code == 401
        and long_comment.status_code == 400
    )
    rec = json.loads((tmp_path / "fb.jsonl").read_text().splitlines()[0])
    assert (
        rec["user"] == "alice"
        and rec["rating"] == -1
        and rec["reason"] == "bad_citation"
        and rec["request_id"] == "r1"
        and set(rec) == {"at", "user", "request_id", "rating", "reason", "comment"}
    )
    assert (
        'llmapi_feedback_total{rating="down",reason="bad_citation"} 1' in metrics.text
        and len(app.state.feedback) == 1
    )


# ----------------------------------------------------------------------------- the UI client


def test_sse_parsing_yields_sources_tokens_usage_errors_and_done():
    lines = [
        "event: sources",
        'data: [{"n": 1, "doc": "a.md"}]',
        "",
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}}',
        'data: {"error": {"message": "boom", "code": "backend_error"}}',
        "data: not json",
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "after done"}}]}',
    ]
    evs = list(UI.parse_sse(iter(lines)))
    assert (
        [e.kind for e in evs] == ["sources", "token", "usage", "error", "done"]
        and evs[1].data == "Hi"
        and evs[2].data["total_tokens"] == 9
        and evs[3].data["code"] == "backend_error"
    )


def fake_transport(handler):
    return httpx.MockTransport(handler)


def test_the_client_fills_the_stats_from_a_stream_and_reports_the_request_id():
    sse = 'event: sources\ndata: [{"n": 1}]\n\ndata: {"choices": [{"delta": {"content": "a "}}]}\n\ndata: {"choices": [{"delta": {"content": "b"}}]}\n\ndata: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}\n\ndata: [DONE]\n\n'

    def handler(request):
        body = json.loads(request.content)
        assert (
            request.url.path == "/v1/ask"
            and body["stream"] is True
            and request.headers["authorization"] == "Bearer k"
        )
        return httpx.Response(
            200, text=sse, headers={"x-request-id": "rid9", "content-type": "text/event-stream"}
        )

    c = UI.ApiClient("http://api/", "k", transport=fake_transport(handler))
    stats = UI.AnswerStats()
    text = "".join(e.data for e in c.ask("q", stats) if e.kind == "token")
    assert (
        text == "a b"
        and stats.request_id == "rid9"
        and stats.status == 200
        and stats.chunks == 2
        and stats.sources == [{"n": 1}]
    )
    assert (
        (stats.prompt_tokens, stats.completion_tokens, stats.total_tokens) == (10, 2, 12)
        and stats.ttft is not None
        and stats.seconds >= stats.ttft
        and stats.error is None
    )


def test_http_errors_become_error_events_with_the_retry_after_hint_and_network_errors_too():
    def limited(request):
        return httpx.Response(
            429,
            json={"error": {"message": "too many requests", "code": "rate_limited"}},
            headers={"retry-after": "7"},
        )

    stats = UI.AnswerStats()
    evs = list(
        UI.ApiClient("http://api", "k", transport=fake_transport(limited)).chat(
            [{"role": "user", "content": "x"}], stats
        )
    )
    assert (
        [e.kind for e in evs] == ["error"]
        and evs[0].data["retry_after"] == "7"
        and stats.status == 429
        and stats.error["code"] == "rate_limited"
    )

    def down(request):
        raise httpx.ConnectError("refused")

    stats2 = UI.AnswerStats()
    evs2 = list(UI.ApiClient("http://api", "k", transport=fake_transport(down)).chat([], stats2))
    assert evs2[0].data["code"] == "unreachable" and stats2.seconds >= 0

    def html(request):
        return httpx.Response(502, text="<html>bad gateway</html>")

    s3 = UI.AnswerStats()
    assert (
        list(UI.ApiClient("http://api", "k", transport=fake_transport(html)).ask("q", s3))[0].data[
            "code"
        ]
        == "http_error"
    )


def test_feedback_and_usage_calls_report_failure_instead_of_raising():
    ok = UI.ApiClient(
        "http://api",
        "k",
        transport=fake_transport(
            lambda r: httpx.Response(200, json={"status": "recorded", "tokens_today": 5})
        ),
    )
    assert ok.feedback("r", 1) is True and ok.usage()["tokens_today"] == 5

    def boom(request):
        raise httpx.ConnectError("x")

    dead = UI.ApiClient("http://api", "k", transport=fake_transport(boom))
    assert dead.feedback("r", 1) is False and dead.usage() is None
    assert (
        UI.ApiClient(
            "http://api", "k", transport=fake_transport(lambda r: httpx.Response(401, json={}))
        ).usage()
        is None
    )


def test_the_citation_report_and_link_rendering():
    rep = UI.citation_report("LoRA freezes weights [1][3] and merges [9].", 3)
    assert rep == {
        "cited": [1, 3],
        "invalid": [9],
        "unused": [2],
        "abstained": False,
        "uncited_claims": False,
    }
    assert (
        UI.citation_report("I don't know based on the provided sources.", 3)["abstained"]
        and UI.citation_report("It is a thing.", 3)["uncited_claims"] is True
    )
    assert UI.citation_report("", 3)["uncited_claims"] is False
    assert UI.render_with_links("a [1] b [7]", 2) == "a [**1**] b ~~[7]~~"


def test_cost_arithmetic_uses_reported_usage_and_flags_estimates():
    card = UI.PriceCard(input_per_m=2.0, output_per_m=10.0)
    assert card.cost(1_000_000, 100_000) == 3.0 and card.cost(None, None) == 0.0
    s = UI.AnswerStats(prompt_tokens=500, completion_tokens=100, ttft=0.2, seconds=1.2, chunks=11)
    c = UI.format_cost(card, s, "x" * 400)
    assert (
        c["estimated"] is False
        and c["dollars"] == pytest.approx((500 * 2 + 100 * 10) / 1e6)
        and s.tokens_per_second == pytest.approx(10.0)
    )
    est = UI.format_cost(card, UI.AnswerStats(seconds=1.0), "x" * 400)
    assert (
        est["estimated"] is True and est["completion_tokens"] == 100 and UI.estimate_tokens("") == 1
    )


# ----------------------------------------------------------------------------- the page, headless


class FakeClient:
    """Stands in for ApiClient: scripted events, recorded feedback."""

    def __init__(
        self,
        answer="Scores are divided by sqrt d [1] to keep softmax stable [5].",
        sources=None,
        error=None,
    ):
        self.answer, self.sources, self.error = (
            answer,
            sources
            if sources is not None
            else [
                {
                    "n": 1,
                    "id": "a#0",
                    "doc": "a/attention.md",
                    "heading": "Attention > Scaling",
                    "snippet": "We divide the scores",
                    "score": 3.1,
                }
            ],
            error,
        )
        self.feedback_calls, self.asked = [], []

    def usage(self):
        return {
            "tokens_today": 1000,
            "daily_tokens": 10_000,
            "in_flight": 0,
            "requests": 1,
            "rejected": 0,
        }

    def _events(self, stats):
        if self.error:
            stats.status, stats.error = self.error["status"], self.error
            yield UI.Event("error", self.error)
            return
        stats.status, stats.request_id = 200, "rid-ui"
        if self.sources:
            stats.sources = self.sources
            yield UI.Event("sources", self.sources)
        for w in self.answer.split(" "):
            stats.chunks += 1
            stats.ttft = stats.ttft or 0.05
            yield UI.Event("token", w + " ")
        stats.prompt_tokens, stats.completion_tokens, stats.seconds = 120, 14, 0.9
        yield UI.Event("done")

    def ask(self, question, stats, **kw):
        self.asked.append(("ask", question, kw))
        return self._events(stats)

    def chat(self, messages, stats, **kw):
        self.asked.append(("chat", messages, kw))
        return self._events(stats)

    def feedback(self, request_id, rating, reason="other", comment=""):
        self.feedback_calls.append((request_id, rating, reason))
        return True


def page(fake):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(HERE / "ui" / "chat_app.py"), default_timeout=20)
    at.session_state["_client"] = fake
    return at.run()


def texts(at, kind):
    return [e.value for e in getattr(at, kind)]


def test_the_page_renders_with_examples_and_a_usage_meter_and_no_exceptions():
    at = page(FakeClient())
    assert (
        not at.exception
        and any("continuous batching" in c for c in texts(at, "caption"))
        and at.title[0].value == "Course assistant"
    )


def test_asking_a_question_shows_the_answer_its_citations_the_cost_and_a_bad_citation_warning():
    fake = FakeClient()
    at = page(fake).chat_input[0].set_value("why scale attention?").run()
    assert (
        not at.exception
        and fake.asked[0][0] == "ask"
        and fake.asked[0][1] == "why scale attention?"
        and fake.asked[0][2]["k"] == 4
    )
    md = " ".join(texts(at, "markdown"))
    assert "[**1**]" in md and "~~[5]~~" in md, (
        "a valid citation is bold, the invented one struck through"
    )
    assert any("do not exist" in e for e in texts(at, "error"))
    assert any(
        "120 prompt + 14 answer tokens" in c and "first token 50 ms" in c
        for c in texts(at, "caption")
    )
    assert [b.label for b in at.button if b.label in ("👍", "👎")] == ["👍", "👎"]


def test_a_thumbs_down_is_sent_with_the_request_id_and_then_disabled():
    fake = FakeClient(answer="Fine [1].")
    at = page(fake).chat_input[0].set_value("q").run()
    next(b for b in at.button if b.label == "👎").click().run()
    assert fake.feedback_calls == [("rid-ui", -1, "other")]
    assert all(b.disabled for b in at.button if b.label in ("👍", "👎")) and any(
        "recorded" in c for c in texts(at, "caption")
    )


def test_an_answer_without_citations_is_flagged_and_an_abstention_is_not():
    at = page(FakeClient(answer="It just works.")).chat_input[0].set_value("q").run()
    assert any("cites no source" in w for w in texts(at, "warning"))
    at2 = (
        page(FakeClient(answer="I don't know based on the provided sources.", sources=[]))
        .chat_input[0]
        .set_value("q")
        .run()
    )
    assert not texts(at2, "warning") and not texts(at2, "error")


def test_a_rate_limit_is_a_warning_with_the_retry_hint_and_other_errors_are_errors():
    at = (
        page(
            FakeClient(
                error={
                    "status": 429,
                    "message": "too many requests",
                    "code": "rate_limited",
                    "retry_after": "7",
                }
            )
        )
        .chat_input[0]
        .set_value("q")
        .run()
    )
    assert (
        any("too many requests" in w and "7 s" in w for w in texts(at, "warning"))
        and not at.exception
    )
    at2 = (
        page(
            FakeClient(
                error={"status": 502, "message": "the model server failed", "code": "backend_error"}
            )
        )
        .chat_input[0]
        .set_value("q")
        .run()
    )
    assert any("model server failed" in e for e in texts(at2, "error"))


def test_extract_mode_sends_the_system_instruction_and_shows_the_json():
    fake = FakeClient(answer='{"is_order": true, "order_id": "A-1"}', sources=[])
    at = page(fake)
    at.radio[0].set_value("Extract an order").run()
    at = at.chat_input[0].set_value("Hi, order A-1 please").run()
    kind, messages, kw = fake.asked[0]
    assert (
        kind == "chat"
        and messages[0]["content"] == "Extract the order from the email as JSON."
        and messages[1]["content"] == "Hi, order A-1 please"
        and kw["model"] == "order-extractor"
    )
    assert not at.exception and at.json and "A-1" in at.json[0].value


def test_citation_number_zero_is_invalid_and_struck_through():
    assert "~~[0]~~" in UI.render_with_links(
        "see [0] and [1]", 3
    ) and "[**1**]" in UI.render_with_links("see [0] and [1]", 3)
    assert UI.citation_report("see [0]", 3)["invalid"] == [0]


def test_a_failing_feedback_status_is_reported_as_failure():
    for status in (400, 401, 429, 500):
        c = UI.ApiClient(
            "http://api",
            "k",
            transport=fake_transport(lambda r, s=status: httpx.Response(s, json={})),
        )
        assert c.feedback("r", 1) is False


class SlowSync(httpx.SyncByteStream):
    def __init__(self, pieces, gap):
        self.pieces, self.gap = pieces, gap

    def __iter__(self):
        import time

        for p in self.pieces:
            yield p
            time.sleep(self.gap)


def test_the_first_token_time_is_measured_once_not_overwritten_by_later_tokens():
    pieces = [
        b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "b"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "c"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    c = UI.ApiClient(
        "http://api",
        "k",
        transport=fake_transport(lambda r: httpx.Response(200, stream=SlowSync(pieces, 0.05))),
    )
    stats = UI.AnswerStats()
    list(c.ask("q", stats))
    assert stats.ttft < 0.04 and stats.seconds >= 0.14 and stats.chunks == 3
    assert stats.tokens_per_second == pytest.approx((3 - 1) / (stats.seconds - stats.ttft))


def slow_retriever(delay: float):
    import time

    def retrieve(question: str, k: int):
        time.sleep(delay)  # a blocking call, like a cross-encoder
        return [{"role": "user", "content": question}], [{"n": 1, "doc": "d.md", "heading": "", "snippet": "s", "score": 1.0, "id": "d#1"}]

    return retrieve


def test_a_slow_retriever_does_not_block_the_event_loop():
    import time

    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_tokens_cap=100))
    app = create_app(EchoBackend(), keys, retriever=slow_retriever(0.4), settings=Settings(max_inflight=2, max_queue=4))

    async def go():
        async with client_for(app) as c:
            ask = asyncio.create_task(c.post("/v1/ask", json={"question": "hello there"}, headers=HEADERS))
            await asyncio.sleep(0.05)  # the retriever is now sleeping in its thread
            t0 = time.perf_counter()
            r = await c.get("/healthz")
            health = time.perf_counter() - t0
            return r.status_code, health, (await ask).status_code

    status, health_seconds, ask_status = asyncio.run(go())
    assert (status, ask_status) == (200, 200) and health_seconds < 0.15  # it answered while the retriever was still working


def test_overload_while_preparing_the_answer_is_a_fast_503_not_a_long_wait():
    import time

    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_concurrent=50, max_tokens_cap=100))
    app = create_app(EchoBackend(), keys, retriever=slow_retriever(0.3), settings=Settings(max_inflight=1, max_queue=1))

    async def one(c):
        t0 = time.perf_counter()
        r = await c.post("/v1/ask", json={"question": "hello there"}, headers=HEADERS)
        return r.status_code, time.perf_counter() - t0, r.json().get("error", {}).get("code")

    async def go():
        async with client_for(app) as c:
            return await asyncio.gather(*(one(c) for _ in range(6)))

    out = asyncio.run(go())
    refused = [o for o in out if o[0] == 503]
    served = [o for o in out if o[0] == 200]
    assert len(served) == 2 and len(refused) == 4  # capacity is max_inflight + max_queue = 2 preparing at once
    assert all(code == "overloaded" and secs < 0.1 for _, secs, code in refused)  # refused immediately


def test_the_preparing_count_returns_to_zero_and_at_most_max_inflight_retrievals_run_at_once():
    import threading
    import time

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def tracked(question, k):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        return [{"role": "user", "content": question}], []

    keys = ApiKeyStore()
    keys.add(KEY, User("alice", rpm=6000, burst=100, max_concurrent=50, max_tokens_cap=100))
    app = create_app(EchoBackend(), keys, retriever=tracked, settings=Settings(max_inflight=2, max_queue=4))

    async def go():
        async with client_for(app) as c:
            first = await asyncio.gather(*(c.post("/v1/ask", json={"question": "hello there"}, headers=HEADERS) for _ in range(6)))
            later = await asyncio.gather(*(c.post("/v1/ask", json={"question": "hello again"}, headers=HEADERS) for _ in range(6)))
            return [r.status_code for r in first], [r.status_code for r in later]

    first, later = asyncio.run(go())
    assert first == [200] * 6 and later == [200] * 6  # a second burst is served: nothing was left counted as "preparing"
    assert state["peak"] == 2  # never more than max_inflight at once (and it did reach it)
