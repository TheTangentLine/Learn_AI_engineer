"""Offline tests for llm-cli (no API calls): run with  `pytest weeks/week01_how-llms-work/solutions/weekly`."""

from __future__ import annotations

import pytest
from llm_cli import commands
from llm_cli.chat import ChatSession, is_retryable

from common.llm import LLMResponse, Usage


class HTTPError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def make_stream(script):
    """Build a fake `stream()` from a list of behaviours consumed one per call.

    Each behaviour is either a list of chunks (success) or an Exception to raise on the call;
    ("partial", [chunks], exc) yields the chunks and then raises exc mid-stream.
    """
    calls: list[dict] = []

    def fake(messages, **kw):
        calls.append({"messages": messages, **kw})
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, tuple) and step[0] == "partial":
            yield from step[1]
            raise step[2]
        yield from step
        text = "".join(step)
        kw["on_done"](
            LLMResponse(
                text,
                kw["provider"],
                kw["model"],
                Usage(len(messages) * 10, len(text)),
                cost_usd=0.001,
                latency_s=0.1,
                stop_reason="end_turn",
            )
        )

    fake.calls = calls
    return fake


def session(script, **kw) -> ChatSession:
    kw.setdefault("provider", "anthropic")
    kw.setdefault("model", "test-model")
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("token_counter", len)  # 1 char = 1 token: deterministic budgets
    kw.setdefault("stream_fn", make_stream(script))
    return ChatSession(**kw)


def test_send_appends_history_and_meters_cost():
    s = session([["Hel", "lo"]])
    chunks: list[str] = []
    r = s.send("hi", on_chunk=chunks.append)
    assert chunks == ["Hel", "lo"] and r.text == "Hello"
    assert [m["role"] for m in s.messages] == ["user", "assistant"]
    assert s.meter.calls == 1 and s.meter.cost_usd == pytest.approx(0.001)
    assert r.ttft_s is not None and r.error is None


def test_api_messages_strip_bookkeeping_keys():
    s = session([["a"]])
    s.messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y", "partial": True},
    ]
    assert s.api_messages() == [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]


def test_retries_transient_error_before_first_token():
    s = session([HTTPError(429), HTTPError(503), ["ok"]])
    seen = []
    r = s.send("hi", on_retry=lambda n, d, e: seen.append(n))
    assert r.text == "ok" and r.retries == 2 and seen == [1, 2] and r.error is None


def test_fatal_error_not_retried_and_history_stays_clean():
    fake = make_stream([HTTPError(400)])
    s = session([], stream_fn=fake)
    r = s.send("hi")
    assert isinstance(r.error, HTTPError) and r.retries == 0 and len(fake.calls) == 1
    assert s.messages == []  # no dangling user message


def test_retries_exhausted_surfaces_error():
    s = session([HTTPError(500)] * 4, max_retries=3)
    r = s.send("hi")
    assert r.retries == 3 and r.error is not None and s.messages == []


def test_mid_stream_failure_keeps_partial_and_does_not_retry():
    fake = make_stream([("partial", ["half an ans"], ConnectionError("reset"))])
    s = session([], stream_fn=fake)
    r = s.send("hi")
    assert r.text == "half an ans" and r.error is not None and len(fake.calls) == 1
    assert s.messages[-1] == {"role": "assistant", "content": "half an ans", "partial": True}


def test_keyboard_interrupt_keeps_partial():
    fake = make_stream([("partial", ["abc"], KeyboardInterrupt())])
    s = session([], stream_fn=fake)
    r = s.send("hi")
    assert r.interrupted and r.text == "abc" and s.messages[-1]["partial"] is True


def test_trim_drops_oldest_and_keeps_conversation_valid():
    s = session([["x" * 10]] * 3, max_context_tokens=120)
    for _ in range(3):
        s.send("u" * 30)
    # Whatever remains must start with a user turn, end with the newest exchange, and fit.
    assert s.messages[0]["role"] == "user"
    assert s.messages[-1]["role"] == "assistant"
    assert s.context_tokens() <= 120
    assert len(s.messages) < 6


def test_trim_counts_system_prompt_and_reports_dropped():
    s = session([["y"], ["z"]], system="S" * 50, max_context_tokens=100)
    s.send("a" * 20)
    r = s.send("b" * 20)
    assert r.dropped_messages >= 1


def test_oversized_newest_message_is_kept_and_flagged_not_truncated():
    s = session([["ok"]], max_context_tokens=100)
    r = s.send("q" * 500)
    assert r.over_budget and s.messages[0]["content"] == "q" * 500


def test_switching_provider_keeps_history_and_passes_new_provider():
    fake = make_stream([["one"], ["two"]])
    s = session([], stream_fn=fake)
    s.send("first")
    s.set_provider("openai")
    s.send("second")
    assert fake.calls[1]["provider"] == "openai"
    assert len(fake.calls[1]["messages"]) == 3  # user, assistant, user: history carried over
    assert set(s.meter.by_model) == {"anthropic/test-model", "openai/gpt-6.1-sol"}


def test_unknown_provider_rejected():
    s = session([])
    with pytest.raises(ValueError):
        s.set_provider("nope")


def test_save_and_load_roundtrip(tmp_path):
    s = session([["hello"]], system="be brief")
    s.send("hi")
    path = s.save(tmp_path / "chat.json")
    s2 = session([])
    s2.load(path)
    assert s2.messages == s.messages and s2.system == "be brief" and s2.model == "test-model"


def test_commands():
    s = session([["a"], ["b"]])
    s.send("q1")
    assert "unknown" in commands.handle(s, "/nope").output
    assert "anthropic" in commands.handle(s, "/provider").output
    assert "now using openai" in commands.handle(s, "/provider openai").output
    assert "unknown provider" in commands.handle(s, "/provider zzz").output
    assert (
        "system prompt set" in commands.handle(s, "/system be terse").output
        and s.system == "be terse"
    )
    assert "context budget = 500" in commands.handle(s, "/budget 500").output
    assert commands.handle(s, "/budget abc").output.startswith("usage")
    assert "total cost" in commands.handle(s, "/cost").output
    assert "q1" in commands.handle(s, "/history").output
    res = commands.handle(s, "/retry")
    assert res.resend == "q1" and s.messages == []
    s.send("q2")
    assert "removed" in commands.handle(s, "/undo").output and s.messages == []
    assert commands.handle(s, "/quit").quit


@pytest.mark.parametrize(
    "exc,expected",
    [
        (HTTPError(429), True),
        (HTTPError(529), True),
        (HTTPError(400), False),
        (HTTPError(401), False),
        (TimeoutError(), True),
        (ValueError("x"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected
