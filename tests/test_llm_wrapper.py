"""End-to-end tests of common/llm.py through the real SDKs against a local fake server."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from fake_llm_server import ANSWER_TEXT, CHUNKS, FakeLLMServer  # noqa: E402

from common import llm  # noqa: E402


class Person(BaseModel):
    name: str
    age: int


@pytest.fixture()
def server(monkeypatch):
    with FakeLLMServer() as srv:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
        monkeypatch.setenv("OPENAI_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("OLLAMA_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()
        llm.SESSION.__init__()
        yield srv
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()


# ------------------------------------------------------------------ complete


def test_anthropic_complete_and_usage_normalisation(server):
    r = llm.complete("hi", system="be brief", provider="anthropic", cache_prompt=True)
    assert r.text == ANSWER_TEXT and r.stop_reason == "end_turn"
    # input_tokens excludes cached tokens on Anthropic; cache read/write are separate fields
    assert (r.usage.input_tokens, r.usage.output_tokens) == (10, 5)
    assert (r.usage.cache_read_tokens, r.usage.cache_write_tokens) == (7, 3)
    expected = (10 * 5.00 + 7 * 0.50 + 3 * 5.00 * 1.25 + 5 * 25.00) / 1e6
    assert r.cost_usd == pytest.approx(expected)
    body = server.requests[-1]["body"]
    assert body["system"] == "be brief" and body["cache_control"] == {"type": "ephemeral"}
    assert body["model"] == "claude-opus-5" and body["max_tokens"] == 16000


def test_openai_complete_subtracts_cached_from_input(server):
    r = llm.complete("hi", system="be brief", provider="openai")
    assert r.text == ANSWER_TEXT
    # OpenAI's input_tokens (20) INCLUDES the 8 cached ones -> normalised to 12 + 8
    assert (r.usage.input_tokens, r.usage.cache_read_tokens, r.usage.output_tokens) == (12, 8, 6)
    body = server.requests[-1]["body"]
    assert body["instructions"] == "be brief" and body["max_output_tokens"] == 16000
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert r.cost_usd == pytest.approx((12 * 2.00 + 8 * 0.10 + 6 * 10.00) / 1e6)


def test_ollama_complete_is_free_and_prepends_system(server):
    r = llm.complete("hi", system="sys", provider="ollama")
    assert r.text == ANSWER_TEXT and r.cost_usd == 0.0 and r.usage.input_tokens == 12
    assert server.requests[-1]["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_provider_from_env_and_session_tracker(server, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert llm.resolve() == ("openai", "gpt-6.1-sol")
    llm.complete("a")
    llm.complete("b", provider="anthropic")
    assert llm.SESSION.calls == 2 and llm.SESSION.cost_usd > 0


# ------------------------------------------------------------------ streaming


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_stream_yields_chunks_and_final_response(server, provider):
    done = []
    chunks = list(llm.stream("hi", provider=provider, on_done=done.append))
    assert chunks == CHUNKS
    (resp,) = done
    assert resp.text == ANSWER_TEXT and resp.provider == provider
    assert resp.usage.output_tokens > 0 and resp.stop_reason


def test_anthropic_stream_usage_combines_start_and_delta(server):
    done = []
    list(llm.stream("hi", provider="anthropic", on_done=done.append))
    u = done[0].usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (10, 5, 7)


def test_stream_request_flags(server):
    list(llm.stream("hi", provider="ollama"))
    body = server.requests[-1]["body"]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


# ------------------------------------------------------------------ async


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_acomplete_and_astream(server, provider):
    async def go():
        r = await llm.acomplete("hi", provider=provider)
        parts = [c async for c in llm.astream("hi", provider=provider)]
        both = await asyncio.gather(*(llm.acomplete(f"q{i}", provider=provider) for i in range(4)))
        return r, parts, both

    r, parts, both = asyncio.run(go())
    assert r.text == ANSWER_TEXT and parts == CHUNKS and len(both) == 4


# ------------------------------------------------------------------ structured


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_structured_returns_validated_model(server, provider):
    person, resp = llm.structured("Extract: Ada is 36", Person, provider=provider)
    assert person == Person(name="Ada", age=36)
    assert resp.provider == provider


# ------------------------------------------------------------------ errors / helpers


def test_http_error_propagates_with_status(server):
    server.fail_next = [400]
    with pytest.raises(Exception) as ei:
        llm.complete("hi", provider="anthropic")
    assert getattr(ei.value, "status_code", None) == 400


def test_sdk_retry_setting_is_applied(server, monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    for fn in (llm._anthropic, llm._openai, llm._ollama):
        fn.cache_clear()
    server.fail_next = [503, 503]
    r = llm.complete("hi", provider="anthropic")  # the SDK retries twice, third attempt succeeds
    assert r.text == ANSWER_TEXT and len(server.requests) == 3


def test_unknown_provider_and_available_providers(server):
    with pytest.raises(ValueError):
        llm.resolve("nope")
    assert set(llm.available_providers()) >= {"anthropic", "openai", "ollama"}


def test_cost_of_unknown_model_is_zero():
    assert llm.cost_usd("llama3.2:3b", llm.Usage(1000, 1000)) == 0.0
