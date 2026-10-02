"""Tests for common/chat.py: conversation adapters, tool-call parsing through the real SDKs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_llm_server import FakeLLMServer  # noqa: E402

from common import chat, llm  # noqa: E402
from common.chat import ToolCall, assistant_message, tool_message  # noqa: E402

WEATHER = {
    "name": "get_weather",
    "description": "Weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

HISTORY = [
    {"role": "user", "content": "Weather in Tokyo and Paris?"},
    {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {"id": "c1", "name": "get_weather", "args": {"city": "Tokyo"}},
            {"id": "c2", "name": "get_weather", "args": {"city": "Paris"}},
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "get_weather",
        "content": "22C",
        "is_error": False,
    },
    {
        "role": "tool",
        "tool_call_id": "c2",
        "name": "get_weather",
        "content": "boom",
        "is_error": True,
    },
]


# ------------------------------------------------------------------ pure adapter tests


def test_anthropic_adapter_merges_all_results_into_one_user_message_first_tool_result_blocks():
    out = chat.to_anthropic(HISTORY)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "text", "text": "Checking."}
    assert out[1]["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "get_weather",
        "input": {"city": "Tokyo"},
    }
    assert out[1]["content"][2]["input"] == {"city": "Paris"}, "arguments must travel with the call"
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"] and results[1]["is_error"] is True
    assert "is_error" not in results[0]
    assert "_results" not in out[2], "internal bookkeeping must not leak into the request"


def test_anthropic_adapter_keeps_separate_turns_separate():
    msgs = [
        *HISTORY,
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ]
    out = chat.to_anthropic(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]


def test_openai_adapter_uses_function_call_items_matched_by_call_id():
    out = chat.to_openai_input(HISTORY)
    kinds = [o.get("type") or o["role"] for o in out]
    assert kinds == [
        "user",
        "assistant",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert out[2]["call_id"] == "c1" and out[2]["arguments"] == '{"city": "Tokyo"}'
    assert out[4] == {"type": "function_call_output", "call_id": "c1", "output": "22C"}


def test_chat_completions_adapter_uses_tool_role_and_system_first():
    out = chat.to_chat_completions(HISTORY, "be brief")
    assert out[0] == {"role": "system", "content": "be brief"}
    assistant = out[2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"city": "Tokyo"}'
    assert [m["role"] for m in out[3:]] == ["tool", "tool"] and out[3]["tool_call_id"] == "c1"


def test_local_adapter_passes_dict_arguments_to_the_chat_template():
    out = chat.to_local(HISTORY)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Tokyo"}
    assert out[2] == {"role": "tool", "content": "22C"}


def test_local_output_parser_handles_parallel_calls_text_and_garbage():
    raw = (
        'Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>'
    )
    text, calls = chat.parse_local_output(raw)
    assert text == "Sure." and [c.args["city"] for c in calls] == ["Tokyo", "Paris"]
    assert [c.id for c in calls] == ["call_0", "call_1"]
    text, calls = chat.parse_local_output("<tool_call>{not json}</tool_call>Hello<|im_end|>")
    assert calls == [] and text == "Hello", "malformed calls are dropped, not crashed on"
    assert chat.parse_local_output("just text")[1] == []


def test_message_helpers_round_trip():
    t = chat.Turn(
        "hi", [ToolCall("c9", "f", {"a": 1})], "tool_use", llm.Usage(), 0.0, 0.0, "x", "m"
    )
    m = assistant_message(t)
    assert m == {
        "role": "assistant",
        "content": "hi",
        "tool_calls": [{"id": "c9", "name": "f", "args": {"a": 1}}],
    }
    assert "tool_calls" not in assistant_message(
        chat.Turn("x", [], "end_turn", llm.Usage(), 0, 0, "x", "m")
    )
    assert tool_message(ToolCall("c9", "f", {}), "ok", True) == {
        "role": "tool",
        "tool_call_id": "c9",
        "name": "f",
        "content": "ok",
        "is_error": True,
    }


def test_tool_choice_mapping():
    assert chat._anthropic_choice("auto") is None and chat._anthropic_choice("any") == {
        "type": "any"
    }
    assert chat._anthropic_choice("get_weather") == {"type": "tool", "name": "get_weather"}
    assert chat._openai_choice("required") == "required"
    assert chat._openai_choice("get_weather") == {"type": "function", "name": "get_weather"}


# ------------------------------------------------------------------ through the real SDKs


@pytest.fixture()
def server(monkeypatch):
    with FakeLLMServer() as srv:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
        monkeypatch.setenv("OPENAI_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("OLLAMA_BASE_URL", srv.url + "/v1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "t")
        monkeypatch.setenv("OPENAI_API_KEY", "t")
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()
        yield srv
        for fn in (llm._anthropic, llm._openai, llm._ollama):
            fn.cache_clear()


TWO_CALLS = {
    "text": "Let me check.",
    "tool_calls": [
        {"name": "get_weather", "args": {"city": "Tokyo"}},
        {"name": "get_weather", "args": {"city": "Paris"}},
    ],
}


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_parallel_tool_calls_are_parsed_and_the_loop_round_trips(server, provider):
    server.script = [TWO_CALLS, {"text": "Tokyo 22C, Paris error."}]
    msgs = [{"role": "user", "content": "weather?"}]
    t = chat.turn(msgs, [WEATHER], system="s", provider=provider)
    assert t.wants_tools and t.stop_reason == "tool_use" and t.text == "Let me check."
    assert [(c.name, c.args["city"]) for c in t.tool_calls] == [
        ("get_weather", "Tokyo"),
        ("get_weather", "Paris"),
    ]
    assert len({c.id for c in t.tool_calls}) == 2, "each call needs its own id"
    first = server.requests[-1]["body"]  # what we SENT: tools in this provider's schema
    names = [x.get("name") or x["function"]["name"] for x in first["tools"]]
    assert names == ["get_weather"]

    msgs += [
        assistant_message(t),
        tool_message(t.tool_calls[0], "22C"),
        tool_message(t.tool_calls[1], "no data", True),
    ]
    final = chat.turn(msgs, [WEATHER], system="s", provider=provider)
    assert (
        final.text == "Tokyo 22C, Paris error."
        and not final.wants_tools
        and final.stop_reason == "end_turn"
    )
    second = server.requests[-1]["body"]
    sent = second["messages"] if provider != "openai" else second["input"]
    blob = str(sent)
    assert "22C" in blob and "no data" in blob and t.tool_calls[0].id in blob, (
        "results reference the call ids"
    )


def test_anthropic_request_has_exactly_one_tool_result_message(server):
    server.script = [TWO_CALLS, {"text": "ok"}]
    msgs = [{"role": "user", "content": "weather?"}]
    t = chat.turn(msgs, [WEATHER], provider="anthropic")
    msgs += [assistant_message(t)] + [tool_message(c, "r") for c in t.tool_calls]
    chat.turn(msgs, [WEATHER], provider="anthropic")
    sent = server.requests[-1]["body"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert [b["type"] for b in sent[2]["content"]] == ["tool_result", "tool_result"]


def test_forced_tool_choice_is_sent_in_each_providers_shape(server):
    server.script = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    chat.turn(
        [{"role": "user", "content": "x"}],
        [WEATHER],
        provider="anthropic",
        tool_choice="get_weather",
    )
    assert server.requests[-1]["body"]["tool_choice"] == {"type": "tool", "name": "get_weather"}
    chat.turn(
        [{"role": "user", "content": "x"}], [WEATHER], provider="openai", tool_choice="get_weather"
    )
    assert server.requests[-1]["body"]["tool_choice"] == {"type": "function", "name": "get_weather"}
    chat.turn(
        [{"role": "user", "content": "x"}], [WEATHER], provider="ollama", tool_choice="required"
    )
    assert server.requests[-1]["body"]["tool_choice"] == "required"


def test_text_only_turn_and_usage_and_cost_are_filled_in(server):
    server.script = [{"text": "just text"}]
    t = chat.turn([{"role": "user", "content": "hi"}], None, provider="anthropic")
    assert (
        t.text == "just text"
        and t.tool_calls == []
        and t.usage.output_tokens == 5
        and t.cost_usd > 0
    )
    assert "tools" not in server.requests[-1]["body"], "no tools -> no tools field"


def test_malformed_tool_arguments_do_not_crash_parsing(server):
    server.script = [
        {"text": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}}]}
    ]
    server.requests.clear()
    t = chat.turn([{"role": "user", "content": "x"}], [WEATHER], provider="openai")
    assert t.tool_calls[0].args == {"city": "Tokyo"}
    assert chat._safe_json("{not json") == {"_unparseable": "{not json"} and chat._safe_json(
        "[1,2]"
    ) == {"_value": [1, 2]}
    assert chat._safe_json(None) == {} and chat._safe_json({"a": 1}) == {"a": 1}


def test_local_parser_reads_unclosed_blocks_extra_text_and_drops_garbage():
    unclosed = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    text, calls = chat.parse_local_output(unclosed)
    assert text == "" and [(c.name, c.args) for c in calls] == [("get_weather", {"city": "Tokyo"})]
    chatty = 'Sure thing! <tool_call>{"name": "f", "arguments": {"a": 1}} trailing words </tool_call> and then <tool_call>{"name": "g"}</tool_call> bye'
    text, calls = chat.parse_local_output(chatty)
    assert (
        [(c.name, c.args) for c in calls] == [("f", {"a": 1}), ("g", {})]
        and "trailing words" in text
        and text.startswith("Sure thing!")
        and text.endswith("bye")
    )
    assert [c.id for c in calls] == ["call_0", "call_1"]
    nested = '<tool_call>{"name": "f", "arguments": {"o": {"k": [1, 2, "}"]}}}</tool_call>'
    assert chat.parse_local_output(nested)[1][0].args == {"o": {"k": [1, 2, "}"]}}, (
        "braces inside strings and nesting are handled"
    )
    text, calls = chat.parse_local_output("<tool_call>{broken</tool_call>after<tool_call>")
    assert calls == [] and text == "after"
    assert chat.parse_local_output('<tool_call>{"arguments": {}}</tool_call>x')[1] == [], (
        "a call without a name is dropped"
    )


def test_forced_prefill_for_models_without_a_tool_choice_switch():
    tools = [WEATHER]
    assert (
        chat.forced_prefill(None, tools)
        == chat.forced_prefill("auto", tools)
        == chat.forced_prefill("none", tools)
        == ""
    )
    assert chat.forced_prefill("required", []) == "", "no tools, nothing to force"
    assert (
        chat.forced_prefill("required", tools)
        == chat.forced_prefill("any", tools)
        == '<tool_call>\n{"name": "'
    )
    assert (
        chat.forced_prefill("get_weather", tools)
        == '<tool_call>\n{"name": "get_weather", "arguments": '
    )
    text, calls = chat.parse_local_output(
        chat.forced_prefill("get_weather", tools) + '{"city": "Paris"}}'
    )
    assert [(c.name, c.args) for c in calls] == [("get_weather", {"city": "Paris"})], (
        "prefill + the model's continuation parses"
    )


def test_a_named_tool_choice_reaches_chat_completions_endpoints_in_their_own_shape(server):
    server.script = [{"text": "a"}]
    chat.turn(
        [{"role": "user", "content": "x"}], [WEATHER], provider="ollama", tool_choice="get_weather"
    )
    assert server.requests[-1]["body"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }
    server.script = [{"text": "b"}]
    chat.turn([{"role": "user", "content": "x"}], [WEATHER], provider="ollama", tool_choice="auto")
    assert "tool_choice" not in server.requests[-1]["body"]
