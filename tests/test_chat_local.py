"""A REAL model turn: Qwen2.5-0.5B (local) must request the right tool for an obvious question."""

from __future__ import annotations

import pytest

from common import chat

MULTIPLY = {
    "name": "multiply",
    "description": "Multiply two numbers.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
}
WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.fixture(scope="module")
def tools():
    return [MULTIPLY, WEATHER]


def test_local_model_calls_the_right_tool_with_the_right_arguments(tools):
    t = chat.turn(
        [{"role": "user", "content": "What is 4123.5 multiplied by 89.2?"}], tools, provider="local"
    )
    assert t.wants_tools and t.tool_calls[0].name == "multiply"
    assert t.tool_calls[0].args == {"a": 4123.5, "b": 89.2}


def test_local_model_makes_parallel_calls_and_declines_when_no_tool_is_needed(tools):
    t = chat.turn(
        [{"role": "user", "content": "What's the weather in Tokyo and Paris?"}],
        tools,
        provider="local",
    )
    assert sorted(c.args["city"] for c in t.tool_calls) == ["Paris", "Tokyo"]
    joke = chat.turn([{"role": "user", "content": "Tell me a joke."}], tools, provider="local")
    assert not joke.wants_tools and len(joke.text) > 10


def test_local_model_uses_the_tool_result_to_answer(tools):
    msgs = [{"role": "user", "content": "What is 6 times 7?"}]
    t = chat.turn(msgs, tools, provider="local")
    msgs += [chat.assistant_message(t)] + [chat.tool_message(c, "42") for c in t.tool_calls]
    final = chat.turn(msgs, tools, provider="local")
    assert not final.wants_tools and "42" in final.text


def test_local_turns_report_token_usage_and_the_model_is_loaded_once(tools):
    chat._local_chat.cache_clear()
    msgs = [{"role": "user", "content": "What is 4123.5 multiplied by 89.2?"}]
    t = chat.turn(msgs, tools, provider="local")
    assert t.usage.input_tokens > 100, "the tool schemas are part of the prompt"
    assert 0 < t.usage.output_tokens < 100
    no_tools = chat.turn(msgs, tools, provider="local", tool_choice="none")
    assert no_tools.usage.input_tokens < t.usage.input_tokens - 50, (
        "tool_choice=none hides the schemas"
    )
    assert not no_tools.wants_tools
    assert chat._local_chat.cache_info().misses == 1, (
        "same model -> one LocalChat, not one per turn"
    )


def test_a_forced_tool_call_happens_even_when_the_model_would_just_chat(tools):
    msgs = [{"role": "user", "content": "Say hello in French."}]
    forced = chat.turn(msgs, tools, provider="local", tool_choice="required")
    assert forced.wants_tools and forced.tool_calls[0].name in {"multiply", "get_weather"}
    named = chat.turn(
        [{"role": "user", "content": "What is the weather in Paris?"}],
        tools,
        provider="local",
        tool_choice="get_weather",
    )
    assert (
        named.wants_tools
        and named.tool_calls[0].name == "get_weather"
        and isinstance(named.tool_calls[0].args, dict)
    )
