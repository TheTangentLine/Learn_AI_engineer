"""Tests for common/local_server.py: message conversion, HTTP shape (with a stubbed model), and one REAL generation."""

from __future__ import annotations

import json

import pytest
from openai import OpenAI

from common.local_server import MODEL_NAME, LocalOpenAIServer, to_local_messages

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
]


class StubChat:
    """Stands in for LocalChat: returns scripted raw text, records what it was asked."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.seen = []

    def chat(self, messages, tools, system, max_new_tokens=0, prefill=""):
        self.seen.append(
            {
                "messages": messages,
                "tools": tools,
                "max_new_tokens": max_new_tokens,
                "prefill": prefill,
            }
        )
        return prefill + self.outputs.pop(
            0
        )  # the real LocalChat returns the prefill plus what the model generated

    def count_tokens(self, messages, tools, system):
        return 100

    def text_tokens(self, text):
        return len(text.split())


@pytest.fixture()
def server():
    with LocalOpenAIServer() as srv:
        srv.chat = StubChat()
        yield srv


def test_message_conversion_handles_every_shape_frameworks_send():
    msgs = [
        {"role": "developer", "content": "be brief"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "42"}]},
    ]
    out = to_local_messages(msgs)
    assert out[0] == {"role": "system", "content": "be brief"} and out[1] == {
        "role": "user",
        "content": "hi there",
    }
    assert out[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}],
    }
    assert out[3] == {"role": "tool", "content": "42"}
    assert (
        to_local_messages(
            [
                {
                    "role": "assistant",
                    "content": "x",
                    "tool_calls": [
                        {"id": "c", "type": "function", "function": {"name": "g", "arguments": ""}}
                    ],
                }
            ]
        )[0]["tool_calls"][0]["function"]["arguments"]
        == {}
    )


def test_tool_calls_and_text_are_returned_in_chat_completions_shape(server):
    server.chat = StubChat(
        'Sure.\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call>\n<tool_call>\n{"name": "multiply", "arguments": {"a": 4, "b": 5}}\n</tool_call>',
        "The answer is 6.",
    )
    client = OpenAI(base_url=server.url, api_key="x")
    r = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": "go"}], tools=TOOLS
    )
    choice = r.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content == "Sure."
    calls = choice.message.tool_calls
    assert [json.loads(c.function.arguments) for c in calls] == [
        {"a": 2, "b": 3},
        {"a": 4, "b": 5},
    ] and len({c.id for c in calls}) == 2
    assert (
        r.usage.prompt_tokens == 100
        and r.usage.completion_tokens > 0
        and r.usage.total_tokens == 100 + r.usage.completion_tokens
    )
    final = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "go"},
            choice.message.model_dump(exclude_none=True),
            *[{"role": "tool", "tool_call_id": c.id, "content": "x"} for c in calls],
        ],
    )
    assert (
        final.choices[0].finish_reason == "stop"
        and final.choices[0].message.content == "The answer is 6."
        and final.choices[0].message.tool_calls is None
    )
    assert server.chat.seen[0]["tools"] == [t for t in TOOLS] and server.chat.seen[1]["messages"][
        1
    ]["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 3}


def test_tool_choice_none_hides_tools_and_max_tokens_is_capped(server):
    server.chat = StubChat("plain answer")
    client = OpenAI(base_url=server.url, api_key="x")
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "x"}],
        tools=TOOLS,
        tool_choice="none",
        max_tokens=100000,
    )
    assert server.chat.seen[0]["tools"] == [] and server.chat.seen[0]["max_new_tokens"] == 400


def test_unsupported_requests_fail_loudly(server):
    import httpx

    r = httpx.post(
        server.url + "/chat/completions", json={"model": "m", "messages": [], "stream": True}
    )
    assert r.status_code == 400 and "streaming" in r.json()["error"]["message"]
    r = httpx.post(server.url + "/responses", json={})
    assert r.status_code == 404 and "/v1/chat/completions" in r.json()["error"]["message"]
    assert httpx.get(server.url + "/models").json()["data"][0]["id"] == MODEL_NAME
    assert httpx.get(server.url + "/nope").status_code == 404
    bad = httpx.post(
        server.url + "/chat/completions", json={"model": "m"}
    )  # no messages: surfaces as HTTP 500 text
    assert bad.status_code == 500 and "KeyError" in bad.json()["error"]["message"]


def test_real_local_model_makes_a_tool_call_through_the_server():
    with LocalOpenAIServer() as srv:
        client = OpenAI(base_url=srv.url, api_key="x")
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What is 4123.5 times 89.2?"}],
            tools=TOOLS,
        )
        call = r.choices[0].message.tool_calls[0]
        assert call.function.name == "multiply" and json.loads(call.function.arguments) == {
            "a": 4123.5,
            "b": 89.2,
        }
        assert r.usage.prompt_tokens > 100


def test_json_mode_adds_an_instruction_with_the_schema_to_the_system_message(server):
    from common.local_server import json_instruction

    assert json_instruction(None) == "" and json_instruction({"type": "text"}) == ""
    hint = json_instruction(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "r",
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
    )
    assert "valid JSON object" in hint and '"x": {"type": "integer"}' in hint
    assert "JSON Schema" not in json_instruction({"type": "json_object"})
    server.chat = StubChat('{"x": 1}', '{"x": 2}')
    client = OpenAI(base_url=server.url, api_key="x")
    rf = {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "be a router"}, {"role": "user", "content": "go"}],
        response_format=rf,
    )
    first = server.chat.seen[0]["messages"]
    assert (
        first[0]["content"].startswith("be a router\n\nReply with ONLY a single valid JSON object")
        and len(first) == 2
    )
    client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": "go"}], response_format=rf
    )
    assert server.chat.seen[1]["messages"][0]["role"] == "system", (
        "a system message is inserted when there was none"
    )


def test_tool_choice_required_or_named_forces_the_reply_to_start_with_a_tool_call(server):
    server.chat = StubChat(
        'multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call>',
        '{"a": 1, "b": 1}}\n</tool_call>',
        "plain",
    )
    client = OpenAI(base_url=server.url, api_key="x")
    msgs = [{"role": "user", "content": "hello"}]
    r = client.chat.completions.create(
        model=MODEL_NAME, messages=msgs, tools=TOOLS, tool_choice="required"
    )
    assert server.chat.seen[0]["prefill"] == '<tool_call>\n{"name": "'
    assert (
        r.choices[0].finish_reason == "tool_calls"
        and r.choices[0].message.tool_calls[0].function.name == "multiply"
    ), "prefill + continuation parsed as one call"
    r2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "multiply"}},
    )
    assert server.chat.seen[1]["prefill"] == '<tool_call>\n{"name": "multiply", "arguments": '
    assert json.loads(r2.choices[0].message.tool_calls[0].function.arguments) == {"a": 1, "b": 1}
    client.chat.completions.create(model=MODEL_NAME, messages=msgs, tools=TOOLS, tool_choice="auto")
    assert server.chat.seen[2]["prefill"] == "", "auto leaves the model free"


def test_choice_name_helper():
    from common.local_server import _choice_name

    assert (
        _choice_name({"type": "function", "function": {"name": "f"}}) == "f"
        and _choice_name("required") == "required"
        and _choice_name(None) is None
    )
