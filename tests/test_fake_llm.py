from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from common import llm
from common.fake import fake_llm


class P(BaseModel):
    name: str


def test_rules_in_order_default_and_callable():
    with fake_llm([(r"capital", "Paris"), (r"echo: (\w+)", lambda p: p.upper())], default="?") as f:
        assert llm.complete("capital of France").text == "Paris"
        assert llm.complete("echo: hi").text == "ECHO: HI"
        assert llm.complete("unknown").text == "?"
        assert len(f.calls) == 3 and f.calls[0].kind == "complete"


def test_cycling_list_and_exceptions():
    with fake_llm([(r"x", ["bad", "good"]), (r"boom", RuntimeError("down"))]):
        assert [llm.complete("x").text for _ in range(3)] == ["bad", "good", "good"]
        with pytest.raises(RuntimeError):
            llm.complete("boom")


def test_system_and_messages_in_prompt_and_structured_stream_async():
    with fake_llm([(r"Ada", '{"name": "Ada"}')]) as f:
        obj, resp = llm.structured([{"role": "user", "content": "who? Ada"}], P, system="sys")
        assert obj == P(name="Ada") and "sys" in f.calls[-1].prompt
        assert "".join(llm.stream("Ada")) == '{"name": "Ada"}'
        assert asyncio.run(llm.acomplete("Ada")).text == '{"name": "Ada"}'
        assert llm.SESSION.calls >= 3


def test_patch_is_restored():
    original = llm.complete
    with fake_llm():
        assert llm.complete is not original
    assert llm.complete is original


def test_fake_turn_scripts_tool_calls_then_text_and_is_patched_on_chat():
    from common import chat
    from common.fake import tool_calls

    rules = [
        (r"tool results", "Final: 6"),
        (r"(?s).*", tool_calls(("multiply", {"a": 2, "b": 3}), text="Computing")),
    ]
    with fake_llm(rules) as f:
        t = chat.turn(
            [{"role": "user", "content": "2*3?"}],
            [{"name": "multiply", "description": "", "parameters": {}}],
        )
        assert t.wants_tools and t.text == "Computing" and t.tool_calls[0].args == {"a": 2, "b": 3}
        assert t.tool_calls[0].id != ""
        msgs = [
            {"role": "user", "content": "2*3?"},
            chat.assistant_message(t),
            chat.tool_message(t.tool_calls[0], "6"),
        ]
        t2 = chat.turn(msgs + [{"role": "user", "content": "tool results ready"}], None)
        assert t2.text == "Final: 6" and not t2.wants_tools
        assert "multiply({'a': 2, 'b': 3})" in f.calls[1].prompt and "6" in f.calls[1].prompt
    assert chat.turn.__module__ == "common.chat", "the patch is undone on exit"
