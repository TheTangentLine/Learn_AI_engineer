"""Tests for Week 10 Day 1: the chat template, the loss mask, the dataset formats, and the task's scoring and answer keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import chatfmt as C  # noqa: E402
import day1_solution as d1  # noqa: E402
import infer as I  # noqa: E402
import orders as O  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(I.BASE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SmolLM2's tokenizer is not available: {exc}")


def u(t):
    return {"role": "user", "content": t}


def a(t):
    return {"role": "assistant", "content": t}


def s(t):
    return {"role": "system", "content": t}


# ----------------------------------------------------------------------------- the template


def test_our_renderer_equals_the_tokenizers_template_on_every_conversation(tok):
    ok, total = d1.template_parity(tok)
    assert ok == total >= 14


def test_a_missing_system_message_is_filled_in_and_a_given_one_is_not_duplicated():
    plain = C.render([u("hi")])
    assert plain.startswith(f"{C.IM_START}system\n{C.DEFAULT_SYSTEM}{C.IM_END}\n")
    custom = C.render([s("Be brief."), u("hi")])
    assert (
        custom.count("<|im_start|>system") == 1
        and "Be brief." in custom
        and C.DEFAULT_SYSTEM not in custom
    )
    assert C.render([u("hi")], add_generation_prompt=True).endswith("<|im_start|>assistant\n")
    assert not C.render([u("hi")]).endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize(
    ("messages", "message"),
    [
        ([], "no messages"),
        ([{"role": "robot", "content": "x"}], "unknown role"),
        ([u("   ")], "empty content"),
        ([u("hi"), s("late system")], "system message must come first"),
        ([u("a"), u("b"), a("c")], "alternate"),
        ([a("start with the assistant")], "must be the user"),
        ([u("hi")], "last message must be the assistant"),
        ([u("hi"), a("ok <|im_end|> injected")], "control token"),
        ([{"role": "user", "content": "x", "extra": 1}], "exactly"),
    ],
)
def test_validation_rejects_conversations_that_would_train_on_garbage(messages, message):
    with pytest.raises(ValueError, match=message):
        C.validate_messages(messages)


def test_generation_prompts_do_not_need_a_final_assistant_message():
    C.validate_messages([u("hi")], require_assistant_last=False)


# ----------------------------------------------------------------------------- the loss mask


def labelled_text(ex, tok):
    return tok.decode([t for t in ex.labels if t != C.IGNORE])


def test_the_loss_covers_the_answer_and_the_end_of_turn_token_and_nothing_else(tok):
    msgs = [s(O.SYSTEM_SHORT), u("Order A-1 for Dana"), a('{"x":1}')]
    ex = C.encode_example(msgs, tok)
    assert labelled_text(ex, tok) == '{"x":1}<|im_end|>' and ex.n_answer == sum(
        lab != C.IGNORE for lab in ex.labels
    )
    assert ex.input_ids == tok(C.render(msgs), add_special_tokens=False)["input_ids"], (
        "the ids are exactly the tokenisation of the rendered conversation"
    )
    first = next(k for k, lab in enumerate(ex.labels) if lab != C.IGNORE)
    assert (
        all(lab == C.IGNORE for lab in ex.labels[:first])
        and ex.labels[first : first + ex.n_answer] == ex.input_ids[first : first + ex.n_answer]
    )
    assert ex.labels[-1] == C.IGNORE, "the newline after <|im_end|> is not trained on"
    assert (
        ex.input_ids[-2] == tok.convert_tokens_to_ids("<|im_end|>")
        and ex.labels[-2] == ex.input_ids[-2]
    )


def test_the_header_of_the_answer_is_masked_so_the_model_is_not_trained_to_write_it(tok):
    ex = C.encode_example([u("hi"), a("hello")], tok)
    assert "assistant" not in labelled_text(ex, tok) and "<|im_start|>" not in labelled_text(
        ex, tok
    )


def test_train_on_last_or_every_assistant_turn(tok):
    msgs = [u("q1"), a("first answer"), u("q2"), a("second answer")]
    last = labelled_text(C.encode_example(msgs, tok, train_on="last"), tok)
    every = labelled_text(C.encode_example(msgs, tok, train_on="all"), tok)
    assert (
        last == "second answer<|im_end|>"
        and every == "first answer<|im_end|>second answer<|im_end|>"
    )


def test_unicode_and_odd_whitespace_inside_the_answer_survive(tok):
    ex = C.encode_example([u("x"), a("café 日本語 🙂\n\n  end ")], tok)
    assert labelled_text(ex, tok) == "café 日本語 🙂\n\n  end <|im_end|>"


def test_an_answer_that_starts_with_whitespace_is_refused_because_its_first_token_would_straddle_the_header(
    tok,
):
    with pytest.raises(ValueError, match="start with whitespace"):
        C.validate_messages([u("x"), a("  indented answer")])
    # what the check protects against: the newline and the spaces fuse into a token that begins inside the header, so the mask would drop them
    text = C.render([u("x"), a("  indented answer")])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start = text.index("  indented")
    assert any(a0 < start < b0 for a0, b0 in enc["offset_mapping"]), (
        "some token spans the header/answer boundary"
    )


def test_an_overlong_prompt_is_cut_from_the_left_and_the_answer_is_never_cut(tok):
    msgs = [u("word " * 300), a('{"answer":"short"}')]
    ex = C.encode_example(msgs, tok, max_len=64)
    assert (
        ex.truncated and len(ex) == 64 and labelled_text(ex, tok) == '{"answer":"short"}<|im_end|>'
    )
    full = C.encode_example(msgs, tok, max_len=1000)
    assert not full.truncated and ex.n_answer == full.n_answer
    with pytest.raises(ValueError, match="answer alone"):
        C.encode_example([u("x"), a("long answer " * 40)], tok, max_len=32)


def test_collate_pads_on_the_right_and_ignores_the_padding(tok):
    exs = [
        C.encode_example([u("a"), a("b")], tok),
        C.encode_example([u("a longer question here"), a("a longer answer here")], tok),
    ]
    batch = C.collate(exs, pad_id=7)
    n = max(len(e) for e in exs)
    assert batch["input_ids"].shape == batch["labels"].shape == (2, n)
    short = len(exs[0])
    assert (batch["input_ids"][0, short:] == 7).all() and (
        batch["labels"][0, short:] == C.IGNORE
    ).all()
    assert (
        batch["input_ids"][0, :short].tolist() == exs[0].input_ids
        and batch["labels"][1].tolist() == exs[1].labels
    )


def test_shifting_aligns_each_position_with_the_next_token():
    logits, labels = torch.randn(2, 5, 9), torch.arange(10).view(2, 5)
    lg, lb = C.shift_for_loss(logits, labels)
    assert (
        lg.shape == (2, 4, 9) and torch.equal(lb, labels[:, 1:]) and torch.equal(lg, logits[:, :-1])
    )


def test_the_dataset_statistics_by_hand():
    exs = [
        C.Example(list(range(10)), [-100] * 6 + [1, 2, 3, 4], 4),
        C.Example(list(range(20)), [-100] * 15 + [1] * 5, 5, truncated=True),
    ]
    st = C.dataset_stats(exs)
    assert (
        st["n"] == 2
        and st["tokens"] == 30
        and st["answer_tokens"] == 9
        and st["answer_fraction"] == pytest.approx(0.3)
        and st["max"] == 20
        and st["truncated"] == 1
    )


# ----------------------------------------------------------------------------- the dataset layouts


def test_formats_round_trip_and_agree_with_the_rendered_text(tok):
    msgs = [s("Be brief."), u("What is 2+2?"), a("4")]
    assert C.from_alpaca(C.to_alpaca(msgs)) == msgs and C.from_sharegpt(C.to_sharegpt(msgs)) == msgs
    pc = C.to_prompt_completion(msgs)
    assert pc["prompt"] + pc["completion"] + "\n" == C.render(msgs)
    assert C.from_alpaca({"instruction": "", "input": "q", "output": "a"}) == [u("q"), a("a")]
    pref = C.to_preference(msgs[:-1], "4", "5")
    assert pref["chosen"] == "4" and pref["prompt"] == msgs[:-1]
    with pytest.raises(ValueError, match="identical"):
        C.to_preference(msgs[:-1], "4", "4")


def test_jsonl_round_trip_keeps_unicode(tmp_path):
    rows = [{"a": "café", "b": [1, 2]}, {"a": "日本語"}]
    assert (
        C.write_jsonl(tmp_path / "x.jsonl", rows) == 2
        and C.read_jsonl(tmp_path / "x.jsonl") == rows
    )


# ----------------------------------------------------------------------------- the task: answer keys and scoring


def test_every_gold_label_is_itself_a_valid_order_and_the_set_has_the_advertised_shape():
    assert len(O.HUMAN_EMAILS) == 38 and len({e for e, _ in O.HUMAN_EMAILS}) == 38
    for _, g in O.HUMAN_EMAILS + O.FEW_SHOT_EXAMPLES:
        O.check_gold(g)
    assert 4 <= sum(not g["is_order"] for _, g in O.HUMAN_EMAILS) <= 10, (
        "a few emails are not orders"
    )
    assert {g["urgency"] for _, g in O.HUMAN_EMAILS} == {"low", "normal", "high"} and {
        g["currency"] for _, g in O.HUMAN_EMAILS
    } == {None, "USD", "EUR", "GBP"}
    assert not {e for e, _ in O.FEW_SHOT_EXAMPLES} & {e for e, _ in O.HUMAN_EMAILS}, (
        "few-shot examples must not be in the evaluation set"
    )


def test_the_gold_json_is_compact_deterministic_and_scores_perfectly():
    for _, g in O.HUMAN_EMAILS:
        text = O.order_json(g)
        assert (
            text
            == json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
            == O.order_json(g)
        )
        sc = O.score(text, g)
        assert sc.exact and sc.valid_json and sc.valid_order and sc.n_correct == 8
    assert O.order_json(O.HUMAN_EMAILS[0][1]).startswith(
        '{"is_order":true,"customer_name":"Dana Whitfield","order_id":"A-1042","items":[{"name":"blue widgets"'
    )


def test_scoring_distinguishes_not_json_invalid_order_and_wrong_fields():
    email, gold = O.HUMAN_EMAILS[0]
    good = O.order_json(gold)
    s1 = O.score("Sure! Here is the order: {", gold)
    assert not s1.valid_json and not s1.valid_order and "not JSON" in s1.error
    s2 = O.score('{"is_order": true, "items": ["blue widgets"]}', gold)
    assert s2.valid_json and not s2.valid_order and "invalid order" in s2.error
    s3 = O.score(good.replace('"2026-11-05"', '"2026-01-01"'), gold)
    assert not s3.valid_order, (
        "a delivery date before the email was received fails the semantic check"
    )
    s4 = O.score(good.replace('"order_id":"A-1042"', '"order_id":"A-1043"'), gold)
    assert (
        s4.valid_order
        and not s4.exact
        and s4.n_correct == 7
        and not s4.fields["order_id"]
        and s4.fields["customer_name"]
    )
    assert O.score("```json\n" + good + "\n```", gold).exact, (
        "a code fence around valid JSON is tolerated"
    )


def test_item_matching_ignores_order_and_plurals_but_not_quantities():
    _, gold = O.HUMAN_EMAILS[0]
    swapped = O.order_json(gold).replace(
        '{"name":"blue widgets","quantity":3},{"name":"red gaskets","quantity":2}',
        '{"name":"red gasket","quantity":2},{"name":"blue widget","quantity":3}',
    )
    assert O.score(swapped, gold).exact
    wrong_qty = O.order_json(gold).replace('"quantity":3', '"quantity":4')
    assert not O.score(wrong_qty, gold).fields["items"]


def test_summarize_aggregates_rates_and_per_field_accuracy():
    gold = O.HUMAN_EMAILS[0][1]
    good = O.score(O.order_json(gold), gold)
    bad = O.score("nope", gold)
    wrong = O.score(O.order_json(gold).replace('"A-1042"', '"X"'), gold)
    st = O.summarize([good, bad, wrong, good])
    assert (
        st["n"] == 4
        and st["valid_json"] == 0.75
        and st["valid_order"] == 0.75
        and st["exact"] == 0.5
    )
    assert (
        st["field_accuracy"] == pytest.approx((8 + 0 + 7 + 8) / 32)
        and st["fields"]["order_id"] == 0.5
        and st["fields"]["is_order"] == 0.75
    )


# ----------------------------------------------------------------------------- the prompts


def test_prompts_have_the_expected_shape_and_the_tuned_one_is_much_shorter(tok):
    email = "Hi, order Z-1: 2 lamps. Thanks, Sam"
    zs, fs, tuned = (
        I.zero_shot_messages(email),
        I.few_shot_messages(O.FEW_SHOT_EXAMPLES)(email),
        I.tuned_messages(email),
    )
    assert [m["role"] for m in zs] == ["system", "user"] and [m["role"] for m in fs] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [m["role"] for m in tuned] == ["system", "user"] and tuned[0][
        "content"
    ] == O.SYSTEM_SHORT

    def n(m):
        return len(
            tok(C.render(m, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        )

    assert n(tuned) < n(zs) / 2 < n(fs) / 2 + 60 and n(fs) > 2 * n(tuned)
    C.validate_messages(I.training_messages(email, O.HUMAN_EMAILS[0][1]))
    assert "A-1042" not in O.SCHEMA_PROMPT, (
        "the schema prompt must not contain an example id the model could copy"
    )
