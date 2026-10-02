"""Tests for Week 10 Day 4: the probe graders, the evaluation helpers (intervals, paired comparison, error types), loading adapters, the forgetting measures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))

import blocks as B  # noqa: E402
import chatfmt as C  # noqa: E402
import evalrun as E  # noqa: E402
import gen_data as G  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402
import orders as O  # noqa: E402
import probes as P  # noqa: E402

CORRECT = [
    "The capital of France is Paris.",
    "Tokyo",
    "Mars.",
    "There are 7 days in a week.",
    "Green.",
    "William Shakespeare.",
    "H2O",
    "The Pacific Ocean.",
    "A spider has eight legs.",
    "Tuesday.",
    "12",
    "5",
    "18",
    "5",
    "42",
    "64",
    "63",
    "24",
    "YES",
    "BANANA.",
    "Forty-two",
    "apples, pears, plums",
    "Hola.",
    "Merci.",
    "TAC",
    "True.",
    "blue",
    "HELLO",
    '{"x": 1}',
    '{"name": "Sara"}',
    '{"city": "Madrid"}',
    '{"sentiment": "positive"}',
    '{"sentiment": "negative"}',
    '{"n": 7}',
    '{"color": "red"}',
    '{"animal": "dog"}',
]


# ----------------------------------------------------------------------------- the probes


def test_there_are_36_probes_in_four_kinds_each_with_a_correct_answer_that_passes():
    assert len(P.PROBES) == len(CORRECT) == 36
    assert {p.kind for p in P.PROBES} == {"facts", "arithmetic", "format", "json"}
    for probe, answer in zip(P.PROBES, CORRECT, strict=True):
        assert probe.passes(answer), (probe.prompt, answer)


def test_obviously_wrong_replies_fail_every_probe():
    for probe in P.PROBES:
        assert not probe.passes("I am not sure, sorry."), probe.prompt
        assert not probe.passes('{"is_order":true,"customer_name":null}'), probe.prompt
        assert not probe.passes(""), probe.prompt


def test_the_graders_are_strict_where_they_should_be():
    by_prompt = {p.prompt: p for p in P.PROBES}
    yes = by_prompt["Reply with exactly the word YES and nothing else."]
    assert by_prompt["Answer with true or false: the sun is a star."].passes(
        "True."
    ) and not by_prompt["Answer with true or false: the sun is a star."].passes('{"is_order":true}')
    assert yes.passes(" YES. ") and not yes.passes("YES, of course!") and not yes.passes("yes")
    caps = by_prompt["Write the word 'hello' in capital letters."]
    assert caps.passes("HELLO") and not caps.passes("hello")
    three = by_prompt["List three fruits separated by commas, and nothing else."]
    assert (
        not three.passes("apple, pear")
        and not three.passes("apple, pear, plum, fig")
        and three.passes("fig, kiwi, lime.")
    )
    js = by_prompt['Return JSON with a single key "x" whose value is 1.']
    assert (
        js.passes('```json\n{"x": 1}\n```')
        and not js.passes('{"x": 2}')
        and not js.passes("x: 1")
        and not js.passes("[1]")
    )
    assert by_prompt["What is 7 + 5?"].passes("it is 12") and not by_prompt[
        "What is 7 + 5?"
    ].passes("112")


def test_the_order_json_signature_detects_an_overfitted_reply():
    assert P.emits_order_json('{"is_order":true,"customer_name":null}') and P.emits_order_json(
        '  {"is_order":false}'
    )
    assert not P.emits_order_json('{"x":1}') and not P.emits_order_json("Paris")


# ----------------------------------------------------------------------------- evaluation helpers


def result(reply, gold, email="e"):
    return I.Result(email, gold, reply, O.score(reply, gold), 100, 50, 1.0)


GOLDS = [g for _, g in O.HUMAN_EMAILS[:6]]


def test_records_round_trip_back_to_the_gold_labels():
    samples = G.generate(30, seed=2)
    recs = [{"messages": I.training_messages(s.email, s.gold)} for s in samples]
    assert E.records_to_pairs(recs) == [
        (s.email, {**s.gold, "items": [tuple(i) for i in s.gold["items"]]}) for s in samples
    ]


def test_summaries_carry_wilson_intervals_and_costs():
    results = [result(O.order_json(g), g) for g in GOLDS[:5]] + [result("junk", GOLDS[5])]
    s = E.summarize_results(results)
    assert (
        s["n"] == 6
        and s["exact"] == pytest.approx(5 / 6)
        and s["valid_json"] == pytest.approx(5 / 6)
    )
    p, lo, hi = s["exact_ci"]
    assert (
        p == pytest.approx(5 / 6)
        and 0.4 < lo < p < hi <= 1.0
        and s["prompt_tokens"] == 100
        and s["new_tokens"] == 50
    )


def test_paired_comparison_counts_wins_losses_and_ties_per_email():
    good = [result(O.order_json(g), g) for g in GOLDS]
    bad = [result("nope", g) for g in GOLDS]
    mixed = good[:3] + bad[3:]
    c = E.compare(good, mixed)
    assert (c["wins"], c["losses"], c["ties"]) == (3, 0, 3) and c["diff"] == pytest.approx(0.5)
    assert E.compare(mixed, good)["diff"] == pytest.approx(-0.5)
    f = E.compare(good, bad, "fields")
    assert f["diff"] == pytest.approx(1.0) and f["p"] < 0.05


def test_partial_credit_metric_counts_the_fraction_of_fields_right():
    g = GOLDS[0]
    one_wrong = result(O.order_json(g).replace(g["order_id"], "ZZ-1"), g)
    c = E.compare([one_wrong], [result("nope", g)], "fields")
    assert c["diff"] == pytest.approx(7 / 8)


def test_error_types_separate_not_json_invalid_order_and_wrong_fields():
    g = GOLDS[0]
    good = result(O.order_json(g), g)
    wrong = result(O.order_json(g).replace(g["order_id"], "ZZ-1"), g)
    invalid = result('{"is_order":true,"items":[]}', g)
    junk = result("not json at all", g)
    e = E.error_types([good, wrong, wrong, invalid, junk])
    assert (e["exact"], e["wrong_fields"], e["invalid_order"], e["not_json"]) == (1, 2, 1, 1) and e[
        "wrong:order_id"
    ] == 2


def test_rate_with_ci_matches_the_wilson_interval():
    p, lo, hi = E.rate_with_ci(0, 38)
    assert p == 0 and lo == 0 and 0.05 < hi < 0.15, (
        "even 0 of 38 leaves room for a true rate near 8%"
    )
    assert E.fmt_ci((0.5, 0.4, 0.6)) == "50% [40%, 60%]"


# ----------------------------------------------------------------------------- adapters and the forgetting measures


def tiny(seed=0):
    torch.manual_seed(seed)
    m = B.Decoder(
        B.Config(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    ).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    return m


def test_load_tuned_rebuilds_the_adapted_model_and_merges_it(tmp_path):
    src = tiny()
    L.add_lora(src, r=4, alpha=8, targets=("q_proj", "v_proj"))
    with torch.no_grad():
        for n, p in src.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.2)
    torch.save(
        {
            "state": L.lora_state_dict(src),
            "r": 4,
            "alpha": 8,
            "targets": ["q_proj", "v_proj"],
            "epoch": 1,
        },
        tmp_path / "a.pt",
    )
    ids = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        want = src(ids)
    model, _ = E.load_tuned(tmp_path / "a.pt", base=(tiny(), None))
    assert not L.lora_modules(model), "merged"
    with torch.no_grad():
        assert torch.allclose(model(ids), want, atol=1e-5)
    unmerged, _ = E.load_tuned(tmp_path / "a.pt", base=(tiny(), None), merge=False)
    assert len(L.lora_modules(unmerged)) == 4


def test_prose_loss_is_finite_and_lower_for_a_model_that_has_seen_the_text():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(I.BASE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {exc}")
    cfg = B.Config(
        vocab_size=49152, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=256
    )
    torch.manual_seed(0)
    m = B.Decoder(cfg).eval()
    text = "the cat sat on the mat. " * 200
    before = E.text_loss(m, tok, text, window=64, max_tokens=1500)
    assert before == pytest.approx(10.8, abs=1.0)
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"][:1500])
    n = (len(ids) - 1) // 64 * 64
    x, y = ids[:n].view(-1, 64), ids[1 : n + 1].view(-1, 64)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(25):
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, 49152), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    assert E.text_loss(m, tok, text, window=64, max_tokens=1500) < before - 3


def test_run_probes_reports_per_kind_accuracy_and_the_order_json_rate():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(I.BASE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {exc}")
    cfg = B.Config(
        vocab_size=49152, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=512
    )
    torch.manual_seed(1)
    out = E.run_probes(B.Decoder(cfg).eval(), tok, max_new_tokens=3)
    assert (
        out["n"] == 36
        and set(out["by_kind"]) == {"facts", "arithmetic", "format", "json"}
        and 0 <= out["accuracy"] <= 1
        and out["order_json_rate"] == 0
    )
    assert all(isinstance(reply, str) for _, reply, *_ in out["outcomes"])


def test_the_training_prompt_is_the_short_one_and_validates():
    msgs = I.training_messages("Hi, order A-1: 2 lamps. Sam", O.HUMAN_EMAILS[0][1])
    C.validate_messages(msgs)
    assert msgs[0]["content"] == O.SYSTEM_SHORT and len(msgs) == 3
