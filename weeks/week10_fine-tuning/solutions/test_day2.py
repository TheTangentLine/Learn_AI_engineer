"""Tests for Week 10 Day 2: the generator, the label checks (each defect kind), deduplication, decontamination, balance and the split."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day2_solution as d2  # noqa: E402
import gen_data as G  # noqa: E402
import orders as O  # noqa: E402

RAW = G.generate(1200, seed=11, defect_rate=0.0)
ORDERS = [s for s in RAW if s.gold["is_order"]]


# ----------------------------------------------------------------------------- the generator


def test_generation_is_deterministic_and_seeded():
    assert [(s.email, s.gold) for s in G.generate(50, seed=3)] == [
        (s.email, s.gold) for s in G.generate(50, seed=3)
    ]
    assert [s.email for s in G.generate(50, seed=3)] != [s.email for s in G.generate(50, seed=4)]


def test_every_clean_label_is_a_valid_order_and_a_non_order_has_no_fields():
    for s in RAW:
        O.check_gold(s.gold)
        assert not s.defect
        if s.template == "non_order":
            assert not s.gold["is_order"]


def test_clean_samples_always_pass_the_label_checks():
    assert [G.failures(s) for s in G.generate(6000, seed=21) if G.failures(s)] == []


def test_the_generator_covers_the_layouts_urgencies_currencies_and_non_orders():
    assert {s.template for s in RAW} == {"prose", "list", "form", "terse", "non_order"}
    assert {s.gold["urgency"] for s in ORDERS} == {"low", "normal", "high"} and {
        s.gold["currency"] for s in ORDERS
    } == {None, "USD", "EUR", "GBP"}
    share = sum(not s.gold["is_order"] for s in RAW) / len(RAW)
    assert 0.08 < share < 0.2
    assert len({s.email for s in RAW}) > 0.97 * len(RAW), "very few exact repeats"


def test_a_first_name_only_email_gets_a_first_name_only_label():
    firsts = [s for s in ORDERS if " " not in s.gold["customer_name"]]
    assert firsts, "the generator produces some"
    for s in firsts:
        assert s.gold["customer_name"] in s.email


def test_every_delivery_date_is_after_the_received_date():
    for s in ORDERS:
        if s.gold["delivery_date"]:
            assert date.fromisoformat(s.gold["delivery_date"]) >= G.EARLIEST


def test_no_generated_email_equals_a_hand_written_one():
    human = {e for e, _ in O.HUMAN_EMAILS + O.FEW_SHOT_EXAMPLES}
    assert not human & {s.email for s in RAW}


# ----------------------------------------------------------------------------- the label checks


def test_every_kind_of_injected_defect_is_caught_by_some_check():
    caught = {
        k: (n, c) for k, n, c in d2.defect_catch_table(G.generate(2500, seed=5, defect_rate=0.2))
    }
    assert set(caught) == set(G.DEFECTS)
    for kind, (n, c) in caught.items():
        assert n > 10 and c == n, f"{kind}: {c} of {n} caught"


def test_the_completeness_check_catches_an_omitted_item_which_the_grounding_checks_cannot():
    email = "Hi, order C-9 for Dana Lee: 3 desk lamps and 5 notebooks. Thanks"
    full = dict(
        is_order=True,
        customer_name="Dana Lee",
        order_id="C-9",
        items=[("desk lamps", 3), ("notebooks", 5)],
        urgency="normal",
        delivery_date=None,
        total_amount=None,
        currency=None,
    )
    assert G.failures(G.Sample(email, full, "t")) == []
    dropped = {**full, "items": [("desk lamps", 3)]}
    assert G.failures(G.Sample(email, dropped, "t")) == ["number_in_email_not_in_label"]
    assert G.unexplained_numbers(email, dropped) == ["5"]


def test_each_individual_check_by_hand():
    email = "Order K-7, Zed Park: 4 mugs. Deliver by 12 December 2026. Total GBP 40. ASAP."
    ok = dict(
        is_order=True,
        customer_name="Zed Park",
        order_id="K-7",
        items=[("mugs", 4)],
        urgency="high",
        delivery_date="2026-12-12",
        total_amount=40.0,
        currency="GBP",
    )
    assert G.failures(G.Sample(email, ok, "t")) == []
    cases = {
        "order_id_not_in_email": {**ok, "order_id": "K-8"},
        "name_not_in_email": {**ok, "customer_name": "Zoe Park"},
        "item_not_in_email": {**ok, "items": [("lamps", 4)]},
        "quantity_not_in_email": {**ok, "items": [("mugs", 9)]},
        "date_not_in_email": {**ok, "delivery_date": "2026-12-13"},
        "total_not_in_email": {**ok, "total_amount": 41.0},
        "currency_not_in_email": {**ok, "currency": "EUR"},
        "urgency_high_without_cue": None,
        "urgency_normal_but_cue_present": {**ok, "urgency": "normal"},
    }
    for name, label in cases.items():
        if label is None:
            plain = "Order K-7, Zed Park: 4 mugs."
            assert "urgency_high_without_cue" in G.failures(
                G.Sample(
                    plain,
                    {**ok, "delivery_date": None, "total_amount": None, "currency": None},
                    "t",
                )
            )
            continue
        assert name in G.failures(G.Sample(email, label, "t")), name
    past = "Order K-7, Zed Park: 4 mugs. Deliver by 1 September 2026."
    assert G.failures(
        G.Sample(
            past,
            {
                **ok,
                "delivery_date": "2026-09-01",
                "total_amount": None,
                "currency": None,
                "urgency": "normal",
            },
            "t",
        )
    ) == ["date_before_received"]
    assert "non_order_has_fields" in G.failures(G.Sample("hello", {**ok, "is_order": False}, "t"))


def test_a_low_priority_phrase_that_contains_the_word_urgent_is_not_an_urgency_cue():
    email = "Order K-7, Zed Park: 4 mugs. Nothing urgent."
    label = dict(
        is_order=True,
        customer_name="Zed Park",
        order_id="K-7",
        items=[("mugs", 4)],
        urgency="low",
        delivery_date=None,
        total_amount=None,
        currency=None,
    )
    assert G.failures(G.Sample(email, label, "t")) == []
    assert "urgency_high_without_cue" in G.failures(
        G.Sample(email, {**label, "urgency": "high"}, "t")
    )


def test_date_and_money_readers():
    assert G.find_dates(
        "by 5 Nov 2026 and November 6th, 2026 and the 7th of November 2026 and 2026-11-08, also nov 9, 2026"
    ) == {"2026-11-05", "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09"}
    assert G.find_dates("on 31 February 2026") == set()
    assert G.money_values("$1,234.50 or 142,50 EUR or 75 or 3.5") == {1234.5, 142.5, 75.0, 3.5}


# ----------------------------------------------------------------------------- deduplication, contamination


def test_exact_and_near_duplicates_are_removed_and_distinct_emails_are_kept():
    a = G.Sample(
        "Hi, order A-1 please: 3 desk lamps and 2 notebooks. Thanks Sam", O.HUMAN_EMAILS[0][1], "t"
    )
    exact = G.Sample(
        "hi order a-1 please: 3 desk lamps and 2 notebooks thanks sam!", O.HUMAN_EMAILS[0][1], "t"
    )
    near = G.Sample(
        "Hi, order A-1 please: 3 desk lamps and 2 notebooks. Thanks Sam, regards",
        O.HUMAN_EMAILS[0][1],
        "t",
    )
    other = G.Sample(
        "Completely different email about cancelling something entirely else today",
        O.HUMAN_EMAILS[0][1],
        "t",
    )
    kept, n_exact, n_near = G.dedup([a, exact, near, other])
    assert [s.email for s in kept] == [a.email, other.email] and (n_exact, n_near) == (1, 1)


def test_lsh_finds_the_near_duplicate_pairs_that_exact_comparison_finds():
    texts = [s.email for s in G.generate(500, seed=2)]
    truth = {
        (i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if G.jaccard(G.shingles(texts[i]), G.shingles(texts[j])) >= 0.8
    }
    found = G.near_duplicate_pairs(texts, 0.8)
    assert found <= truth, "every pair it reports is verified with the exact Jaccard"
    assert len(found) >= 0.9 * len(truth)


def test_minhash_agreement_estimates_the_jaccard_similarity():
    a, b = (
        G.shingles("the quick brown fox jumps over the lazy dog near the river bank today"),
        G.shingles("the quick brown fox jumps over the lazy dog near the river bank tonight"),
    )
    sa, sb = G.minhash(a, 256), G.minhash(b, 256)
    assert sum(x == y for x, y in zip(sa, sb, strict=True)) / 256 == pytest.approx(
        G.jaccard(a, b), abs=0.1
    )


def test_decontamination_removes_samples_that_share_a_long_sequence_with_the_evaluation_set():
    ref = "Please process order H-410 for Jamal Brooks. We need 8 ergonomic chairs"
    a = G.Sample(
        "Hello. Please process order H-410 for Jamal Brooks. We need 8 ergonomic chairs and more",
        O.HUMAN_EMAILS[0][1],
        "t",
    )
    b = G.Sample(
        "Please process order H-411 for Someone Else. We need 9 stools", O.HUMAN_EMAILS[0][1], "t"
    )
    kept, n = G.decontaminate([a, b], [ref], n=8)
    assert [s.email for s in kept] == [b.email] and n == 1


# ----------------------------------------------------------------------------- the pipeline


@pytest.fixture(scope="module")
def built():
    return d2.build(seed=0)


def test_the_pipeline_catches_every_injected_defect_and_removes_no_clean_sample(built):
    raw, kept, report, _ = built
    assert (
        report.defects_injected > 60
        and report.defects_caught == report.defects_injected
        and report.defects_missed == 0
    )
    assert report.false_rejections == 0 and report.generated == len(raw) == 1500
    removed = sum(n for _, n in report.stages)
    assert report.generated - removed == report.kept == len(kept)
    assert all(not s.defect for s in kept)


def test_the_report_counts_defects_that_slip_through_as_missed():
    clean = G.generate(40, seed=8)
    orders = [s for s in clean if s.gold["is_order"]]
    flagged_but_fine = G.Sample(
        orders[0].email + " extra words to be different from the original email text",
        orders[0].gold,
        orders[0].template,
        "wrong_urgency",
    )
    bad_label, kind = G.inject_defect(orders[1].gold, __import__("random").Random(1))
    caught = G.Sample(orders[1].email, bad_label, orders[1].template, kind)
    _, rep = G.curate([flagged_but_fine, caught, *orders[2:6]])
    assert (
        rep.defects_injected == 2
        and rep.defects_caught == 1
        and rep.defects_missed == 1
        and rep.false_rejections == 0
    )


def test_the_splits_are_disjoint_the_right_size_and_free_of_evaluation_text(built):
    _, kept, _, (train, dev, test) = built
    assert (len(dev), len(test)) == (100, 150) and len(train) == len(kept) - 250 > 1000
    texts = [{s.email for s in part} for part in (train, dev, test)]
    assert not texts[0] & texts[1] and not texts[0] & texts[2] and not texts[1] & texts[2]
    for s in train + dev + test:
        assert not any(G.ngram_overlap(s.email, e) for e in d2.eval_texts())


def test_the_label_distribution_is_roughly_what_the_generator_asked_for(built):
    d = G.distribution(built[3][0])
    assert (
        0.8 < d["is_order"][True] < 0.92
        and set(d["urgency"]) == {"low", "normal", "high"}
        and 0.1 < d["urgency"]["low"] < 0.3
    )
    assert d["template"]["prose"] > d["template"]["form"]


def test_split_is_deterministic_and_covers_everything_once():
    samples = G.generate(60, seed=1)
    a = G.split(samples, 10, 10, seed=4)
    assert a == G.split(samples, 10, 10, seed=4) and a != G.split(samples, 10, 10, seed=5)
    assert sorted(id(s) for part in a for s in part) == sorted(id(s) for s in samples)


def test_records_are_valid_chat_conversations_whose_answer_is_the_gold_json(built):
    import chatfmt as C

    for s in built[3][0][:50]:
        rec = d2.to_record(s)
        C.validate_messages(rec["messages"])
        assert (
            rec["messages"][-1]["content"] == O.order_json(s.gold)
            and rec["messages"][1]["content"] == s.email
        )
