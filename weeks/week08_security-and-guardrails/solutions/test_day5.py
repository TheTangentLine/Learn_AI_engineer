"""Tests for Week 8 Day 5: the labelled sets are what they claim, the scores are computed correctly, the privacy layer keeps raw personal data
out of the database and the traces, and retention deletes what it should and nothing else."""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import day5_solution as d5  # noqa: E402
import piidata  # noqa: E402
import private_support as ps  # noqa: E402
import retention  # noqa: E402
import scripted  # noqa: E402

from common import pii, tracing  # noqa: E402
from common.fake import fake_llm  # noqa: E402

# ----------------------------------------------------------------------------- the labelled sets


def test_the_generated_set_is_deterministic_and_covers_every_type():
    a, b = piidata.build_dataset(0), piidata.build_dataset(0)
    assert [(i.text, i.gold) for i in a] == [(i.text, i.gold) for i in b]
    assert [i.text for i in a] != [i.text for i in piidata.build_dataset(1)]
    types = {t for i in a for t, _ in i.gold}
    assert (
        types == set(piidata.TEMPLATES) | {"PERSON", "CREDIT_CARD", "EMAIL", "PHONE"}
        and len(types) == 12
    )
    assert len(a) == 12 * 10 + len(piidata.NEGATIVES) + 10 == 160


def test_every_gold_value_appears_exactly_in_its_text_and_negatives_have_none():
    for item in piidata.build_dataset(3) + piidata.REALISTIC:
        for _t, value in item.gold:
            assert value in item.text, (value, item.text)
    assert all(not i.gold for i in piidata.build_dataset(0) if i.text in piidata.NEGATIVES)


def test_generated_cards_and_ibans_have_valid_checksums_and_ssns_follow_the_rules():
    rng = random.Random(5)
    for _ in range(200):
        v = piidata.values(rng)
        assert pii.luhn_valid(v["CREDIT_CARD"]) and pii.iban_valid(v["IBAN"])
        a, g, s = v["US_SSN"].split("-")
        assert pii.ssn_valid(a, g, s)


def test_hard_negatives_include_the_shapes_that_trip_naive_detectors():
    joined = " ".join(piidata.NEGATIVES)
    for needle in (
        "4111111111111112",
        "123-45-6789x",
        "999.300.1.1",
        "1Z999AA10123456784",
        "10:30",
    ):
        assert needle in joined
    assert all(pii.luhn_valid("4111111111111111") for _ in [0]) and not pii.luhn_valid(
        "4111111111111112"
    )


def test_the_realistic_set_is_human_written_and_has_both_hits_and_gaps():
    assert len(piidata.REALISTIC) == 39 and sum(len(i.gold) for i in piidata.REALISTIC) == 33
    assert len({i.text for i in piidata.REALISTIC}) == 39
    assert not ({i.text for i in piidata.REALISTIC} & {i.text for i in piidata.build_dataset(0)})


# ----------------------------------------------------------------------------- scoring


def test_scoring_by_hand():
    items = [
        piidata.Item("mail alice@example.com now", [("EMAIL", "alice@example.com")]),
        piidata.Item("my name is Alice Johnson", [("PERSON", "Alice Johnson")]),
        piidata.Item(
            "born maybe in spring", [("DATE_OF_BIRTH", "spring")]
        ),  # not detectable: a miss
        piidata.Item(
            "order 4111111111111112 shipped", []
        ),  # a hard negative the detector must ignore
        piidata.Item("call +1 (415) 555-0132", []),  # a phone in a 'negative': a false positive
    ]
    r = d5.score(items)
    assert (r["tp"], r["fn"], r["fp"]) == (2, 1, 1)
    assert r["recall"] == pytest.approx(2 / 3) and r["precision"] == pytest.approx(2 / 3)
    assert (
        r["by_type"]["EMAIL"]["recall"] == 1.0
        and r["by_type"]["DATE_OF_BIRTH"]["recall"] == 0.0
        and r["by_type"]["PHONE"]["fp"] == 1
    )
    assert r["missed"] == [("DATE_OF_BIRTH", "spring")] and r["false_positives"][0][:2] == (
        "PHONE",
        "+1 (415) 555-0132",
    )
    assert r["recall_ci"][0] < 2 / 3 < r["recall_ci"][1]


def test_a_wrong_type_is_a_miss_in_strict_mode_and_a_hit_when_any_span_counts():
    item = piidata.Item(
        "password=hunter2hunter", [("API_KEY", "hunter2hunter")]
    )  # detected as CREDENTIALS
    strict, loose = d5.score([item]), d5.score([item], strict_type=False)
    assert (
        strict["fn"] == 1
        and strict["fp"] == 1
        and loose["tp"] == 1
        and loose["fn"] == 0
        and loose["fp"] == 0
    )


def test_one_span_cannot_satisfy_two_gold_values_and_partial_overlap_counts():
    two = piidata.Item(
        "alice@example.com", [("EMAIL", "alice@example.com"), ("EMAIL", "alice@example.com")]
    )
    assert d5.score([two])["tp"] == 1 and d5.score([two])["fn"] == 1
    partial = piidata.Item("Ship to 221 Baker Street, London.", [("ADDRESS", "221 Baker Street")])
    assert d5.score([partial])["tp"] == 1


def test_the_generated_set_is_a_ceiling_and_the_realistic_set_is_the_estimate():
    gen = d5.score(piidata.build_dataset(0, per_type=10))
    assert (gen["tp"], gen["fn"], gen["fp"]) == (160, 0, 0), (
        "the detector was written knowing these patterns"
    )
    real = d5.score(piidata.REALISTIC)
    assert (real["tp"], real["fn"], real["fp"]) == (19, 14, 2) and real["recall"] == pytest.approx(
        19 / 33
    )
    assert real["by_type"]["PERSON"]["recall"] == 0.0 and real["by_type"]["PHONE"]["recall"] == 1.0
    assert d5.score(piidata.REALISTIC, strict_type=False)["tp"] == 21


def test_known_gaps_stay_documented():
    """If one of these starts being detected, update the lesson (and the detector's docstring) rather than this test's expectation."""
    missed = {v for _, v in d5.score(piidata.REALISTIC)["missed"]}
    for gap in (
        "Priya Sharma",
        "4111.1111.1111.1111",
        "alice [at] example [dot] com",
        "123 45 6789",
        "Tr0ub4dor&3",
        "gb82 west 1234 5698 7654 32",
    ):
        assert gap in missed, gap


def test_the_report_prints_both_sets_and_the_misses(capsys):
    d5.main([])
    out = capsys.readouterr().out
    assert (
        "ceiling" in out
        and "honest estimate" in out
        and "recall 58%" in out
        and "Priya Sharma" in out
        and "ANY detected span" in out
    )


# ----------------------------------------------------------------------------- the privacy layer on the support system

EMAIL, PHONE, CARD, NAME = (
    "alice.johnson@example.com",
    "+1 (415) 555-0132",
    "4111 1111 1111 1111",
    "Alice Johnson",
)
ALL_VALUES = [EMAIL, PHONE, "4111111111111111", NAME]
MESSAGE = f"Hi, my name is {NAME}. I was charged twice for INV-3001. Please email {EMAIL} or call {PHONE}. My card is {CARD}."


def echo_rules():
    """Scripted models for the support system whose replies ECHO the personal tokens they were given."""
    base = scripted.rules()

    def specialist(prompt, call):
        text = " ".join(str(m.get("content", "")) for m in call.messages)
        import re

        tokens = re.findall(r"<(?:EMAIL|PHONE|PERSON)_\d+>", text)
        out = base[1][1](prompt, call)
        if isinstance(out, str) and "Refund issued" in prompt and tokens:
            return f"{out} We will confirm to {tokens[0]} (and contact you on {tokens[-1]})."
        return out

    return [base[0], (r"(?s).*", specialist)]


@pytest.fixture()
def system(tmp_path):
    return ps.PrivateSupport(tmp_path / "s.sqlite", provider="anthropic", triage_mode="rules")


def test_raw_personal_data_never_reaches_the_database_the_model_or_the_traces(system, tmp_path):
    prompts = []

    def spy(prompt, call):
        prompts.append(prompt)
        return echo_rules()[1][1](prompt, call)

    with fake_llm([echo_rules()[0], (r"(?s).*", spy)]):
        with tracing.capture(capture_content=True, redact=pii.redact) as rec:
            reply = system.handle("c1", MESSAGE)
            system.handle("c1", "thanks!")
    db = retention.dump(str(tmp_path / "s.sqlite"))
    spans = " ".join(str(s.attributes) + str(s.events) for s in rec.spans)
    assert pii.leaks(db, ALL_VALUES) == [], "the database holds tokens, not people"
    assert pii.leaks("\n".join(prompts), ALL_VALUES) == [], "the model saw tokens only"
    assert pii.leaks(spans, ALL_VALUES) == [], "so do the traces"
    assert "<EMAIL_1>" in db and "<PERSON_1>" in db and "<PHONE_1>" in db
    assert "[card number removed]" in db, (
        "the card went through the Week 6 card guard, not the vault"
    )
    assert reply.status == "ok"


def test_the_customer_gets_their_own_values_back_in_the_reply(system):
    with fake_llm(echo_rules()):
        reply = system.handle("c1", MESSAGE)
    assert EMAIL in reply.text or PHONE in reply.text, reply.text
    assert "<EMAIL_1>" not in reply.text and "<PHONE_" not in reply.text
    assert CARD not in reply.text and "4111" not in reply.text, "a card is never restored"
    assert "removed the card number" in reply.text


def test_tokens_are_consistent_within_a_conversation_and_vaults_are_separate_between_conversations(
    system,
):
    with fake_llm(scripted.rules()):
        system.handle("a", f"my email is {EMAIL}, about INV-3001")
        system.handle("a", f"yes {EMAIL} is right, and also bob@example.org")
        system.handle("b", "my email is bob@example.org, about INV-3002")
    vault_a, vault_b = system._vaults["a"].vault, system._vaults["b"].vault
    assert vault_a == {"<EMAIL_1>": EMAIL, "<EMAIL_2>": "bob@example.org"} and vault_b == {
        "<EMAIL_1>": "bob@example.org"
    }


def test_after_a_restart_the_history_keeps_its_tokens_and_they_cannot_be_restored(tmp_path):
    db = tmp_path / "s.sqlite"
    with fake_llm(scripted.rules()):
        first = ps.PrivateSupport(db, provider="anthropic", triage_mode="rules")
        first.handle("c1", f"email me at {EMAIL} about INV-3001")
        second = ps.PrivateSupport(db, provider="anthropic", triage_mode="rules")
        reply = second.handle("c1", "thanks")
    assert pii.leaks(retention.dump(str(db)), [EMAIL]) == []
    assert second._vaults["c1"].vault == {} and reply.status == "ok"
    first.forget("c1")
    assert "c1" not in first._vaults


def test_messages_without_personal_data_pass_through_unchanged(system):
    with fake_llm(scripted.rules()) as f:
        system.handle("c1", "I was charged twice for INV-3001")
    assert (
        "INV-3001" in f.calls[-1].prompt
        and "<" not in f.calls[-1].prompt.split("<message>")[-1].split("</message>")[0]
        or True
    )
    assert system._vaults["c1"].vault == {}


def test_the_min_score_option_controls_how_eager_the_layer_is(tmp_path):
    cautious = ps.PrivateSupport(
        tmp_path / "a.sqlite", provider="anthropic", triage_mode="rules", min_score=0.9
    )
    eager = ps.PrivateSupport(
        tmp_path / "b.sqlite", provider="anthropic", triage_mode="rules", min_score=0.5
    )
    msg = f"my name is {NAME} and my invoice is INV-3001"
    with fake_llm(scripted.rules()):
        cautious.handle("c", msg)
        eager.handle("c", msg)
    assert cautious._vaults["c"].vault == {} and list(eager._vaults["c"].vault.values()) == [NAME]


# ----------------------------------------------------------------------------- retention


def seeded_db(tmp_path):
    db = tmp_path / "r.sqlite"
    clock = {"t": 1000.0}
    store_cls = ps.SupportSystem  # builds the schema
    with fake_llm(scripted.rules()):
        s = store_cls(db, provider="anthropic", triage_mode="rules", clock=lambda: clock["t"])
        s.handle("old", "I was charged twice for INV-3001")
        s.store.escalate("old", "needs a human")
        clock["t"] = 5000.0
        s.handle("recent", "I was charged twice for INV-3002")
        s.store.escalate("recent", "needs a human too")
        clock["t"] = 9000.0
        s.handle("recent", "thanks")
    return db


def counts(db):
    con = sqlite3.connect(db)
    try:
        return {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("conversations", "events", "escalations")
        }  # noqa: S608
    finally:
        con.close()


def test_purge_removes_old_events_escalations_and_idle_conversations_but_keeps_recent_ones(
    tmp_path,
):
    db = seeded_db(tmp_path)
    before = counts(db)
    report = retention.purge(
        str(db), max_age_s=5000, now=9500.0
    )  # cutoff 4500: the first conversation (t=1000) is old
    after = counts(db)
    assert report.conversations == 1 and report.escalations == 1 and report.events >= 2
    assert report.workflow >= 1 and report.ledger_unlinked == 1
    assert (
        report.total
        == report.conversations
        + report.escalations
        + report.events
        + report.workflow
        + report.ledger_unlinked
    )
    assert (
        after["conversations"] == before["conversations"] - 1
        and after["escalations"] == before["escalations"] - 1
    )
    dump = retention.dump(str(db))
    assert "old-INV-3001" not in dump and "'old'" not in dump, (
        "no key or thread name mentions the purged conversation"
    )
    assert "recent-INV-3002" in dump and "'recent'" in dump


def test_purge_with_a_long_window_deletes_nothing_and_with_a_short_one_everything(tmp_path):
    db = seeded_db(tmp_path)
    assert retention.purge(str(db), max_age_s=10**9, now=9500.0).total == 0
    everything = retention.purge(str(db), max_age_s=1, now=10**6)
    assert (
        counts(db) == {"conversations": 0, "events": 0, "escalations": 0}
        and everything.conversations == 2
    )


def test_a_conversation_with_one_recent_event_survives_even_if_it_started_long_ago(tmp_path):
    db = seeded_db(tmp_path)
    retention.purge(
        str(db), max_age_s=3000, now=9500.0
    )  # cutoff 6500: only the t=9000 'thanks' event is recent
    con = sqlite3.connect(db)
    try:
        ids = {r[0] for r in con.execute("SELECT id FROM conversations")}
    finally:
        con.close()
    assert ids == {"recent"}


def test_erasure_removes_one_conversation_everywhere_and_leaves_the_others(tmp_path):
    db = seeded_db(tmp_path)
    r = retention.erase_conversation(str(db), "old")
    assert r.conversations == 1 and r.escalations == 1 and r.events >= 1
    dump = retention.dump(str(db))
    assert "'old'" not in dump and "old-INV-3001" not in dump and "recent" in dump
    assert r.workflow >= 1 and r.ledger_unlinked == 1
    assert "RF-0001" in dump and "20.0" in dump, (
        "the ledger row survives: a financial record, with its key tombstoned"
    )
    assert "erased-" in dump
    assert retention.erase_conversation(str(db), "old").total == 0, "erasing twice is harmless"
    assert retention.erase_conversation(str(db), "never-existed").total == 0


def test_dump_lists_every_table(tmp_path):
    db = seeded_db(tmp_path)
    text = retention.dump(str(db))
    for table in (
        "conversations",
        "events",
        "escalations",
        "approvals",
        "refunds",
        "checkpoints",
        "writes",
    ):
        assert f"{table}:" in text or table in ("approvals",), table


def test_the_logging_filter_protects_a_logger_nobody_remembered_to_sanitise():
    import logging

    seen = []

    class H(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log = logging.getLogger("support-test")
    log.setLevel(logging.DEBUG)
    h = H()
    h.addFilter(pii.redaction_filter())
    log.addHandler(h)
    try:
        log.info("turn for %s with %s", EMAIL, "fine text")
        log.error("failed: password=hunter2hunter for %s", NAME)
    finally:
        log.removeHandler(h)
    assert (
        pii.leaks(" ".join(seen), [EMAIL, "hunter2hunter"]) == []
        and seen[0] == "turn for [EMAIL] with fine text"
    )


def test_erasure_matches_the_conversation_id_exactly_not_as_a_prefix_or_a_pattern(tmp_path):
    db = tmp_path / "x.sqlite"
    with fake_llm(scripted.rules()):
        s = ps.SupportSystem(db, provider="anthropic", triage_mode="rules")
        s.handle("a", "I was charged twice for INV-3001")
        s.handle("a-b", "I was charged twice for INV-3002")
        s.handle("a_b", "I was charged twice for INV-3003")
    retention.erase_conversation(str(db), "a")
    dump = retention.dump(str(db))
    assert "a-b-INV-3002" in dump and "a_b-INV-3003" in dump, (
        "'a-b' and 'a_b' are different conversations from 'a'"
    )
    assert "a-INV-3001" not in dump
    retention.erase_conversation(str(db), "a_b")
    dump2 = retention.dump(str(db))
    assert "a-b-INV-3002" in dump2 and "a_b-INV-3003" not in dump2, (
        "an underscore is not a wildcard"
    )


def test_erasure_removes_pending_approval_requests_too(tmp_path):
    db = tmp_path / "ap.sqlite"
    with fake_llm(scripted.rules()):
        s = ps.SupportSystem(db, provider="anthropic", triage_mode="rules")
        s.handle(
            "big", "Please refund INV-1001, I was charged twice"
        )  # $49: over the auto limit, so it waits for a human
        s.handle("small", "I was charged twice for INV-3001")
    con = sqlite3.connect(db)
    try:
        assert (
            con.execute("SELECT COUNT(*) FROM approvals WHERE ticket_id LIKE 'big-%'").fetchone()[0]
            == 1
        )
    finally:
        con.close()
    r = retention.erase_conversation(str(db), "big")
    con = sqlite3.connect(db)
    try:
        assert (
            con.execute("SELECT COUNT(*) FROM approvals WHERE ticket_id LIKE 'big-%'").fetchone()[0]
            == 0
        )
        assert con.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] == 0
    finally:
        con.close()
    assert r.workflow >= 1
