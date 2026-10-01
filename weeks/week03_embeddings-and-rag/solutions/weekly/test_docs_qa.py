"""Tests for docs_qa: golden-set integrity, evaluator maths, offline stand-ins, CLI wiring."""

from __future__ import annotations

import pytest
from docs_qa import app, golden
from docs_qa.__main__ import main
from docs_qa.evaluate import check_gold, evaluate, has_answer
from docs_qa.offline import make_rules

from common import embed
from common.embed import HashEmbedder
from common.fake import fake_llm
from common.rag import RagBot, RagIndex, SourceDoc

MINI = [
    SourceDoc(
        "w1/d1",
        "# Retries\n\nUse exponential backoff with jitter when the API returns HTTP 429. "
        "Honour the Retry-After header every time.",
        {"week": 1},
    ),
    SourceDoc(
        "w1/d2",
        "# KV cache\n\nThe KV cache stores keys and values for earlier tokens in GPU memory "
        "so generation can skip recomputation.",
        {"week": 1},
    ),
]
MINI_GOLD = [
    (
        "how should clients space out retries after HTTP 429",
        "w1/d1",
        "exponential backoff with jitter",
    ),
    ("what does the kv cache hold in gpu memory", "w1/d2", "keys and values"),
]


@pytest.fixture()
def bot():
    emb = HashEmbedder()
    idx = RagIndex(emb)
    idx.sync(MINI)
    b = RagBot(idx, k=2)
    b.calibrate([q for q, _, _ in MINI_GOLD], ["zebra giraffe savannah", "volcano lava eruption"])
    return b, emb


def test_golden_phrases_exist_in_the_real_course_corpus(tmp_path):
    """If a lesson edit removes an answer phrase, the golden set silently stops measuring anything."""
    idx, report = app.build_index(HashEmbedder(), tmp_path / "i")
    assert report.added and idx.chunks
    bot = RagBot(idx)
    assert check_gold(bot, golden.ANSWERABLE) == []
    assert (
        len(golden.ANSWERABLE) >= 15
        and len(golden.UNANSWERABLE) >= 6
        and len(golden.FOLLOWUPS) >= 3
    )
    followup_gold = [(h, lesson, phrase) for h, _, lesson, phrase in golden.FOLLOWUPS]
    assert check_gold(bot, followup_gold) == []


def test_check_gold_flags_a_missing_phrase(bot):
    b, _ = bot
    assert check_gold(b, MINI_GOLD) == []
    assert check_gold(b, [("q", "w1/d1", "a phrase that is not there")]) == [
        "w1/d1: 'a phrase that is not there'"
    ]
    assert (
        check_gold(b, [("q", "w9/d9", "exponential backoff")]) != []
    )  # right phrase, wrong lesson


def test_has_answer_requires_lesson_and_phrase(bot):
    b, _ = bot
    hits = b.index.search("exponential backoff jitter", 2)
    assert has_answer(hits, "w1/d1", "exponential backoff with jitter")
    assert not has_answer(hits, "w1/d2", "exponential backoff with jitter")


def test_evaluate_computes_every_metric_on_a_known_corpus(bot):
    b, emb = bot
    with fake_llm(make_rules(emb)):
        text, m = evaluate(
            b,
            MINI_GOLD,
            ["zebra giraffe savannah", "volcano lava eruption"],
            [("How do I retry?", "and how long to wait?", "w1/d1", "retry-after header")],
        )
    assert m["n"] == 2 and m["retrieved"] == 2 and m["chunks"] == len(b.index.chunks)
    assert m["abstained_unanswerable"] == 2 and m["bad_gold"] == [] and m["clean"] == 2
    assert (
        0.0 <= m["groundedness"] <= 1.0
        and m["answered"] <= 2
    )
    assert (
        "retrieval: gold chunk in top-2: **2/2**" in text
        and "unanswerable questions refused: **2/2**" in text
    )
    assert m["n_followups"] == 1 and set(m["followups"]) == {"raw follow-up", "rewritten"}


def test_evaluate_reports_unanswerable_leaks_and_bad_gold(bot):
    b, emb = bot
    b.tau = 0.0  # gate wide open: the reader must refuse on its own, otherwise we flag a leak
    with fake_llm(
        [(r"<question>", '{"answerable": true, "answer": "Sure [1]", "citations": [1]}')]
    ):
        text, m = evaluate(
            b, [*MINI_GOLD, ("q", "w1/d1", "phrase not present")], ["zebra giraffe savannah"]
        )
    assert m["abstained_unanswerable"] == 0 and "ANSWERED (hallucination risk)" in text
    assert m["bad_gold"] and "WARNING: gold phrases not found" in text


def test_offline_rewriter_prepends_the_previous_user_turn(bot):
    b, emb = bot
    with fake_llm(make_rules(emb)) as fake:
        a = b.ask("and how long to wait?", [("How do I retry after 429?", "Use backoff.")])
    assert a.standalone == "How do I retry after 429? and how long to wait?" and fake.calls


def test_cli_index_and_eval_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setattr(embed, "_SINGLETON", {})
    monkeypatch.setattr(app, "INDEX_DIR", tmp_path / "index")
    import docs_qa.__main__ as cli

    monkeypatch.setattr(cli, "build_index", lambda emb: app.build_index(emb, tmp_path / "index"))
    assert main(["index"]) == 0
    assert "added" in capsys.readouterr().out
    assert (
        main(["index"]) == 0 and "unchanged" in capsys.readouterr().out
    )  # second run embeds nothing
    out = tmp_path / "report.md"
    assert main(["eval", "--offline", "--out", str(out)]) == 0
    report = out.read_text()
    assert report.startswith("# docs_qa evaluation") and "## Per-question" in report


def test_course_sources_have_stable_ids_and_week_metadata():
    docs = app.course_sources()
    assert len({d.id for d in docs}) == len(docs)
    d = next(x for x in docs if x.id == "week01/day2")
    assert d.meta["week"] == 1 and d.meta["path"].startswith("weeks/week01")
