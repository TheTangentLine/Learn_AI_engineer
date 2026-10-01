"""Tests for rag_upgrade: split discipline, selection under a cost cap, verdict rules, report, end to end."""

from __future__ import annotations

import numpy as np
import pytest
from rag_upgrade import configs as cfgmod
from rag_upgrade.__main__ import main
from rag_upgrade.configs import CragRetrieve
from rag_upgrade.harness import (
    Config,
    Evaluated,
    evaluate_config,
    render_report,
    select_best,
    split_golden,
    verdict,
)

from common import evalkit
from common.embed import HashEmbedder
from common.evalkit import GoldQuery, RetrievalResult, save_golden
from common.rag import RagIndex, SourceDoc
from common.rerank import OverlapReranker


def golden(n_manual=10, n_syn=14):
    return [GoldQuery(f"m{i}", f"q{i}", "d", "p", kind="manual") for i in range(n_manual)] + [
        GoldQuery(f"s{i}", f"sq{i}", "d", "p", kind="synthetic") for i in range(n_syn)
    ]


def test_split_is_deterministic_disjoint_complete_and_stratified():
    g = golden()
    dev, test = split_golden(g, seed=1)
    assert split_golden(g, seed=1) == (dev, test)
    assert split_golden(g, seed=2) != (dev, test), "the seed controls the shuffle"
    assert {q.id for q in dev}.isdisjoint({q.id for q in test}) and len(dev) + len(test) == len(g)
    for kind, n in (("manual", 10), ("synthetic", 14)):
        assert sum(q.kind == kind for q in dev) == round(n / 2) and sum(
            q.kind == kind for q in test
        ) == n - round(n / 2)


def fake_eval(name, mrr, ctx, pairs=0.0, k=4):
    res = RetrievalResult(name, mrr=list(mrr), hit5=[1.0] * len(mrr), hit1=[1.0] * len(mrr))
    return Evaluated(Config(name, lambda q: [], k, rerank_pairs=pairs), res, ctx, [1.0] * len(mrr))


def test_select_best_uses_dev_mrr_within_the_cost_cap_and_prefers_cheaper_on_ties():
    dev = {
        "base": fake_eval("base", [0.5] * 4, 1000),
        "great-but-huge": fake_eval("great-but-huge", [0.9] * 4, 5000),
        "good": fake_eval("good", [0.7] * 4, 1500, pairs=20),
        "good-cheaper": fake_eval("good-cheaper", [0.7] * 4, 1200),
    }
    assert select_best(dev, "base", max_context_ratio=2.5) == "good-cheaper"
    assert select_best(dev, "base", max_context_ratio=6.0) == "great-but-huge"
    assert select_best({"base": dev["base"]}, "base", 2.5) == "base"


@pytest.mark.parametrize(
    "paired,ratio,expected",
    [
        ({"ci_low": 0.02, "ci_high": 0.2}, 1.0, "ADOPT"),
        ({"ci_low": 0.02, "ci_high": 0.2}, 3.0, "GAIN BUT TOO COSTLY"),
        ({"ci_low": -0.05, "ci_high": 0.1}, 1.0, "REJECT: no significant improvement"),
        ({"ci_low": -0.2, "ci_high": -0.01}, 1.0, "REJECT: significantly worse"),
        ({"ci_low": 0.0, "ci_high": 0.1}, 1.0, "REJECT: no significant improvement"),
    ],
)
def test_verdict_requires_a_significant_gain_and_acceptable_cost(paired, ratio, expected):
    assert verdict(paired, ratio, 2.5) == expected


def test_evaluate_config_calls_retrieve_once_per_query_and_uses_measured_costs():
    g = [GoldQuery("a", "qa", "d1", "alpha"), GoldQuery("b", "qb", "d2", "beta")]
    calls = []

    class Counting:
        def __call__(self, q):
            calls.append(q)
            return {
                "qa": [("d1", "alpha text"), ("d9", "x")],
                "qb": [("d9", "n"), ("d2", "beta text")],
            }[q]

        def per_query_cost(self):
            return 21.5, 0.25

    cfg = Config("c", Counting(), 2)
    e = evaluate_config(cfg, g)
    assert sorted(calls) == ["qa", "qb"], "each query is retrieved exactly once"
    assert e.result.mrr == [1.0, 0.5] and e.hit_at_4 == [1.0, 1.0]
    assert (cfg.rerank_pairs, cfg.llm_calls) == (21.5, 0.25)
    assert e.context_chars == np.mean([len("alpha text") + 1, 1 + len("beta text")])


def test_report_has_every_section_and_a_verdict():
    dev = {
        "base": fake_eval("base", [0.5, 0.6, 0.4, 0.7], 1000),
        "better": fake_eval("better", [0.9, 0.9, 0.8, 0.9], 1100),
    }
    test = {
        "base": fake_eval("base", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1000),
        "better": fake_eval("better", [0.9, 0.8, 0.9, 0.8, 0.9, 0.9], 1100),
    }
    text, summary = render_report(dev, test, "base", "better", 4, 6, 2.5)
    for section in (
        "# RAG upgrade report",
        "## Step 1",
        "## Step 2",
        "## Verdict",
        "## Caveats",
        "paired diff",
    ):
        assert section in text
    assert summary["verdict"] == "ADOPT" and "Chosen on dev: `better`" in text
    same, s2 = render_report(dev, test, "base", "base", 4, 6, 2.5)
    assert "NO CHANGE" in same and s2["verdict"].startswith("NO CHANGE")


def test_dev_winner_that_does_not_replicate_on_test_is_rejected():
    """The regression-to-the-mean story: great on dev, nothing on test -> no ADOPT."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.9, 25)
    dev = {
        "base": fake_eval("base", base, 1000),
        "lucky": fake_eval("lucky", np.clip(base + 0.15, 0, 1), 1000),
    }
    test_base = rng.uniform(0.3, 0.9, 25)
    test = {
        "base": fake_eval("base", test_base, 1000),
        "lucky": fake_eval("lucky", test_base + rng.normal(0, 0.1, 25), 1000),
    }
    text, summary = render_report(dev, test, "base", "lucky", 25, 25, 2.5)
    assert summary["verdict"] == "REJECT: no significant improvement"


def test_crag_retrieve_counts_its_own_work():
    idx = RagIndex(HashEmbedder())
    idx.sync([SourceDoc(f"d{i}", f"# T{i}\n\n" + f"topic{i} words " * 40, {}) for i in range(3)])
    crag = CragRetrieve(idx, OverlapReranker(), None, 4)
    for q in ("topic0 words", "topic1 words", "zzz qqq"):
        out = crag(q)
        assert out and all(isinstance(d, str) and isinstance(t, str) for d, t in out)
    pairs, llm = crag.per_query_cost()
    assert pairs > 0 and llm == 0.0 and crag.queries == 3


def test_build_configs_smoke_and_names():
    idx = RagIndex(HashEmbedder())
    idx.sync(
        [
            SourceDoc(f"w1/d{i}", f"# Lesson {i}\n\n" + f"alpha{i} beta gamma. " * 60, {"week": 1})
            for i in range(4)
        ]
    )
    configs = cfgmod.build_configs(idx, idx.emb, OverlapReranker(), chat=None)
    names = [c.name for c in configs]
    assert names[0].startswith("baseline") and len(names) == len(set(names)) >= 7
    for c in configs:
        out = c.retrieve("alpha1 beta")
        assert out and out[0][0].startswith("w1/")


def test_end_to_end_cli_with_stubbed_models(tmp_path, monkeypatch):
    import rag_upgrade.__main__ as cli

    from common import embed

    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setattr(embed, "_SINGLETON", {})
    monkeypatch.setattr(cli, "get_reranker", lambda: OverlapReranker())
    monkeypatch.setattr(cli, "RagIndex", lambda emb, path: RagIndex(emb, tmp_path / "idx"))
    gold = [
        GoldQuery(
            f"g{i}",
            f"unbounded gather stampede concurrency semaphore limit {i}",
            "week01/day5",
            "unbounded gather = stampede",
            kind="manual" if i % 2 else "synthetic",
        )
        for i in range(8)
    ]
    path = tmp_path / "g.jsonl"
    save_golden(path, gold)
    out = tmp_path / "report.md"
    assert main(["--golden", str(path), "--out", str(out)]) == 0
    report = out.read_text()
    assert "## Verdict" in report and "baseline (hybrid, k=4)" in report
    assert "4 dev + 4 test" in report, "stratified 50/50 split of 8 questions"
    rows = [ln for ln in report.splitlines() if ln.startswith("| baseline (hybrid, k=4) |")]
    test_row = rows[-1]  # the last table in the report is "All candidates on TEST"
    assert "0.00 [0.00-0.00]" not in test_row, (
        "the pipeline must actually retrieve the answer chunk"
    )
    assert evalkit.load_golden(path) == gold


def test_equal_mrr_and_context_prefers_fewer_rerank_pairs():
    dev = {
        "base": fake_eval("base", [0.5] * 4, 1000),
        "reranked": fake_eval("reranked", [0.8] * 4, 1000, pairs=20),
        "plain": fake_eval("plain", [0.8] * 4, 1000, pairs=0),
    }
    assert select_best(dev, "base", 2.5) == "plain"
