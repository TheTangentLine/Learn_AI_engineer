"""Tests for the Week 11 weekly challenge: the question set, the evaluation arithmetic, the load-test helpers, the report, the compose wiring, and the load generator's handling of in-band stream errors."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE.parents[1]
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SOLUTIONS))

import evalrag as E  # noqa: E402
import loadgen as G  # noqa: E402
import loadtest as LT  # noqa: E402
import questions as Q  # noqa: E402
import report as R  # noqa: E402
import stackctl as S  # noqa: E402

# ----------------------------------------------------------------------------- the question set


def test_questions_have_unique_ids_alternating_splits_and_both_kinds():
    rows = Q.questions()
    assert len({r["id"] for r in rows}) == len(rows) == len(Q.IN_SCOPE) + len(Q.OUT_OF_SCOPE)
    assert [r["split"] for r in rows[:4]] == ["dev", "test", "dev", "test"]
    assert all(r["expected"] for r in rows if r["in_scope"])
    assert all(not r["expected"] for r in rows if not r["in_scope"])
    dev, test = ([r for r in rows if r["split"] == s and r["in_scope"]] for s in ("dev", "test"))
    assert abs(len(dev) - len(test)) <= 1


def test_every_expected_lesson_exists_in_the_repository():
    lessons = {p.name for p in (ROOT / "weeks").glob("week*/day*.md")}
    missing = {f for r in Q.questions() for f in r["expected"]} - lessons
    assert not missing


def test_no_question_is_a_copy_of_a_lesson_heading():
    headings = set()
    for p in (ROOT / "weeks").glob("week*/day*.md"):
        for line in p.read_text().splitlines():
            if line.startswith("#"):
                headings.add(line.lstrip("# ").strip().rstrip("?").lower())
    assert not [
        r["question"] for r in Q.questions() if r["question"].rstrip("?").lower() in headings
    ]


# ----------------------------------------------------------------------------- evaluation arithmetic


def test_wilson_matches_known_values_and_handles_empty_and_extreme_counts():
    p, lo, hi = E.wilson(7, 10)
    assert (
        p == 0.7 and lo == pytest.approx(0.397, abs=0.002) and hi == pytest.approx(0.892, abs=0.002)
    )
    assert E.wilson(0, 10)[1] == 0.0 and E.wilson(10, 10)[2] == 1.0
    assert all(math.isnan(x) for x in E.wilson(0, 0))
    assert E.fmt(E.wilson(0, 0)) == "n/a" and E.fmt(E.wilson(5, 10)).startswith("50%")


def test_retrieval_hit_and_rank_use_the_lesson_file_name_not_the_directory():
    src = [{"doc": "week09_x/day2_a.md"}, {"doc": "week10_y/day3_b.md"}]
    assert E.retrieval_hit(src, ["day3_b.md"]) and E.retrieval_rank(src, ["day3_b.md"]) == 2
    assert E.retrieval_rank(src, ["day2_a.md", "day3_b.md"]) == 1
    assert not E.retrieval_hit(src, ["day9_z.md"]) and E.retrieval_rank(src, ["day9_z.md"]) is None
    assert not E.retrieval_hit([], ["day2_a.md"])


ROW_IN = {"id": "i", "split": "dev", "in_scope": True, "expected": ["a.md"]}
ROW_OUT = {"id": "o", "split": "test", "in_scope": False, "expected": []}


def test_judging_separates_valid_cited_invalid_and_abstained_answers():
    src = [{"doc": "x/a.md"}, {"doc": "x/b.md"}]
    ok = E.judge(ROW_IN, src, "It is so [1].")
    assert ok["hit"] and ok["rank"] == 1 and ok["cited_valid"] and not ok["abstained"]
    bad = E.judge(ROW_IN, src, "It is so [1] and also [7].")
    assert bad["invalid_citation"] and not bad["cited_valid"]  # one fabricated number spoils it
    none = E.judge(ROW_IN, src, "It is so.")
    assert not none["cited_valid"] and not none["abstained"]
    abst = E.judge(ROW_OUT, [], "I don't know based on the provided sources.")
    assert abst["abstained"] and abst["hit"] is None and abst["n_sources"] == 0


def test_summaries_count_each_behaviour_per_split_with_errors_separate():
    def o(row, sources, answer, error=None):
        j = E.judge(row, sources, answer)
        j["error"] = error
        return j

    src = [{"doc": "a.md"}]
    outcomes = [
        o(ROW_IN, src, "yes [1]"),
        o(
            {**ROW_IN, "id": "i2", "split": "test"},
            src,
            "I don't know based on the provided sources.",
        ),
        o({**ROW_IN, "id": "i3"}, src, "", error="backend_error"),
        o(ROW_OUT, [], "I don't know based on the provided sources."),
        o({**ROW_OUT, "id": "o2"}, src * 4, "Paris."),
        o({**ROW_OUT, "id": "o3"}, src * 4, "Pancakes are good."),
    ]
    s = E.summarize(outcomes)
    assert (s["n_in"], s["n_out"], s["errors"]) == (3, 3, 1)
    assert s["hit"][0] == 1.0 and s["cited_in"][0] == pytest.approx(1 / 3)
    assert s["abstained_in"][0] == pytest.approx(1 / 3)
    assert s["abstained_out"][0] == pytest.approx(1 / 3) and s["no_sources_out"][
        0
    ] == pytest.approx(1 / 3)
    assert s["answered_out"][0] == pytest.approx(2 / 3)
    dev = E.summarize(outcomes, "dev")
    assert dev["n_in"] == 2 and dev["n_out"] == 0 and dev["errors"] == 1
    assert math.isnan(dev["abstained_out"][0])  # nothing out of scope in dev: no rate, not 0%


def test_best_floor_keeps_the_required_share_of_in_scope_questions():
    scores = [(30, True), (25, True), (20, True), (15, True), (11, True), (9, False), (14, False)]
    floor = E.best_floor(scores, max_in_scope_loss=0.0)
    assert floor < 11 and sum(s > floor for s, ok in scores if ok) == 5
    f20 = E.best_floor(scores, max_in_scope_loss=0.2)  # may lose one of five
    assert 11 <= f20 < 15 and sum(s > f20 for s, ok in scores if ok) == 4
    assert E.best_floor([(5, False)]) == 0.0 and E.best_floor([]) == 0.0


def test_retrieval_only_reports_hits_ranks_and_the_top_score():
    chunk = SimpleNamespace(doc="w/a.md")

    def search(q, k, floor):
        return [(chunk, 9.0)] if "alpha" in q and 9.0 > floor else []

    rows = [
        {"id": "i", "split": "dev", "in_scope": True, "expected": ["a.md"], "question": "alpha?"},
        {"id": "o", "split": "test", "in_scope": False, "expected": [], "question": "beta?"},
    ]
    r = E.retrieval_only(search, rows)
    assert r[0]["hit"] and r[0]["rank"] == 1 and r[0]["top_score"] == 9.0 and r[1]["n_sources"] == 0
    assert E.retrieval_only(search, rows, min_score=10)[0]["n_sources"] == 0


# ----------------------------------------------------------------------------- load-test helpers


def test_request_bodies_cycle_through_the_questions():
    rows = [{"question": "a"}, {"question": "b"}]
    make = LT.ask_bodies(rows, k=3, max_tokens=50)
    assert [make(i)["question"] for i in range(5)] == ["a", "b", "a", "b", "a"]
    assert make(0) == {"question": "a", "k": 3, "max_tokens": 50, "temperature": 0}


def test_the_knee_is_the_first_rate_that_blows_up_p95_or_fails():
    ok = {"errors": {}, "lat95": 1.0}
    assert LT.find_knee([{"rate": 1, **ok}, {"rate": 2, **ok}]) is None
    assert LT.find_knee([{"rate": 1, **ok}, {"rate": 2, "errors": {}, "lat95": 3.5}]) == 2
    assert (
        LT.find_knee([{"rate": 1, **ok}, {"rate": 2, "errors": {"HTTP 503": 1}, "lat95": 1.0}]) == 2
    )
    assert (
        LT.find_knee([{"rate": 1, **ok}, {"rate": 2, "errors": {}, "lat95": 3.0}]) is None
    )  # exactly 3x is not above 3x
    assert LT.find_knee([]) is None


def test_row_of_flattens_a_summary():
    s = G.Summary(
        n=4,
        ok=3,
        errors={"HTTP 503": 1},
        wall=2.0,
        tokens=60,
        throughput=30.0,
        requests_per_second=1.5,
        latency={"p50": 0.5, "p95": 0.9},
        ttft={"p50": 0.1, "p95": 0.2},
        per_user_tps=40.0,
    )
    r = LT.row_of(s)
    assert (r["n"], r["ok"], r["lat95"], r["ttft50"], r["rps"]) == (4, 3, 0.9, 0.1, 1.5)


# ----------------------------------------------------------------------------- the load generator and in-band errors


def sse(*events: str) -> bytes:
    return ("".join(f"{e}\n\n" for e in events)).encode()


def transport_for(body: bytes, status=200):
    return httpx.MockTransport(lambda req: httpx.Response(status, content=body))


def run_one(body: bytes, path="/v1/ask", status=200):
    async def go():
        async with httpx.AsyncClient(transport=transport_for(body, status)) as c:
            return await G.one_request(c, "http://t", {"question": "q"}, 0.0, path=path)

    return asyncio.run(go())


def chunk(text):
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})


def test_the_load_generator_skips_the_sources_event_and_counts_tokens_on_ask():
    r = run_one(sse('event: sources\ndata: [{"n": 1}]', chunk("a"), chunk("b"), "data: [DONE]"))
    assert r.ok and r.tokens == 2 and r.text == "ab" and r.error == ""


def test_an_in_band_error_after_a_200_status_is_a_failure_with_its_code():
    err = "data: " + json.dumps({"error": {"code": "backend_error", "message": "x"}})
    r = run_one(sse(chunk("a"), err, "data: [DONE]"))
    assert (
        not r.ok
        and r.error.startswith("stream_error")
        and "backend_error" in r.error
        and r.status == 200
    )
    only = run_one(sse(err, "data: [DONE]"))
    assert not only.ok and only.tokens == 0 and "backend_error" in only.error
    summary = G.summarize([r, only], 1.0)
    assert summary.errors == {"stream_error": 2}


def test_a_stream_with_no_tokens_and_no_error_is_still_a_failure():
    r = run_one(sse("data: [DONE]"))
    assert not r.ok and r.error == "no tokens"


def test_the_path_argument_selects_the_endpoint():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, content=sse(chunk("a"), "data: [DONE]"))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await G.one_request(c, "http://t", {}, 0.0, path="/v1/ask")
            await G.one_request(c, "http://t", {}, 0.0)

    asyncio.run(go())
    assert seen == ["/v1/ask", "/v1/chat/completions"]


# ----------------------------------------------------------------------------- compose wiring and the report


def test_compose_environment_passes_models_context_keys_and_the_floor_as_strings():
    env = S.compose_env(Path("/m"), Path("/c"), min_score=13.4, port=8123)
    assert env["MODELS_DIR"] == "/m" and env["API_CONTEXT"] == "/c"
    assert env["LLMAPI_MIN_SCORE"] == "13.4" and env["API_PORT"] == "8123"
    assert env["LLMAPI_ABSTAIN"] == "0"
    assert S.compose_env(Path("/m"), Path("/c"), abstain=True)["LLMAPI_ABSTAIN"] == "1"
    users = json.loads(env["LLMAPI_KEYS"])
    assert (
        users[0]["key"] == S.KEY and users[0]["rpm"] >= 600
    )  # the load test must not be rate limited
    assert env["COMPOSE_PROJECT_NAME"] == S.PROJECT


def fake_results():
    def outcome(i, in_scope, n_sources, answer, error=None):
        row = {
            "id": f"{'in' if in_scope else 'out'}{i}",
            "split": "dev" if i % 2 == 0 else "test",
            "in_scope": in_scope,
            "expected": ["a.md"] if in_scope else [],
        }
        o = E.judge(row, [{"doc": "w/a.md"}] * n_sources, answer)
        o["error"] = error
        return o

    outcomes = [
        outcome(0, True, 4, "yes [1]"),
        outcome(1, True, 4, "I don't know based on the provided sources."),
        outcome(2, False, 0, "I don't know based on the provided sources."),
        outcome(3, False, 4, "Paris.", error=None),
    ]
    load = {
        "errors": {},
        "n": 8,
        "ok": 8,
        "wall": 5.0,
        "throughput": 50.0,
        "rps": 1.6,
        "lat50": 0.7,
        "lat95": 1.1,
        "ttft50": 0.05,
        "ttft95": 0.09,
        "per_user": 40.0,
    }
    return {
        "floor": 13.4,
        "retrieval": {
            "chunks": 100,
            "docs": 10,
            "n_in": 2,
            "hit_all": (1.0, 0.5, 1.0),
            "rank1": 2,
            "in_min": 11.0,
            "in_max": 30.0,
            "in_median": 20.0,
            "out_min": 5.0,
            "out_max": 20.0,
            "out_median": 8.0,
            "test_in": 1,
            "test_in_kept": 1,
            "test_out": 1,
            "test_out_dropped": 1,
        },
        "quality": {
            "baseline": {"outcomes": outcomes},
            "floor": {"outcomes": outcomes},
            "abstain": {"outcomes": outcomes},
        },
        "n_in": 2,
        "n_out": 2,
        "image_mb": 229.0,
        "up_seconds": 6.0,
        "ready_seconds": 1.2,
        "extraction": {"exact": (0.71, 0.55, 0.83), "field_accuracy": 0.89, "valid_order": 0.92},
        "closed": [{"users": 1, **load}, {"users": 2, **{**load, "rps": 2.0}}],
        "open": [{"rate": 0.5, **load}, {"rate": 1.0, **{**load, "errors": {"HTTP 503": 2}}}],
        "knee": 1.0,
        "per_level": 8,
        "open_duration": 30.0,
        "container_stats": [{"name": "api", "memory": "55MiB / 3GiB", "cpu": "0.3%"}],
    }


def test_the_report_renders_every_section_from_the_results_without_nan_or_placeholders():
    text = R.render(fake_results())
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## Not run", "## Limits"):
        assert heading in text
    assert "relevance floor 13.4" in text and "71% [55%, 83%]" in text
    assert (
        "floor 13.4 + abstain without calling the model" in text and "floor + abstain **0**" in text
    )
    assert "about **1.0 requests per second**" in text and "$0.2/hour" in text
    assert "nan" not in text.lower().replace("finance", "") and "{" not in text.replace("{{", "")
    assert "$0.028 per 1,000" in text  # 0.2 / 3600 / 2.0 req/s * 1000
    assert (
        "| 1 | 8 | 0 | 1.60 | 50 | 50 ms | 90 ms | 0.70 s | 1.10 s |" in text
    )  # milliseconds and seconds are formatted, not mixed up


def test_the_report_says_when_no_knee_was_found_and_when_extraction_differs():
    res = fake_results()
    res["knee"] = None
    res["extraction"]["exact"] = (0.40, 0.3, 0.5)
    text = R.render(res)
    assert "no capacity limit was found" in text and "the deployment differs" in text


def test_the_report_names_what_was_not_run():
    text = R.render(fake_results())
    for item in ("Fly.io", "GPU", "vLLM", "Ollama", "hosted frontier model"):
        assert item in text


class SlowStream(httpx.AsyncByteStream):
    """A body that arrives in pieces with pauses, so time to first token and total latency can differ."""

    def __init__(self, pieces, gap):
        self.pieces, self.gap = pieces, gap

    async def __aiter__(self):
        for p in self.pieces:
            yield p
            await asyncio.sleep(self.gap)


def test_time_to_first_token_is_the_first_token_and_latency_the_last_byte():
    pieces = [(chunk(t) + "\n\n").encode() for t in "abc"] + [b"data: [DONE]\n\n"]
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, stream=SlowStream(pieces, 0.05))
    )

    async def go():
        async with httpx.AsyncClient(transport=transport) as c:
            return await G.one_request(c, "http://t", {}, 0.0)

    r = asyncio.run(go())
    assert r.ok and r.tokens == 3
    assert (
        r.ttft < 0.04 and r.latency >= 0.14
    )  # first token at once; the last piece 0.1 s later plus the final pause
    assert r.per_user_tps == pytest.approx((3 - 1) / (r.latency - r.ttft))


def test_per_user_speed_counts_the_tokens_after_the_first_over_the_time_after_the_first():
    assert G.Result(ok=True, latency=2.0, ttft=1.0, tokens=11).per_user_tps == pytest.approx(10.0)
    assert G.Result(ok=True, latency=2.0, ttft=1.0, tokens=1).per_user_tps == 0.0
    assert G.Result(ok=True, latency=1.0, ttft=1.0, tokens=5).per_user_tps == 0.0
