"""Tests for Week 11 Day 2: the load generator (percentiles, streaming, closed and open loops against an ASGI app), the block pool, paged attention, the batching simulator."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import httpx
import pytest
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import loadgen as G  # noqa: E402
import paged_kv as P  # noqa: E402
import scheduler as S  # noqa: E402

# ----------------------------------------------------------------------------- percentiles and the load generator


def test_percentiles_by_hand():
    assert G.percentile([1, 2, 3, 4, 5], 50) == 3 and G.percentile([1, 2, 3, 4], 50) == 2.5
    assert (
        G.percentile([10], 99) == 10
        and G.percentile([1, 2, 3, 4, 5], 100) == 5
        and G.percentile([1, 2, 3, 4, 5], 0) == 1
    )
    assert G.percentile(list(range(101)), 95) == pytest.approx(95) and math.isnan(
        G.percentile([], 50)
    )
    assert G.percentile([5, 1, 3], 50) == 3, "input need not be sorted"


async def fake_sse_app(scope, receive, send):
    """An ASGI app speaking the OpenAI streaming format: 5 tokens, then [DONE]; a path /fail answers 500 and /empty sends no tokens."""
    if scope["type"] != "http":
        return
    await receive()
    path = scope["path"]
    if path == "/fail/v1/chat/completions":
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"boom"})
        return
    events = (
        []
        if path == "/empty/v1/chat/completions"
        else [
            f"data: {json.dumps({'choices': [{'delta': {'content': f't{i} '}}]})}\n\n"
            for i in range(5)
        ]
    )
    events += [
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    await send({"type": "http.response.body", "body": "".join(events).encode()})


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_one_request_counts_streamed_tokens_and_records_the_timings():
    async def run():
        async with client_for(fake_sse_app) as c:
            return await G.one_request(c, "http://test", {"messages": []}, 0.0)

    r = asyncio.run(run())
    assert (
        r.ok
        and r.tokens == 5
        and r.text == "t0 t1 t2 t3 t4 "
        and 0 <= r.ttft <= r.latency
        and r.status == 200
    )


def test_errors_and_empty_streams_are_failures_not_crashes():
    async def run(prefix):
        async with client_for(fake_sse_app) as c:
            return await G.one_request(c, "http://test" + prefix, {}, 0.0)

    bad, empty = asyncio.run(run("/fail")), asyncio.run(run("/empty"))
    assert not bad.ok and bad.status == 500 and "500" in bad.error
    assert not empty.ok and empty.error == "no tokens"


def test_summarize_computes_throughput_percentiles_and_error_counts():
    rs = [G.Result(True, latency=1.0 + i, ttft=0.1 * (i + 1), tokens=11) for i in range(4)] + [
        G.Result(False, error="HTTP 429")
    ]
    s = G.summarize(rs, wall=10.0)
    assert (
        (s.n, s.ok, s.tokens) == (5, 4, 44)
        and s.throughput == pytest.approx(4.4)
        and s.errors == {"HTTP 429": 1}
        and s.requests_per_second == pytest.approx(0.4)
    )
    assert (
        s.latency["p50"] == pytest.approx(2.5)
        and s.ttft["p95"] == pytest.approx(0.385)
        and s.per_user_tps > 0
    )
    assert (
        G.summarize([], 1.0).throughput == 0
        and G.Result(True, latency=1, ttft=1, tokens=1).per_user_tps == 0
    )


def test_the_closed_loop_runs_exactly_n_requests_with_bounded_concurrency(monkeypatch):
    live, peak = [0], [0]

    async def fake(client, url, body, t0, **kw):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0.01)
        live[0] -= 1
        return G.Result(True, latency=0.01, ttft=0.001, tokens=3, started=0.0, text=str(body["i"]))

    monkeypatch.setattr(G, "one_request", fake)
    results, wall = asyncio.run(
        G.closed_loop("http://x", lambda i: {"i": i}, concurrency=3, n_requests=10)
    )
    assert (
        len(results) == 10
        and sorted(int(r.text) for r in results) == list(range(10))
        and peak[0] == 3
        and wall >= 0.03
    )


def test_the_open_loop_sends_a_poisson_stream_independent_of_completions(monkeypatch):
    starts = []

    async def fake(client, url, body, t0, **kw):
        starts.append(__import__("time").perf_counter() - t0)
        await asyncio.sleep(0.2)  # slower than the arrival gap: requests overlap
        return G.Result(True, latency=0.2, ttft=0.01, tokens=2)

    monkeypatch.setattr(G, "one_request", fake)
    results, _ = asyncio.run(G.open_loop("http://x", lambda i: {}, rate=50, duration=0.5, seed=1))
    assert 10 < len(results) < 45 and starts == sorted(starts)
    assert max(b - a for a, b in zip(starts, starts[1:], strict=False)) < 0.2, (
        "arrivals do not wait for earlier requests to finish"
    )
    results2, _ = asyncio.run(G.open_loop("http://x", lambda i: {}, rate=50, duration=0.5, seed=1))
    assert len(results2) == len(results), "the arrival process is a function of the seed"


# ----------------------------------------------------------------------------- the block pool


def test_blocks_are_allocated_on_demand_and_a_partial_last_block_is_the_only_waste():
    pool = P.BlockPool(10, 16)
    s = pool.new_sequence()
    assert (
        pool.append(s, 40) == [0, 1, 2]
        and s.length == 40
        and pool.used_blocks == 3
        and pool.wasted_slots() == 8
    )
    assert pool.append(s, 8) == [] and pool.wasted_slots() == 0, "the last block had room"
    assert pool.append(s, 1) == [3] and pool.wasted_slots() == 15
    pool.check()


def test_free_returns_blocks_and_reuse_is_possible():
    pool = P.BlockPool(4, 8)
    a, b = pool.new_sequence(), pool.new_sequence()
    pool.append(a, 16)
    pool.append(b, 16)
    assert pool.free_blocks == 0
    pool.free(a)
    assert pool.free_blocks == 2 and pool.can_fit(16) and not pool.can_fit(17)
    c = pool.new_sequence()
    pool.append(c, 16)
    pool.check()
    with pytest.raises(KeyError):
        pool.free(a)


def test_an_append_that_does_not_fit_changes_nothing():
    pool = P.BlockPool(3, 4)
    s = pool.new_sequence()
    pool.append(s, 8)
    before = (list(s.blocks), s.length, pool.free_blocks)
    with pytest.raises(P.OutOfBlocks):
        pool.append(s, 9)  # needs 3 more blocks, only 1 is free
    assert (list(s.blocks), s.length, pool.free_blocks) == before
    pool.check()


def test_fork_shares_blocks_and_a_write_to_a_shared_block_copies_it():
    pool = P.BlockPool(8, 4)
    parent = pool.new_sequence()
    pool.append(parent, 6)  # blocks 0 (full) and 1 (2 of 4 used)
    child = pool.fork(parent)
    assert (
        child.blocks == parent.blocks
        and pool.refcount(0) == pool.refcount(1) == 2
        and pool.used_blocks == 2
    )
    new = pool.append(child, 1)  # writes into the shared, partly-filled block 1: copy-on-write
    assert (
        child.blocks[0] == parent.blocks[0]
        and child.blocks[1] != parent.blocks[1]
        and new == [child.blocks[1]]
        and pool.copies == 1
    )
    assert pool.refcount(parent.blocks[1]) == 1 and pool.used_blocks == 3
    pool.check()
    pool.free(parent)
    assert pool.refcount(child.blocks[0]) == 1 and pool.used_blocks == 2, (
        "the shared full block survives for the child"
    )
    pool.free(child)
    assert pool.used_blocks == 0
    pool.check()


def test_appending_to_a_full_shared_block_needs_no_copy():
    pool = P.BlockPool(8, 4)
    a = pool.new_sequence()
    pool.append(a, 4)
    b = pool.fork(a)
    assert pool.append(b, 1) == [b.blocks[1]] and pool.copies == 0 and b.blocks[0] == a.blocks[0]


def test_random_operations_never_break_the_invariants():
    import random

    rng = random.Random(0)
    pool = P.BlockPool(40, 8)
    live = []
    for _ in range(600):
        op = rng.choice(["new", "append", "append", "fork", "free"])
        try:
            if op == "new":
                s = pool.new_sequence()
                live.append(s)
                pool.append(s, rng.randint(1, 30))
            elif op == "append" and live:
                pool.append(rng.choice(live), rng.randint(1, 12))
            elif op == "fork" and live:
                live.append(pool.fork(rng.choice(live)))
            elif op == "free" and live:
                pool.free(live.pop(rng.randrange(len(live))))
        except P.OutOfBlocks:
            pass
        pool.check()
    for s in live:
        pool.free(s)
    assert pool.used_blocks == 0


def test_paging_wastes_far_less_than_contiguous_reservation():
    lengths = [30, 75, 120, 48, 200, 64, 90, 33]
    assert P.contiguous_waste(lengths, 2048) > 0.9 and P.paged_waste(lengths, 16) < 0.1
    assert P.paged_waste([16, 32], 16) == 0 and P.paged_waste([1], 16) == pytest.approx(15 / 16)


def test_attention_through_a_scrambled_block_table_equals_attention_over_a_contiguous_cache():
    torch.manual_seed(0)
    h, d, bs, length = 4, 8, 4, 11
    k, v, q = torch.randn(length, h, d), torch.randn(length, h, d), torch.randn(h, d)
    table = [5, 2, 7]  # three physical blocks, in no particular order
    k_pool, v_pool = torch.zeros(9, bs, h, d), torch.zeros(9, bs, h, d)
    for t in range(length):
        k_pool[table[t // bs], t % bs], v_pool[table[t // bs], t % bs] = k[t], v[t]
    got = P.paged_attention_decode(q, k_pool, v_pool, table, length)
    scores = torch.einsum("hd,thd->ht", q, k) / math.sqrt(d)
    want = torch.einsum("ht,thd->hd", torch.softmax(scores, -1), v)
    assert torch.allclose(got, want, atol=1e-6)
    k_pool[0] = 99  # an unreferenced block must not matter
    assert torch.allclose(
        P.paged_attention_decode(q, k_pool, v_pool, table, length), want, atol=1e-6
    )


# ----------------------------------------------------------------------------- the batching simulator

M = S.StepModel(base=1.0, per_seq=0.0, prefill_per_token=0.0)


def test_static_batching_returns_everything_when_the_longest_finishes():
    reqs = [S.Request(0, 0.0, 1, 10), S.Request(1, 0.0, 1, 2)]
    rep = S.simulate_static(reqs, M, max_batch=2)
    assert {d.request.id: d.finish for d in rep.done} == {0: 10.0, 1: 10.0}, (
        "the short request waits for the long one"
    )
    assert (
        rep.steps == 10
        and rep.slot_steps_total == 20
        and rep.slot_steps_used == 12
        and rep.utilisation == pytest.approx(0.6)
    )


def test_continuous_batching_returns_each_request_when_it_finishes_and_refills_the_slot():
    reqs = [S.Request(0, 0.0, 1, 10), S.Request(1, 0.0, 1, 2), S.Request(2, 0.0, 1, 3)]
    rep = S.simulate_continuous(reqs, M, max_batch=2)
    fin = {d.request.id: d.finish for d in rep.done}
    assert fin == {1: 2.0, 2: 5.0, 0: 10.0}, "request 2 takes the slot request 1 freed at t = 2"
    assert rep.utilisation == 1.0 and rep.makespan == 10.0


def test_with_one_slot_both_policies_are_the_same_and_idle_time_is_skipped():
    reqs = [S.Request(0, 0.0, 1, 3), S.Request(1, 100.0, 1, 2)]
    a, b = S.simulate_static(reqs, M, 1), S.simulate_continuous(reqs, M, 1)
    assert (
        [(d.request.id, d.finish) for d in a.done]
        == [(d.request.id, d.finish) for d in b.done]
        == [(0, 3.0), (1, 102.0)]
    )


def test_step_cost_and_prefill_cost_enter_the_timeline():
    m = S.StepModel(base=0.5, per_seq=0.25, prefill_per_token=0.01)
    rep = S.simulate_continuous([S.Request(0, 0.0, 100, 2)], m, 4)
    assert rep.done[0].first_token == pytest.approx(1.0 + 0.75) and rep.done[
        0
    ].finish == pytest.approx(1.0 + 1.5)
    assert rep.done[0].ttft == pytest.approx(1.75) and rep.done[0].latency == pytest.approx(2.5)


def test_continuous_batching_beats_static_on_a_realistic_workload_and_the_gap_grows_with_the_batch():
    m = S.StepModel(0.006, 0.0004, 0.0002)
    w = S.poisson_workload(200, 8, seed=1, long_tail=0.1)
    ratios = []
    for b in (1, 4, 8):
        st, co = S.simulate_static(w, m, b), S.simulate_continuous(w, m, b)
        assert len(st.done) == len(co.done) == 200
        ratios.append(co.throughput / st.throughput)
        assert co.latency(95) <= st.latency(95) + 1e-9 and co.utilisation >= st.utilisation
    assert ratios[0] == pytest.approx(1.0) and ratios[1] > 1.4 and ratios[2] > ratios[1]


def test_throughput_rises_with_the_batch_size_because_the_fixed_step_cost_is_shared():
    m = S.StepModel(0.006, 0.0004, 0.0)
    w = S.poisson_workload(150, 50, seed=2)
    tp = [S.simulate_continuous(w, m, b).throughput for b in (1, 2, 4, 8, 16)]
    assert tp == sorted(tp) and tp[-1] > 5 * tp[0]


def test_every_request_is_served_exactly_once_in_both_policies():
    w = S.poisson_workload(60, 5, seed=3, long_tail=0.2)
    for sim in (S.simulate_static, S.simulate_continuous):
        rep = sim(w, S.StepModel(0.01, 0.001, 0.0001), 5)
        assert sorted(d.request.id for d in rep.done) == list(range(60)) and all(
            d.finish >= d.first_token >= d.request.arrival for d in rep.done
        )


def test_the_workload_is_deterministic_and_the_step_model_fits_a_line():
    assert (
        S.poisson_workload(10, 2.0, seed=4)
        == S.poisson_workload(10, 2.0, seed=4)
        != S.poisson_workload(10, 2.0, seed=5)
    )
    base, per = S.fit_step_model([(1, 0.0070), (2, 0.0074), (4, 0.0082), (8, 0.0098)])
    assert base == pytest.approx(0.0066, abs=1e-4) and per == pytest.approx(0.0004, abs=1e-5)
    assert S.fit_step_model([(1, 0.01)]) == (0.01, 0.0)


def test_appending_fills_the_free_slots_of_the_last_block_before_taking_a_new_one():
    pool = P.BlockPool(num_blocks=8, block_size=4)
    seq = pool.new_sequence()
    pool.append(seq, 1)  # one token in a block of four: three free slots
    assert pool.used_blocks == 1
    assert (
        pool.append(seq, 3) == [] and pool.used_blocks == 1
    )  # exactly fills the block: nothing allocated
    assert len(pool.append(seq, 1)) == 1 and pool.used_blocks == 2  # now a new block
    assert (
        len(pool.append(seq, 7)) == 1 and seq.length == 12
    )  # 3 free slots + 4 + 4 -> needs one more block only
    assert pool.wasted_slots() == 0
    pool.check()


def test_a_static_batch_reports_the_first_token_after_one_step_and_everything_when_the_longest_ends():
    m = S.StepModel(base=0.1, per_seq=0.01, prefill_per_token=0.001)
    reqs = [S.Request(0, 0.0, 100, 3), S.Request(1, 0.0, 50, 5)]
    rep = S.simulate_static(reqs, m, max_batch=2)
    prefill, step = 150 * 0.001, m.step(2)
    by_id = {d.request.id: d for d in rep.done}
    assert by_id[0].first_token == pytest.approx(prefill + step)  # first token after ONE step
    assert by_id[0].finish == pytest.approx(
        prefill + 5 * step
    )  # the short request waits for the long one
    assert by_id[1].finish == pytest.approx(prefill + 5 * step)


def test_the_long_tail_share_makes_that_share_of_outputs_four_times_longer():
    none = S.poisson_workload(300, 1.0, output=(10, 20), long_tail=0.0, seed=1)
    assert max(r.output for r in none) <= 20
    every = S.poisson_workload(300, 1.0, output=(10, 20), long_tail=1.0, seed=1)
    assert min(r.output for r in every) >= 40
    some = S.poisson_workload(2000, 1.0, output=(10, 20), long_tail=0.25, seed=1)
    share = sum(r.output > 20 for r in some) / len(some)
    assert 0.2 < share < 0.3
