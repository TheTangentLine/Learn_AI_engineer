"""Tests for Week 11 Day 3: memory and cost arithmetic (by hand, and the inequalities that must hold), and speculative decoding (exactness, acceptance, the sampling distribution)."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))
sys.path.append(
    str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions/weekly/fastgen")
)

import blocks as B  # noqa: E402
import economics as E  # noqa: E402
import speculative as SP  # noqa: E402
from generate import generate  # noqa: E402

LLAMA8B = E.Spec(params=8_030_261_248, layers=32, kv_heads=8, head_dim=128)

# ----------------------------------------------------------------------------- memory


def test_kv_bytes_per_token_and_weight_bytes_by_hand():
    assert LLAMA8B.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2 == 131_072
    assert (
        LLAMA8B.kv_bytes_per_token(1) == 65_536
        and LLAMA8B.weight_bytes(2) == 2 * 8_030_261_248
        and LLAMA8B.weight_bytes(0.5) == 0.5 * 8_030_261_248
    )


def test_a_24gb_gpu_holds_llama3_8b_in_bf16_and_a_few_long_sequences():
    f = E.fit_on_gpu(LLAMA8B, gpu_gb=24, context=8192)
    assert f.weights_gb == pytest.approx(16.06, abs=0.01) and f.cache_budget_gb == pytest.approx(
        24 - 16.06 - 1.5, abs=0.01
    )
    assert (
        f.kv_per_sequence_gb == pytest.approx(131_072 * 8192 / 1e9)
        and f.max_concurrent == 5
        and f.fits
    )


def test_a_model_that_does_not_fit_has_no_capacity_and_more_gpus_or_fewer_bits_fix_it():
    big = E.Spec(params=70_000_000_000, layers=80, kv_heads=8, head_dim=128)
    one = E.fit_on_gpu(big, gpu_gb=80, context=4096)
    assert not one.fits and one.max_concurrent == 0 and one.cache_budget_gb == 0
    assert E.fit_on_gpu(big, gpu_gb=80, context=4096, gpus=2).max_concurrent > 0
    assert E.fit_on_gpu(big, gpu_gb=80, context=4096, weight_bytes=0.5).fits, (
        "4-bit weights fit on one GPU"
    )
    assert (
        E.min_gpus(big, gpu_gb=80, context=4096, concurrent=32) == 3
        and E.min_gpus(big, gpu_gb=1, context=4096, concurrent=1, limit=4) is None
    )


def test_halving_the_cache_precision_doubles_the_concurrency_and_a_longer_context_divides_it():
    a = E.fit_on_gpu(LLAMA8B, gpu_gb=40, context=4096)
    assert E.fit_on_gpu(LLAMA8B, gpu_gb=40, context=4096, kv_bytes=1).max_concurrent in (
        2 * a.max_concurrent,
        2 * a.max_concurrent + 1,
    )
    assert E.fit_on_gpu(LLAMA8B, gpu_gb=40, context=8192).max_concurrent in (
        a.max_concurrent // 2,
        a.max_concurrent // 2 + 1,
    )


def test_the_bandwidth_ceiling_grows_with_batch_and_is_bounded_by_the_weight_read():
    one = E.decode_tokens_per_second_ceiling(LLAMA8B, bandwidth_gbs=1000, context=1024, batch=1)
    assert one == pytest.approx(1000e9 / (16.06e9 + 131_072 * 1024), rel=1e-3)
    ceilings = [
        E.decode_tokens_per_second_ceiling(LLAMA8B, bandwidth_gbs=1000, context=1024, batch=b)
        for b in (1, 4, 16, 64)
    ]
    assert ceilings == sorted(ceilings) and ceilings[-1] < 1000e9 / 131_072 / 1024, (
        "the cache read per sequence is the asymptote"
    )
    assert (
        E.decode_tokens_per_second_ceiling(LLAMA8B, bandwidth_gbs=1000, weight_bytes=0.5, batch=1)
        > one
    )


# ----------------------------------------------------------------------------- money

API = E.ApiPrice(input_per_m=1.0, output_per_m=4.0)
HOST = E.SelfHost(
    gpu_per_hour=2.0,
    tokens_per_second=1000,
    prefill_tokens_per_second=10_000,
    utilisation=0.5,
    min_gpus=1,
    ops_per_month=0.0,
)


def test_the_api_cost_is_tokens_times_price_per_million():
    assert API.cost(1e6, 1e6) == 5.0 and API.cost(0, 0) == 0 and API.cost(2e6, 0) == 2.0


def test_gpus_needed_is_busy_time_over_available_time_rounded_up_with_a_floor():
    month_s = E.HOURS_PER_MONTH * 3600
    busy_for_one_gpu = (
        0.5 * month_s * 1000
    )  # output tokens that keep one GPU 50% utilised for a month with no input
    assert (
        HOST.gpus_needed(0, busy_for_one_gpu) == 1
        and HOST.gpus_needed(0, busy_for_one_gpu * 1.01) == 2
    )
    assert HOST.gpus_needed(0, 1) == 1, "the minimum fleet is paid for even with no traffic"
    assert E.SelfHost(2.0, 1000, 10_000, 0.5, min_gpus=3).gpus_needed(0, 1) == 3


def test_self_hosted_cost_is_gpus_times_hours_plus_operations():
    h = E.SelfHost(2.0, 1000, 10_000, ops_per_month=500.0)
    assert h.cost(0, 1) == pytest.approx(2.0 * E.HOURS_PER_MONTH + 500.0)


def test_compare_names_the_cheaper_option_and_the_ratio():
    low = E.compare(1e6, 1e5, API, HOST)
    assert (
        low.cheaper == "api"
        and low.api == API.cost(1e6, 1e5)
        and low.selfhost == pytest.approx(2.0 * E.HOURS_PER_MONTH)
        and low.ratio > 1
    )
    high = E.compare(3e10, 1e10, API, HOST)
    assert (
        high.cheaper == "self-hosted"
        and high.gpus == HOST.gpus_needed(3e10, 1e10)
        and high.ratio < 1
    )


def test_the_break_even_volume_is_where_the_costs_cross_and_moves_the_right_way_with_each_assumption():
    be = E.break_even_tokens_per_month(API, HOST)
    below, above = (
        E.compare(be * 0.5 * 3, be * 0.5, API, HOST),
        E.compare(be * 1.5 * 3, be * 1.5, API, HOST),
    )
    assert below.cheaper == "api" and above.cheaper == "self-hosted"
    assert E.break_even_tokens_per_month(API, E.SelfHost(4.0, 1000, 10_000, 0.5)) > be, (
        "a dearer GPU needs more volume"
    )
    faster = E.SelfHost(2.0, 2000, 20_000, 0.5)
    assert E.break_even_tokens_per_month(API, faster) == pytest.approx(be), (
        "while ONE GPU is enough the crossover is set by its rent, not its speed"
    )
    assert faster.gpus_needed(3e10, 1e10) < HOST.gpus_needed(3e10, 1e10), (
        "but a faster GPU serves more volume before a second one is needed"
    )
    assert (
        E.break_even_tokens_per_month(API, E.SelfHost(2.0, 1000, 10_000, 0.5, ops_per_month=2000))
        > be
    ), "operations cost raises it"


def test_break_even_is_none_when_the_gpu_can_never_beat_the_api():
    cheap_api = E.ApiPrice(0.0001, 0.0001)
    assert E.break_even_tokens_per_month(cheap_api, HOST) is None
    dear = E.break_even_tokens_per_month(E.ApiPrice(1000, 1000), HOST)
    assert dear is not None and dear < 2e6, (
        "an API that charges $1,000 per million tokens loses to one GPU-month after about a million"
    )


def test_sensitivity_changes_one_assumption_at_a_time():
    s = E.sensitivity(HOST, API, {"gpu_per_hour": [0.5, 1, 2], "utilisation": [0.5, 2, 4]})
    rows = dict(s["gpu_per_hour"])
    assert (
        rows[1] == pytest.approx(E.break_even_tokens_per_month(API, HOST))
        and rows[0.5] < rows[1] < rows[2]
    )
    assert dict(s["utilisation"])[4] == dict(s["utilisation"])[2], "utilisation is capped at 100%"


# ----------------------------------------------------------------------------- speculative decoding


def tiny(seed: int, vocab: int = 40, d: int = 32, layers: int = 2) -> B.Decoder:
    torch.manual_seed(seed)
    m = B.Decoder(
        B.Config(
            vocab_size=vocab, d_model=d, n_layers=layers, n_heads=4, n_kv_heads=2, max_seq_len=256
        )
    ).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.5)
    return m


@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_greedy_speculative_decoding_produces_exactly_the_targets_tokens(k):
    target, draft = tiny(1), tiny(2)
    prompt = [3, 14, 15, 9, 26]
    ref = generate(target, prompt, 30, mode="cache").tokens
    st = SP.speculative_generate(target, draft, prompt, 30, k=k)
    assert st.tokens == ref and len(st.tokens) == 30


def test_a_perfect_draft_is_always_accepted_and_each_round_yields_k_plus_one_tokens():
    target = tiny(3)
    st = SP.speculative_generate(target, target, [1, 2, 3], 41, k=4)
    assert (
        st.acceptance_rate == 1.0
        and st.tokens == generate(target, [1, 2, 3], 41, mode="cache").tokens
    )
    assert st.target_forwards == 1 + st.rounds and st.tokens_per_target_forward == pytest.approx(
        41 / (1 + 8)
    )


def test_a_useless_draft_is_mostly_rejected_but_the_output_is_still_exact():
    target, draft = tiny(4), tiny(5)
    st = SP.speculative_generate(target, draft, [7, 8, 9], 25, k=4)
    assert (
        st.acceptance_rate < 0.5
        and st.tokens == generate(target, [7, 8, 9], 25, mode="cache").tokens
    )
    assert st.tokens_per_target_forward >= 1.0, (
        "every round produces at least one token (the replacement)"
    )


def test_stop_tokens_end_generation_and_the_token_budget_is_respected():
    target, draft = tiny(6), tiny(7)
    free = SP.speculative_generate(target, draft, [1, 2], 20, k=3).tokens
    stop = free[5]
    st = SP.speculative_generate(target, draft, [1, 2], 20, k=3, stop_ids={stop})
    assert st.tokens == free[: free.index(stop) + 1]
    assert len(SP.speculative_generate(target, draft, [1, 2], 7, k=5).tokens) == 7


def test_the_models_must_share_a_vocabulary_and_truncate_cannot_grow_a_cache():
    with pytest.raises(ValueError, match="vocabulary"):
        SP.speculative_generate(tiny(1, vocab=40), tiny(2, vocab=41), [1], 3)
    from kvcache import KVCache

    cache = KVCache.for_model(tiny(1), 1, 16)
    with pytest.raises(ValueError):
        SP.truncate(cache, 5)


def test_sampling_speculative_decoding_samples_the_targets_distribution():
    """Two tokens over a 5-token vocabulary: the empirical joint distribution must match the target's exact joint (rejection sampling is lossless)."""
    vocab = 5
    target, draft = tiny(8, vocab=vocab, d=16, layers=1), tiny(9, vocab=vocab, d=16, layers=1)
    prompt = [1, 2]
    with torch.no_grad():
        p1 = torch.softmax(target(torch.tensor([prompt]))[0, -1].float(), -1)
        exact = {}
        for a in range(vocab):
            p2 = torch.softmax(target(torch.tensor([[*prompt, a]]))[0, -1].float(), -1)
            for b in range(vocab):
                exact[(a, b)] = float(p1[a] * p2[b])
    n = 2500
    counts = dict.fromkeys(exact, 0)
    for seed in range(n):
        toks = SP.speculative_generate(
            target, draft, prompt, 2, k=2, temperature=1.0, seed=seed
        ).tokens
        counts[tuple(toks)] += 1
    chi2 = sum((counts[o] - n * p) ** 2 / (n * p) for o, p in exact.items() if n * p > 5)
    assert chi2 < 45, (
        f"chi-square {chi2:.1f} over {len(exact)} cells (the 99.9th percentile with 24 degrees of freedom is about 51)"
    )
    # and a deliberately WRONG sampler (always accept) fails the same test: the check can fail
    wrong = dict.fromkeys(exact, 0)
    for seed in range(n):
        wrong[
            tuple(
                SP.speculative_generate(
                    draft, draft, prompt, 2, k=2, temperature=1.0, seed=seed
                ).tokens
            )
        ] += 1
    assert sum((wrong[o] - n * p) ** 2 / (n * p) for o, p in exact.items() if n * p > 5) > 45


def test_the_expected_tokens_per_round_and_speedup_formulas():
    assert SP.expected_tokens_per_round(1.0, 4) == 5 and SP.expected_tokens_per_round(0.0, 4) == 1
    assert SP.expected_tokens_per_round(0.5, 2) == pytest.approx(1 + 0.5 + 0.25)
    assert SP.expected_speedup(0.8, 4, 0.1) == pytest.approx(
        SP.expected_tokens_per_round(0.8, 4) / 1.4
    )
    assert SP.expected_speedup(0.2, 4, 0.5) < 1.0, (
        "a poor draft that is not cheap makes decoding slower"
    )
    best = max(range(1, 12), key=lambda k: SP.expected_speedup(0.8, k, 0.1))
    assert 3 <= best <= 8


def test_probs_of_is_a_one_hot_for_greedy_and_a_softmax_otherwise():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert SP.probs_of(logits, 0).tolist() == [[0, 1, 0]] and torch.allclose(
        SP.probs_of(logits, 1.0), torch.softmax(logits, -1)
    )
    assert SP.probs_of(logits, 0.5)[0, 1] > SP.probs_of(logits, 1.0)[0, 1]


_ = (itertools, math)


# ----------------------------------------------------------------------------- edges found by mutation checks


def test_weights_that_fit_but_leave_room_for_no_sequence_do_not_count_as_fitting():
    f = E.fit_on_gpu(
        LLAMA8B, gpu_gb=18, context=8192
    )  # 16.06 GB weights + 1.5 GB overhead leaves 0.44 GB < 1.07 GB per sequence
    assert f.cache_budget_gb > 0 and f.max_concurrent == 0 and not f.fits


def test_prefill_time_adds_gpus_when_prompts_dominate():
    sh = E.SelfHost(
        gpu_per_hour=1.0, tokens_per_second=1000, prefill_tokens_per_second=10, utilisation=1.0
    )
    month_seconds = E.HOURS_PER_MONTH * 3600
    out_only = sh.gpus_needed(0, 1000 * month_seconds * 0.5)  # half a GPU of generation
    with_prompts = sh.gpus_needed(
        10 * month_seconds * 1.2, 1000 * month_seconds * 0.5
    )  # plus 1.2 GPUs of prefill
    assert out_only == 1 and with_prompts == 2


def test_equal_costs_are_reported_as_equal_and_a_bound_that_only_ties_is_not_a_win():
    api = E.ApiPrice(
        input_per_m=0.0, output_per_m=730.0
    )  # a million output tokens cost exactly one GPU-month at $1/hour
    sh = E.SelfHost(gpu_per_hour=1.0, tokens_per_second=1e9, prefill_tokens_per_second=1e9)
    c = E.compare(0, 1e6, api, sh)
    assert (
        c.api == c.selfhost == pytest.approx(730.0)
        and c.cheaper == "equal"
        and c.ratio == pytest.approx(1.0)
    )
    assert (
        E.break_even_tokens_per_month(api, sh, input_ratio=0.0, search_to=1e6) is None
    )  # tied at the top of the range


def test_utilisation_cannot_be_pushed_past_one_in_the_sensitivity_table():
    api = E.ApiPrice(0.5, 1.5)
    base = E.SelfHost(
        gpu_per_hour=0.1,
        tokens_per_second=100,
        prefill_tokens_per_second=800,
        utilisation=0.8,
        ops_per_month=500.0,
    )
    got = E.sensitivity(base, api, {"utilisation": [2.0]})["utilisation"][0][1]
    clamped = E.break_even_tokens_per_month(
        api, E.SelfHost(**{**base.__dict__, "utilisation": 1.0})
    )
    unclamped = E.break_even_tokens_per_month(
        api, E.SelfHost(**{**base.__dict__, "utilisation": 1.6})
    )
    assert clamped != pytest.approx(unclamped, rel=1e-3)  # this scenario can tell the two apart
    assert got == pytest.approx(clamped)


def test_a_stop_token_inside_an_accepted_round_ends_the_output_right_there():
    target = tiny(3)
    prompt = [1, 2, 3]
    ref = generate(target, prompt, 20, mode="cache").tokens
    i = next(
        j for j in range(2, 6) if ref[j] not in ref[:j]
    )  # a stop token that falls mid-round (k = 4)
    st = SP.speculative_generate(target, target, prompt, 20, k=4, stop_ids={ref[i]})
    assert st.tokens == ref[: i + 1] and st.tokens[-1] == ref[i]
