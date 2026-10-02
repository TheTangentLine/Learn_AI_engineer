"""Tests for the Week 9 weekly challenge: the KV cache against the uncached forward pass (and against Hugging Face's generate), top-p sampling, the benchmarks."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import bench  # noqa: E402
import blocks as B  # noqa: E402
import kvcache as K  # noqa: E402
import sampling as S  # noqa: E402
from generate import generate  # noqa: E402


def model(heads=4, kv=2, bias=False, layers=3, d=32, vocab=60, max_len=96, seed=0) -> B.Decoder:
    torch.manual_seed(seed)
    m = B.Decoder(
        B.Config(
            vocab_size=vocab,
            d_model=d,
            n_layers=layers,
            n_heads=heads,
            n_kv_heads=kv,
            max_seq_len=max_len,
            qkv_bias=bias,
        )
    ).eval()
    for p in m.parameters():  # not the tiny default init: make attention matter
        torch.nn.init.normal_(p, std=0.15) if p.dim() > 1 else None
    return m


# ----------------------------------------------------------------------------- the cache containers


def test_a_layer_cache_appends_in_place_and_returns_views_of_the_valid_part():
    lc = K.LayerCache(torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4))
    k1 = torch.ones(1, 2, 3, 4)
    keys, values = lc.append(k1, 2 * k1)
    assert (
        lc.length == 3
        and keys.shape == (1, 2, 3, 4)
        and torch.equal(keys, k1)
        and torch.equal(values, 2 * k1)
    )
    assert keys.data_ptr() == lc.k.data_ptr(), "a view of the preallocated buffer, not a copy"
    keys, _ = lc.append(5 * torch.ones(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))
    assert lc.length == 5 and torch.equal(keys[:, :, :3], k1) and (keys[:, :, 3:] == 5).all()


def test_a_full_cache_refuses_more_tokens():
    lc = K.LayerCache(torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2))
    lc.append(torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2))
    with pytest.raises(ValueError, match="would not fit"):
        lc.append(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))


def test_the_concat_cache_holds_the_same_values_as_the_preallocated_one():
    a = K.LayerCache(torch.zeros(1, 2, 10, 4), torch.zeros(1, 2, 10, 4))
    b = K.ConcatLayerCache(torch.zeros(1, 2, 10, 4), torch.zeros(1, 2, 10, 4))
    for n in (3, 1, 1, 2):
        k, v = torch.randn(1, 2, n, 4), torch.randn(1, 2, n, 4)
        ka, va = a.append(k, v)
        kb, vb = b.append(k, v)
        assert torch.equal(ka, kb) and torch.equal(va, vb) and a.length == b.length


def test_cache_size_matches_the_day_6_formula_and_reset_empties_it():
    import arch

    m = model(heads=8, kv=2, d=64, layers=2)
    cache = K.KVCache.for_model(m, 2, 50)
    with torch.no_grad():
        K.forward_cached(m, torch.randint(0, 60, (2, 17)), cache)
    spec = arch.ModelSpec("t", 60, 64, 2, 8, 2, m.cfg.d_ff)
    assert (
        cache.nbytes() == arch.kv_cache_bytes(spec, 17, batch=2, dtype_bytes=4)
        and cache.length == 17
    )
    assert cache.reserved_bytes() > cache.nbytes()
    cache.reset()
    assert cache.length == 0 and cache.nbytes() == 0


# ----------------------------------------------------------------------------- the cached forward pass


@pytest.mark.parametrize(
    ("heads", "kv", "bias"),
    [(4, 4, False), (4, 2, False), (4, 1, False), (8, 2, True), (6, 3, True)],
)
def test_prefill_then_single_token_decoding_equals_the_full_forward_pass(heads, kv, bias):
    m = model(heads=heads, kv=kv, bias=bias, d=heads * 8)
    ids = torch.randint(0, 60, (2, 23), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = m(ids)
        cache = K.KVCache.for_model(m, 2, 40)
        parts = [K.forward_cached(m, ids[:, :9], cache)] + [
            K.forward_cached(m, ids[:, i : i + 1], cache) for i in range(9, 23)
        ]
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-5)


def test_chunks_of_any_size_work_including_a_second_multi_token_chunk():
    m = model()
    ids = torch.randint(0, 60, (1, 30), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = m(ids)
        cache = K.KVCache.for_model(m, 1, 40)
        parts, i = [], 0
        for n in (7, 1, 12, 1, 1, 8):
            parts.append(K.forward_cached(m, ids[:, i : i + n], cache))
            i += n
    assert i == 30 and torch.allclose(torch.cat(parts, 1), full, atol=1e-5)


def test_the_grouped_single_token_path_equals_the_repeated_head_path():
    m = model(heads=8, kv=2, d=64)
    ids = torch.randint(0, 60, (3, 20), generator=torch.Generator().manual_seed(3))
    outs = []
    for grouped in (True, False):
        cache = K.KVCache.for_model(m, 3, 30)
        with torch.no_grad():
            K.forward_cached(m, ids[:, :15], cache, grouped=grouped)
            outs.append(
                torch.cat(
                    [
                        K.forward_cached(m, ids[:, i : i + 1], cache, grouped=grouped)
                        for i in range(15, 20)
                    ],
                    1,
                )
            )
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_the_grouped_path_really_avoids_copying_the_key_value_heads(monkeypatch):
    m = model(heads=8, kv=2, d=64)
    calls = []
    real = torch.Tensor.repeat_interleave
    monkeypatch.setattr(
        torch.Tensor,
        "repeat_interleave",
        lambda self, *a, **k: calls.append(1) or real(self, *a, **k),
    )
    for grouped, expect_copies in ((True, False), (False, True)):
        calls.clear()
        cache = K.KVCache.for_model(m, 1, 20)
        with torch.no_grad():
            K.forward_cached(
                m, torch.randint(0, 60, (1, 6)), cache, grouped=grouped
            )  # prefill: always the repeated path
            calls.clear()
            K.forward_cached(m, torch.randint(0, 60, (1, 1)), cache, grouped=grouped)
        assert bool(calls) == expect_copies, grouped


def test_the_model_cannot_be_run_past_its_maximum_length():
    m = model(max_len=16)
    cache = K.KVCache.for_model(m, 1, 16)
    with torch.no_grad():
        K.forward_cached(m, torch.randint(0, 60, (1, 16)), cache)
        with pytest.raises(ValueError, match="maximum length"):
            K.forward_cached(m, torch.randint(0, 60, (1, 1)), cache)


def test_a_cache_filled_for_a_different_prefix_gives_different_logits():
    """The cache is real state: decoding the same last token after two different prompts must not give the same answer."""
    m = model()
    outs = []
    for prompt in ([1, 2, 3, 4, 5], [9, 8, 7, 6, 5]):
        cache = K.KVCache.for_model(m, 1, 20)
        with torch.no_grad():
            K.forward_cached(m, torch.tensor([prompt]), cache)
            outs.append(K.forward_cached(m, torch.tensor([[11]]), cache))
    assert not torch.allclose(outs[0], outs[1], atol=1e-3)


# ----------------------------------------------------------------------------- generation


def test_all_three_modes_generate_the_same_tokens_greedily_and_when_sampling_with_one_seed():
    m = model()
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    greedy = [generate(m, prompt, 25, mode=mode).tokens for mode in ("nocache", "concat", "cache")]
    assert greedy[0] == greedy[1] == greedy[2] and len(greedy[0]) == 25
    sampled = [
        generate(m, prompt, 25, mode=mode, temperature=0.9, top_p=0.9, seed=5).tokens
        for mode in ("nocache", "concat", "cache")
    ]
    assert sampled[0] == sampled[1] == sampled[2]
    assert sampled[0] != generate(m, prompt, 25, temperature=0.9, top_p=0.9, seed=6).tokens


def test_generation_agrees_with_hugging_faces_generate_on_a_random_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(4)
    cfg = Qwen2Config(
        vocab_size=80,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    hf = Qwen2ForCausalLM(cfg).eval()
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.2) if p.dim() > 1 else None
    mine = B.Decoder(B.config_from_hf(cfg)).eval()
    B.load_hf_state_dict(mine, hf.state_dict())
    prompt = [5, 17, 3, 42, 8]
    ref = hf.generate(torch.tensor([prompt]), max_new_tokens=20, do_sample=False, pad_token_id=0)[
        0, 5:
    ].tolist()
    assert generate(mine, prompt, 20, mode="cache").tokens == ref


def test_stop_tokens_end_generation_and_are_included():
    m = model()
    prompt = [1, 2, 3]
    free = generate(m, prompt, 20).tokens
    stop = free[6]
    g = generate(m, prompt, 20, stop_ids={stop})
    assert g.tokens == free[: free.index(stop) + 1] and g.stopped_on == stop


def test_timing_fields_are_filled_and_prompt_plus_output_must_fit():
    m = model(max_len=32)
    g = generate(m, [1, 2, 3], 10)
    assert (
        g.prefill_s > 0
        and len(g.step_s) == 9
        and g.tokens_per_second > 0
        and g.total_s >= sum(g.step_s)
    )
    with pytest.raises(ValueError, match="exceeds"):
        generate(m, [1] * 30, 10)


# ----------------------------------------------------------------------------- sampling


def test_top_p_keeps_the_smallest_set_reaching_p():
    """Cumulative mass after each token: 0.5, 0.8, 0.9, 0.96, 1.0. (Exactly AT a boundary, such as p = 0.8, float rounding decides: every
    implementation has that ambiguity, so the tests stay away from it.)"""
    probs = torch.tensor([0.5, 0.3, 0.1, 0.06, 0.04])
    logits = probs.log()
    kept = lambda p: set(S.filtered_probs(logits, top_p=p).nonzero().flatten().tolist())  # noqa: E731
    assert kept(0.4) == {0} and kept(0.49) == {0}, "the first token alone already reaches 0.49"
    assert kept(0.51) == {0, 1} and kept(0.79) == {0, 1}
    assert (
        kept(0.81) == {0, 1, 2}
        and kept(0.89) == {0, 1, 2}
        and kept(0.95) == {0, 1, 2, 3}
        and kept(0.97) == {0, 1, 2, 3, 4}
        and kept(1.0) == {0, 1, 2, 3, 4}
    )
    out = S.filtered_probs(logits, top_p=0.79)
    assert torch.allclose(
        out, torch.tensor([0.5, 0.3, 0, 0, 0]) / 0.8, atol=1e-6
    ) and out.sum().item() == pytest.approx(1.0)


def test_top_p_always_keeps_at_least_the_best_token_and_is_order_independent():
    logits = torch.tensor([1.0, 5.0, 2.0, 0.0])
    assert S.filtered_probs(logits, top_p=1e-6).tolist() == [0, 1, 0, 0]
    shuffled = logits[[2, 0, 3, 1]]
    a, b = S.filtered_probs(logits, top_p=0.9), S.filtered_probs(shuffled, top_p=0.9)
    assert torch.allclose(a[[2, 0, 3, 1]], b)


def test_top_k_keeps_the_k_largest_and_ties_at_the_boundary_are_kept():
    logits = torch.tensor([3.0, 1.0, 2.0, 0.0])
    assert S.filtered_probs(logits, top_k=2).nonzero().flatten().tolist() == [0, 2]
    tie = torch.tensor([2.0, 2.0, 2.0, 0.0])
    assert S.filtered_probs(tie, top_k=2).nonzero().flatten().tolist() == [0, 1, 2]
    assert torch.allclose(S.filtered_probs(logits, top_k=10), F.softmax(logits, -1))


def test_temperature_zero_is_greedy_and_temperature_rescales_the_logits():
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.0, 0.0, 5.0]])
    assert S.filtered_probs(logits, temperature=0).tolist() == [[0, 1, 0], [0, 0, 1]]
    assert torch.allclose(
        S.filtered_probs(logits[:1], temperature=2.0), F.softmax(logits[:1] / 2.0, -1)
    )
    cold, hot = (
        S.filtered_probs(logits[:1], temperature=0.1),
        S.filtered_probs(logits[:1], temperature=10.0),
    )
    assert cold.max() > 0.99 and hot.max() < 0.4


def test_top_k_then_top_p_compose_and_probabilities_are_a_distribution():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    out = S.filtered_probs(logits, temperature=0.8, top_k=20, top_p=0.9)
    assert torch.allclose(out.sum(-1), torch.ones(6), atol=1e-6) and (out >= 0).all()
    assert ((out > 0).sum(-1) <= 20).all() and ((out > 0).sum(-1) >= 1).all()
    assert (S.filtered_probs(logits, top_p=0.9) > 0).sum() <= (
        S.filtered_probs(logits, top_p=0.99) > 0
    ).sum()


def test_bad_arguments_are_rejected():
    for kw in ({"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}, {"temperature": -1.0}):
        with pytest.raises(ValueError):
            S.filtered_probs(torch.zeros(5), **kw)


def test_sampling_frequencies_follow_the_filtered_distribution():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]]).repeat(5000, 1)
    gen = torch.Generator().manual_seed(0)
    draws = S.sample(logits, temperature=1.0, top_p=0.8, generator=gen)
    freq = torch.bincount(draws, minlength=4).float() / 5000
    assert torch.allclose(freq, S.filtered_probs(logits[0], top_p=0.8), atol=0.03)
    assert freq[3] == 0 and S.sample(logits[:3], temperature=0).tolist() == [0, 0, 0]


# ----------------------------------------------------------------------------- the benchmark helpers


def test_sampling_metrics_by_hand():
    samples = [
        {"tokens": [1, 2, 3, 4], "logprobs": [math.log(0.5)] * 4},
        {"tokens": [1, 2, 1, 2, 1, 2, 1, 2], "logprobs": [math.log(0.25)] * 8},
    ]
    m = bench.sampling_metrics(samples)
    assert m["mean_nll"] == pytest.approx((4 * math.log(2) + 8 * math.log(4)) / 12) and m[
        "perplexity"
    ] == pytest.approx(math.exp(m["mean_nll"]))
    assert m["distinct2"] == pytest.approx(4 / 10), (
        "10 bigrams in all; (1,2), (2,3), (3,4) and (2,1) are the distinct ones"
    )
    assert m["loop_rate"] == 0.5, "only the second sample repeats a 4-gram in its last 20 tokens"


def test_compare_modes_reports_identical_tokens_and_a_speed_up_on_a_longer_run():
    m = bench.bench_model(d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, vocab=100, max_len=200)
    res = bench.compare_modes(m, 16, 48, repeats=1)
    assert all(r["same_tokens"] for r in res.values())
    assert res["cache"]["total_s"] < res["nocache"]["total_s"]


def test_cache_memory_helper_agrees_with_the_formula():
    m = bench.bench_model(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab=100, max_len=100)
    r = bench.cache_memory(m, 30)
    assert r["valid_bytes"] == r["formula"] and r["reserved_bytes"] > r["valid_bytes"]


def test_sample_batch_uses_the_cache_and_returns_a_logprob_per_token():
    m = model()
    out = bench.sample_batch(
        m, [[1, 2, 3], [4, 5]], 12, temperature=1.0, top_k=None, top_p=0.9, seed=0
    )
    assert [len(s["tokens"]) for s in out] == [12, 12] and all(
        lp <= 0 for s in out for lp in s["logprobs"]
    )
    again = bench.sample_batch(
        m, [[1, 2, 3], [4, 5]], 12, temperature=1.0, top_k=None, top_p=0.9, seed=0
    )
    assert [s["tokens"] for s in out] == [s["tokens"] for s in again]
    greedy = bench.sample_batch(m, [[1, 2, 3]], 12, temperature=0.0, top_k=None, top_p=None)
    assert greedy[0]["tokens"] == generate(m, [1, 2, 3], 12).tokens
