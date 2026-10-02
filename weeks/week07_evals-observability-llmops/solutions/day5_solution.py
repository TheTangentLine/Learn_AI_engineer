"""Week 7 Day 5 - Solution: cut the Week 6 support system's cost by 40% at an equal eval score.

Method (every step is a function here):
  1. MEASURE   run the 50 Day 1 cases under tracing; tokens and model calls come from the spans, dollars from a SIMULATED
               price card (the local model is free), latency from an ASSUMED model.
  2. LEVERS    each is a real option of the support system, one change at a time and then combined:
                 lean    shorter system prompts + compact tool specs         (fewer input tokens on every call)
                 hide    do not offer the lookup tool the system already ran (no repeated call, a shorter spec)
                 direct  build the refund reply from the tool result in code (one model call fewer)
               plus a response cache (measured on a simulated traffic mix) and a cascade (projected from trace checks).
  3. DECIDE    "equal score" is not "same number": the Day 3 gate decides, on DEV first, and TEST is looked at once.

  uv run python weeks/week07_evals-observability-llmops/solutions/day5_solution.py run <variant>|all
  uv run python weeks/week07_evals-observability-llmops/solutions/day5_solution.py report
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import costlab as cl  # noqa: E402
import day1_solution as d1  # noqa: E402
import day3_solution as d3  # noqa: E402
import evalcases as ec  # noqa: E402
import evalgate as eg  # noqa: E402
import traceeval as te  # noqa: E402

from common import tracing  # noqa: E402

OUT = ROOT / "outputs"
BASE_OPTS = {"triage_mode": "rules", "prefetch_invoice": True}  # Day 1's best configuration

# Each variant differs from BASE by exactly the levers in its name.
LEVERS = {
    "lean": {"lean": True},
    "hide": {"hide_prefetched": True},
    "direct": {"reply_from_tool": True},
}
VARIANTS: dict[str, dict] = {"base": dict(BASE_OPTS)}
for _name in (
    "lean",
    "hide",
    "direct",
    "lean+hide",
    "lean+direct",
    "hide+direct",
    "lean+hide+direct",
):
    VARIANTS[_name] = {
        **BASE_OPTS,
        **{k: v for part in _name.split("+") for k, v in LEVERS[part].items()},
    }


# ----------------------------------------------------------------------------- running and loading


def run_variant_traced(name: str, *, provider: str, model: str | None = None):
    """All 50 cases for one variant, in-memory tracing. Returns (runs, spans)."""
    cases = ec.build_cases(lambda: None)
    runs = []
    with tracing.capture() as rec:
        for case in cases:
            attrs = {"app.eval.case_id": case.id, "app.eval.split": case.split}
            with tracing.span("eval.case", attrs):
                runs.extend(
                    d1.run_variant([case], name, provider=provider, model=model, **VARIANTS[name])
                )
    return runs, rec.spans


def save_variant(name: str, runs, spans, out: Path = OUT) -> None:
    d1.save(runs, out / f"w7d5_{name}.jsonl")
    path = out / f"w7d5_{name}_spans.jsonl"
    path.write_text("".join(json.dumps(tracing.asdict(s)) + "\n" for s in spans))


def load_variant(name: str, out: Path = OUT):
    runs = d1.load(out / f"w7d5_{name}.jsonl")
    spans = tracing.load_spans(out / f"w7d5_{name}_spans.jsonl")
    return runs, spans


def run_local(name: str) -> None:
    from common import llm
    from common.local_server import LocalOpenAIServer

    with LocalOpenAIServer() as srv:
        os.environ["OLLAMA_BASE_URL"] = srv.url
        llm._ollama.cache_clear()
        runs, spans = run_variant_traced(name, provider="ollama", model="local-qwen")
    save_variant(name, runs, spans)
    print(f"{name}: {sum(r.passed for r in runs)}/{len(runs)} passed, {len(spans)} spans")


# ----------------------------------------------------------------------------- per-variant numbers


def per_case(spans) -> dict[str, list]:
    """case id -> the spans of that case's trace."""
    out: dict[str, list] = {}
    for group in te.split_traces(spans).values():
        root = next((s for s in group if s.name == "eval.case"), None)
        if root is not None:
            out[root.get("app.eval.case_id")] = group
    return out


def usage(spans_by_case: dict[str, list], ids: list[str] | None = None) -> dict:
    ids = ids if ids is not None else list(spans_by_case)
    calls = [cl.chat_calls(spans_by_case[i]) for i in ids]
    n = max(len(ids), 1)
    return {
        "cases": len(ids),
        "model_calls": sum(len(c) for c in calls) / n,
        "input_tokens": sum(i for c in calls for i, _ in c) / n,
        "output_tokens": sum(o for c in calls for _, o in c) / n,
    }


def cost_per_1k(spans_by_case: dict[str, list], card: cl.PriceCard, ids=None) -> float:
    ids = ids if ids is not None else list(spans_by_case)
    return sum(cl.trace_cost(spans_by_case[i], card) for i in ids) / max(len(ids), 1) * 1000


def latencies(spans_by_case: dict[str, list], ids=None) -> list[float]:
    ids = ids if ids is not None else list(spans_by_case)
    return [cl.trace_latency(spans_by_case[i]) for i in ids]


def suite_of(runs, ids: set[str] | None = None) -> eg.SuiteResult:
    return d3.suite_from_runs([r for r in runs if ids is None or r.case_id in ids], "x")


def split_ids(runs, split: str) -> set[str]:
    return {r.case_id for r in runs if r.split == split}


def decide(base_runs, cand_runs, split: str | None = None, critical=d3.CRITICAL) -> eg.Decision:
    """The Day 3 gate, restricted to one split if asked."""
    ids = split_ids(base_runs, split) if split else None
    return eg.compare(suite_of(base_runs, ids), suite_of(cand_runs, ids), critical=critical)


def choose_on_dev(results: dict[str, dict], min_dev_rate_of: str = "base") -> str | None:
    """The cheapest variant whose DEV pass rate is not below the baseline's. ``results[name]`` needs ``dev_rate``
    and ``cost``. Test is never consulted: it is reported afterwards, once."""
    floor = results[min_dev_rate_of]["dev_rate"]
    ok = {n: r for n, r in results.items() if r["dev_rate"] >= floor}
    return min(ok, key=lambda n: (ok[n]["cost"], n)) if ok else None


def summarize_all(
    names: list[str], out: Path = OUT, card: cl.PriceCard = cl.CHEAP
) -> dict[str, dict]:
    loaded = {n: load_variant(n, out) for n in names}
    base_runs = loaded["base"][0]
    dev, test = split_ids(base_runs, "dev"), split_ids(base_runs, "test")
    rows = {}
    for n, (runs, spans) in loaded.items():
        by_case = per_case(spans)
        lat = latencies(by_case)
        d_all, d_dev, d_test = (decide(base_runs, runs, s) for s in (None, "dev", "test"))
        rows[n] = {
            "rate": sum(r.passed for r in runs) / len(runs),
            "dev_rate": sum(r.passed for r in runs if r.case_id in dev) / len(dev),
            "test_rate": sum(r.passed for r in runs if r.case_id in test) / len(test),
            **usage(by_case),
            "cost": cost_per_1k(by_case, card),
            "cost_dev": cost_per_1k(by_case, card, sorted(dev)),
            "latency_p50": te.percentile(lat, 50),
            "latency_p95": te.percentile(lat, 95),
            "gate_all": d_all.status,
            "gate_dev": d_dev.status,
            "gate_test": d_test.status,
            "diff_dev": d_dev.diff,
            "ci_dev": d_dev.ci,
            "diff_test": d_test.diff,
            "ci_test": d_test.ci,
            "critical": d_all.critical_regressions,
            "regressed": d_all.regressions,
            "gained": d_all.gains,
        }
    return rows


def report(names: list[str] | None = None, out: Path = OUT) -> str:
    names = names or [n for n in VARIANTS if (out / f"w7d5_{n}.jsonl").exists()]
    rows = summarize_all(names, out)
    base = rows["base"]
    lines = [
        f"{'variant':<20}{'calls':>6}{'in tok':>8}{'out tok':>8}{'$/1k':>8}{'vs base':>9}"
        f"{'dev':>6}{'test':>6}{'all':>6}  gate(dev/test/all)"
    ]
    for n, r in rows.items():
        lines.append(
            f"{n:<20}{r['model_calls']:>6.2f}{r['input_tokens']:>8.0f}{r['output_tokens']:>8.0f}"
            f"{r['cost']:>8.3f}{r['cost'] / base['cost'] - 1:>+9.0%}"
            f"{r['dev_rate']:>6.0%}{r['test_rate']:>6.0%}{r['rate']:>6.0%}  {r['gate_dev']}/{r['gate_test']}/{r['gate_all']}"
        )
    lines.append("")
    pick = choose_on_dev(rows)
    lines.append(f"Chosen on DEV only (cheapest with dev pass rate >= base): {pick}")
    if pick:
        r = rows[pick]
        lines.append(
            f"  cost {r['cost']:.3f} vs {base['cost']:.3f} per 1k conversations ({r['cost'] / base['cost'] - 1:+.0%}); "
            f"dev {r['dev_rate']:.0%} vs {base['dev_rate']:.0%} (diff {r['diff_dev']:+.0%} [{r['ci_dev'][0]:+.0%}, {r['ci_dev'][1]:+.0%}]); "
            f"TEST {r['test_rate']:.0%} vs {base['test_rate']:.0%} (diff {r['diff_test']:+.0%} [{r['ci_test'][0]:+.0%}, {r['ci_test'][1]:+.0%}])"
        )
        lines.append(
            f"  critical regressions: {r['critical'] or 'none'}; latency p50 {r['latency_p50']:.2f}s (base {base['latency_p50']:.2f}s)"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the response cache on simulated traffic

import random  # noqa: E402

import cachelab as cb  # noqa: E402

from common.cache import ResponseCache  # noqa: E402


def intents() -> dict[str, int]:
    """phrasing -> intent id. Both sides of a SAME pair share an id; every other phrasing is its own intent."""
    ids: dict[str, int] = {}
    for i, p in enumerate(cb.SAME):
        for t in (p.a, p.b):
            ids.setdefault(t, ids.get(p.a, ids.get(p.b, i)))
    for p in cb.NEAR:
        for t in (p.a, p.b):
            ids.setdefault(t, 1000 + len(ids))
    return ids


def surface_variant(text: str, rng: random.Random) -> str:
    """The noise people add that does NOT change the question: case, spacing, punctuation, a polite word."""
    t = text
    if rng.random() < 0.4:
        t = t.upper() if rng.random() < 0.2 else t.capitalize()
    if rng.random() < 0.5:
        t += rng.choice(["?", "??", " ?", "!", "."])
    if rng.random() < 0.2:
        t = "  " + t.replace(" ", "  ", 1)
    return t


def traffic(
    n: int = 400, *, seed: int = 0, surface_noise: float = 0.5, zipf: float = 1.1
) -> list[tuple[str, int]]:
    """SIMULATED traffic: (message, intent) drawn from the labelled phrasings with a popularity skew (a few questions
    are asked all the time), and surface noise on a share of them. The mix is invented: only the mechanism is real."""
    rng = random.Random(seed)
    phrasings = sorted(intents())
    rng.shuffle(phrasings)
    weights = [1 / (rank + 1) ** zipf for rank in range(len(phrasings))]
    ids = intents()
    out = []
    for text in rng.choices(phrasings, weights, k=n):
        out.append(
            (surface_variant(text, rng) if rng.random() < surface_noise else text, ids[text])
        )
    return out


def cache_replay(stream, cache: ResponseCache, miss_cost: float) -> dict:
    """Replay a stream through a cache. A hit is WRONG when the cached answer was written for a different intent."""
    stored_intent: dict[str, int] = {}
    hits = wrong = 0
    for message, intent in stream:
        hit = cache.get(message, scope="tech")
        if hit is not None:
            hits += 1
            wrong += stored_intent.get(hit.matched, -1) != intent
        else:
            stored_intent[message] = intent
            cache.put(message, f"answer-for-{intent}", scope="tech")
    n = len(stream)
    return {
        "requests": n,
        "hit_rate": hits / n,
        "wrong_answers": wrong,
        "wrong_rate": wrong / n,
        "cost": (n - hits) * miss_cost,
        "cost_no_cache": n * miss_cost,
        "saved": hits * miss_cost / (n * miss_cost),
    }


def howto_miss_cost(base_spans_by_case: dict[str, list], card: cl.PriceCard = cl.CHEAP) -> float:
    ids = [i for i in base_spans_by_case if i.startswith("tech-howto")]
    return sum(cl.trace_cost(base_spans_by_case[i], card) for i in ids) / max(len(ids), 1)


def embedding_similarity():
    from common.embed import get_embedder

    emb = get_embedder("local")

    def sim(a: str, b: str) -> float:
        va, vb = emb.embed_documents([a, b])
        return float(va @ vb)

    return sim


# ----------------------------------------------------------------------------- prefix caching, cascades, batch


def prefix_tokens(lean: bool) -> dict[str, int]:
    """Tokens of the part of every model call that never changes (system prompt + tool specs), per specialist, under
    the local model's tokenizer. A provider-side prompt cache can only discount THIS part, and only above a minimum
    length (provider-specific: check yours)."""
    import tempfile

    from support_system.system import LEAN_PROMPTS, PROMPTS, SupportSystem
    from support_system.tools import billing_tools, tech_tools

    from common.local_llm import LocalChat

    chat = LocalChat()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        system = SupportSystem(Path(tmp) / "x.sqlite", provider="ollama")
        for agent, build in (("billing", billing_tools), ("tech", tech_tools)):
            reg = build(system, "c")
            reg = reg.compact() if lean else reg
            prompt = (LEAN_PROMPTS if lean else PROMPTS)[agent]
            empty = chat.count_tokens([{"role": "user", "content": "x"}], None, None)
            out[agent] = (
                chat.count_tokens([{"role": "user", "content": "x"}], reg.specs(), prompt) - empty
            )
    return out


def cascade_projection(base_runs, base_spans_by_case, cheap=cl.CHEAP, strong=cl.STRONG) -> dict:
    """PROJECTION, not a measurement (no strong model was run): answer with the cheap tier, escalate to the strong tier
    when the TRACE shows a defect (Day 4 checks), and assume the strong tier then succeeds. That is an upper bound on
    quality and a fair estimate of cost IF the strong tier needs about the same tokens."""
    flagged = {
        cid
        for cid, spans in base_spans_by_case.items()
        if any(v.code in te.DEFECTS for v in te.check_trace(spans))
    }
    passed = {r.case_id for r in base_runs if r.passed}
    failed = {r.case_id for r in base_runs if not r.passed}
    n = len(base_runs)
    cheap_cost = sum(cl.trace_cost(sp, cheap) for sp in base_spans_by_case.values()) / n * 1000
    extra = sum(cl.trace_cost(base_spans_by_case[c], strong) for c in flagged) / n * 1000
    all_strong = sum(cl.trace_cost(sp, strong) for sp in base_spans_by_case.values()) / n * 1000
    return {
        "escalation_rate": len(flagged) / n,
        "escalated_that_failed": len(flagged & failed),
        "escalated_that_passed": len(flagged & passed),
        "failed_not_escalated": len(failed - flagged),
        "cheap_only_cost": cheap_cost,
        "cascade_cost": cheap_cost + extra,
        "strong_only_cost": all_strong,
        "cheap_pass": len(passed) / n,
        "cascade_pass_upper_bound": (len(passed) + len(flagged & failed)) / n,
    }


def batch_cost(sync_cost: float, discount: float = 0.5) -> float:
    """Batch APIs trade latency (results within hours) for a discount. ASSUMED 50%: check your provider's current terms."""
    return sync_cost * (1 - discount)


def by_kind(names: list[str], out: Path = OUT) -> str:
    runs = {n: load_variant(n, out)[0] for n in names}
    kinds = sorted({r.kind for r in runs[names[0]]})
    lines = ["kind".ljust(18) + "".join(n.rjust(20) for n in names)]
    for k in kinds:
        row = k.ljust(18)
        for n in names:
            rs = [r for r in runs[n] if r.kind == k]
            row += f"{sum(r.passed for r in rs)}/{len(rs)}".rjust(20)
        lines.append(row)
    return "\n".join(lines)


def extras(out: Path = OUT) -> str:
    lines = []
    base_runs, base_spans = load_variant("base", out)
    by_case = per_case(base_spans)
    # 1. prefix caching
    lines.append("== Static prefix per call (system prompt + tool specs), local tokenizer")
    for lean in (False, True):
        lines.append(f"  {'lean' if lean else 'full'}: {prefix_tokens(lean)}")
    u = usage(by_case)
    lines.append(
        f"  average input tokens per CONVERSATION (base): {u['input_tokens']:.0f} over {u['model_calls']:.2f} calls"
    )
    # 2. the cache on simulated traffic
    miss = howto_miss_cost(by_case)
    stream = traffic()
    lines.append(
        f"\n== Response cache on SIMULATED traffic ({len(stream)} requests, {len(set(t for t, _ in stream))} distinct strings)"
    )
    lines.append(
        f"  cost of one how-to conversation on a miss (measured tokens, Haiku card): ${miss * 1000:.3f} per 1k"
    )
    for label, cache in (
        ("no cache", None),
        ("exact (normalised)", ResponseCache(max_entries=10_000)),
        ("exact + fuzzy jaccard 0.6", ResponseCache(max_entries=10_000, fuzzy=True, threshold=0.6)),
        ("exact + fuzzy jaccard 0.8", ResponseCache(max_entries=10_000, fuzzy=True, threshold=0.8)),
    ):
        if cache is None:
            continue
        r = cache_replay(stream, cache, miss)
        lines.append(
            f"  {label:<28} hit rate {r['hit_rate']:.0%}  wrong answers served {r['wrong_answers']} ({r['wrong_rate']:.1%})  saved {r['saved']:.0%}"
        )
    sim = embedding_similarity()
    for t in (0.85, 0.90, 0.95):
        r = cache_replay(
            stream, ResponseCache(max_entries=10_000, fuzzy=True, threshold=t, similarity=sim), miss
        )
        lines.append(
            f"  {'exact + fuzzy embeddings ' + str(t):<28} hit rate {r['hit_rate']:.0%}  wrong answers served {r['wrong_answers']} ({r['wrong_rate']:.1%})  saved {r['saved']:.0%}"
        )
    # 3. threshold curves on labelled pairs
    lines.append(
        f"\n== Similarity threshold on {len(cb.SAME)} paraphrase pairs and {len(cb.NEAR)} near-miss pairs (hit on SAME / wrong answer on NEAR)"
    )
    for name, fn, ths in (
        ("jaccard", cb.jaccard_similarity, [0.5, 0.6, 0.7, 0.8]),
        ("embeddings", sim, [0.85, 0.90, 0.92, 0.95]),
    ):
        for guard_name, g in (
            ("similarity only", None),
            ("+ entity/negation guards", cb.guards_ok),
        ):
            cells = "  ".join(
                f"t={r['threshold']:.2f}: {r['hit_rate_same']:.0%}/{r['wrong_rate_near']:.0%}"
                for r in cb.curve(cb.PAIRS, fn, ths, guards=g)
            )
            lines.append(f"  {name:<11}{guard_name:<26}{cells}")
    # 4. cascade projection and batch
    c = cascade_projection(base_runs, by_case)
    lines.append("\n== Cascade PROJECTION (strong tier assumed to succeed; not run)")
    for k, v in c.items():
        lines.append(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    lines.append(
        f"  nightly 50-case eval, per 1k conversations: sync ${c['cheap_only_cost']:.3f} -> batch ${batch_cost(c['cheap_only_cost']):.3f}"
    )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if argv[:1] == ["extras"]:
        print(extras())
        return
    if argv[:1] == ["kinds"]:
        print(by_kind(argv[1:] or ["base", "lean", "hide", "direct", "lean+hide+direct"]))
        return
    if argv[:1] == ["run"] and len(argv) == 2:
        for n in VARIANTS if argv[1] == "all" else [argv[1]]:
            run_local(n)
    elif argv[:1] == ["report"]:
        print(report())
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
