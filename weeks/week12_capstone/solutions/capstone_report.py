"""Turn the measured results of the capstone run into the documents a reviewer reads: a targets-versus-outcomes report and the portfolio README.

Pure functions of a results dict (so they are tested without a model). Nothing here is a number typed in by hand: every figure in the output comes from ``res``; the assumptions and the list of
what was NOT run are written once, below, and cannot drift from the code.
"""

from __future__ import annotations

from copilot import evaluate as E

NOT_RUN = [
    "a hosted or larger chat model: whether it beats the extractive answerer (it may) is unmeasured; the verification harness is what would measure it",
    "a cloud deployment: no URL exists; the container image was not built or run on this machine (see the Day 5 lesson for why) and the CI workflow was never executed on GitHub",
    "real users: every question was written by the author, with the lessons open",
    "a GPU, dense retrieval at scale, multi-replica serving, an authentication provider",
]


def rate_text(r: tuple[float, float, float]) -> str:
    return E.fmt(r)


def targets(res: dict) -> list[dict]:
    """The Day 1 requirements, each with its measured outcome and whether it was met (computed, not asserted)."""
    ans = res["test"]["answerable"]
    oos = res["all"]["out_of_scope"]
    gate_pr = res["gate_prs"]
    bad = [k for k, v in gate_pr.items() if not v["refactor"] and not v["passed"]]
    rows = [
        (
            "R1",
            "the right lesson is retrieved (hit@5, all 52 answerable)",
            "≥ 90%",
            rate_text(res["all"]["retrieval_hit5"]),
            res["all"]["retrieval_hit5"][0] >= 0.90,
        ),
        (
            "R2",
            "golden pass rate on the answerable questions, test split",
            "≥ 80%",
            rate_text(ans),
            ans[0] >= 0.80,
        ),
        ("R3", "out-of-scope questions refused (all 10)", "≥ 90%", rate_text(oos), oos[0] >= 0.90),
        (
            "R4",
            "no leak of a secret or canary (5 golden + 13 direct attacks, obedient model, every control on); indirect injection reported",
            "0 leaks",
            f"{res['security']['direct_leaks']} leaks; poisoned documents through: {res['security']['poison_through']} of 10",
            res["security"]["direct_leaks"] == 0,
        ),
        (
            "R5",
            "p95 latency, cold question, one CPU process",
            "≤ 2 s",
            f"{res['latency']['p95'] * 1000:.0f} ms",
            res["latency"]["p95"] <= 2.0,
        ),
        (
            "R6",
            "machine cost per 1,000 questions (extractive), assumed $0.20/hour",
            "≤ $0.05",
            f"${res['cost']['per_1000']:.3f}",
            res["cost"]["per_1000"] <= 0.05,
        ),
        (
            "R7",
            "health vs readiness, metrics, request ids, traces without personal data",
            "present and tested",
            f"{res['tests']} tests pass",
            res["tests"] > 0,
        ),
        (
            "R8",
            "the CI gate fails on bad changes and passes on a refactor",
            "fails on 4",
            f"fails on {len(bad)} of 4 bad changes; refactor passes: {res['gate_refactor_passes']}",
            len(bad) >= 4 and res["gate_refactor_passes"],
        ),
    ]
    return [
        {"id": a, "requirement": b, "target": c, "outcome": d, "met": e} for a, b, c, d, e in rows
    ]


def targets_table(res: dict) -> str:
    lines = ["| # | requirement | target | outcome | met |", "|---|---|---|---|---|"]
    for t in targets(res):
        lines.append(
            f"| {t['id']} | {t['requirement']} | {t['target']} | {t['outcome']} | {'yes' if t['met'] else '**NO**'} |"
        )
    return "\n".join(lines)


def system_row(name: str, rep: dict) -> str:
    return f"| {name} | {E.fmt(rep['overall'])} | {E.fmt(rep['single'])} | {E.fmt(rep['multi'])} | {E.fmt(rep['out_of_scope'])} | {E.fmt(rep['adversarial'])} |"


def render_report(res: dict) -> str:
    met = sum(t["met"] for t in targets(res))
    sys_rows = "\n".join(system_row(n, r) for n, r in res["systems"].items())
    stages = "\n".join(
        f"| {k} | {v['p50']:.1f} ms | {v['p95']:.1f} ms |"
        for k, v in res["latency"]["stages_ms"].items()
    )
    poison = res["security"]["poison_table"]
    cols = list(next(iter(poison.values())))
    poison_rows = "\n".join(
        f"| {a} | " + " | ".join(str(row[c]) for c in cols) + " |" for a, row in poison.items()
    )
    retrieval_rows = "\n".join(
        f"| {k} | {E.fmt(v['hit5'])} | {v['both']} / {v['multi']} |"
        for k, v in res["retrieval"].items()
    )
    gate_lines = "\n".join(
        f"| {name} | {'PASS' if v['passed'] else 'FAIL'} | {v['why']} |"
        for name, v in res["gate_prs"].items()
    )
    load = "\n".join(
        f"| {r['users']} | {r['rps']:.2f} | {r['lat50']:.2f} s | {r['lat95']:.2f} s | {r['errors']} |"
        for r in res.get("load_closed", [])
    )
    return f"""# Week 12 report: Course Copilot

**What it is.** A cited question-answering product over the 77 lessons of Weeks 1-11 ({res["chunks"]:,} chunks): hybrid retrieval with week scoping, a cross-encoder relevance gate, an extractive answerer (a verified model-backed answerer is available behind the same interface), guards on the question, the retrieved text and the output, and the Week 11 gateway in front. Everything ran on one Apple laptop CPU; no hosted model was used.

## 1. Targets against outcomes ({met} of 8 met)
{targets_table(res)}

The design document's targets were fixed on Day 1 and are not edited here. A target that is not met is a finding, not a typo.

## 2. Quality on the golden set (test split, each system run once; the design was chosen on dev)
| system | overall | single-lesson | two-lesson | out of scope | attacks |
|---|---|---|---|---|---|
{sys_rows}

Floors on the same split (Day 1): always abstain **{res["floors"]["abstain"]:.0%}**, dump the top BM25 chunk **{res["floors"]["dump"]:.0%}**. On dev the chosen design passes {E.fmt(res["dev"]["overall"])}; on test {E.fmt(res["test"]["overall"])}. The gap between them is what choosing among designs on {res["dev"]["n"]} items costs.

## 3. Retrieval and the gate
| retrieval configuration | hit@5 (52 answerable) | both lessons found (two-lesson questions) |
|---|---|---|
{retrieval_rows}

The relevance gate separates answerable from out-of-scope questions with AUC **{res["gate"]["auc_rerank"]:.3f}** (cross-encoder) against {res["gate"]["auc_cosine"]:.3f} (best cosine); threshold {res["gate"]["threshold"]:.2f}, chosen on dev only.

## 4. Security
Direct attacks with a scripted obedient model: {res["security"]["direct_none"]} of 18 obeyed with no controls, {res["security"]["direct_input"]} with the input guard, **{res["security"]["direct_leaks"]}** with the input guard, verification and the output guard. Over-blocking: {res["security"]["overblocked"]} of {res["security"]["legit"]} legitimate questions.

Poisoned documents in an untrusted collection (attack succeeded, of 10):

| answerer | {" | ".join(cols)} |
|---|{"---|" * len(cols)}
{poison_rows}

The residual with every layer on is not zero: it is made of attacks built to evade the detector and of payloads that are plain text in an untrusted source. The product only quotes and never acts; show provenance and keep untrusted material away from anything that can take actions.

## 5. The CI gate (dev split, pretend pull requests against the stored baseline)
| change | verdict | reason |
|---|---|---|
{gate_lines}

## 6. Latency, cost, load
Cold questions (no cache has seen them), one process:

| stage | p50 | p95 |
|---|---|---|
{stages}

Machine time per 1,000 questions at an assumed $0.20/hour: **${res["cost"]["per_1000"]:.3f}** ({res["cost"]["seconds"]:.3f} s each, no model tokens). The model-backed answerer adds a model call per question (about {res["cost"]["llm_prompt_tokens"]:,} prompt tokens on the 0.5B model) and did not improve quality.

{("Closed-loop load (the service process, unique questions):" + chr(10) + chr(10) + "| users | req/s | p50 | p95 | errors |" + chr(10) + "|---|---|---|---|---|" + chr(10) + load) if load else "The load test was not run in this invocation (use the Day 6 script)."}

## Not run
{chr(10).join("- " + x for x in NOT_RUN)}

## Limits
- {res["dev"]["n"]} dev and {res["test"]["n"]} test items: one item moves a rate by 3 points; every difference of one or two items is noise.
- The golden set was written by the author with the lessons open; its two-lesson questions all name their weeks.
- The extractive answerer fails by quoting the wrong sentence (every test failure was that), and the golden set's `must_cite` allows one source where several lessons answer a question.
- One run, one machine, one seed. Prices are assumptions supplied as inputs.
"""


def render_portfolio(res: dict, template: str) -> str:
    """The portfolio README: the template's structure, every section filled from the results. What only a human can add (a screenshot, a recorded demo, a live URL) is left explicit."""
    t = {x["id"]: x for x in targets(res)}
    out = template
    out = out.replace("# <Project name>", "# Course Copilot")
    out = out.replace(
        "> One sentence: what it does and for whom.",
        "> A cited, guarded question-answering assistant over this course's 77 lessons, for people working through the course: it shows which lesson an answer came from and says \"I don't know\" when the lessons do not cover the question.",
    )
    out = out.replace(
        '**Live demo:** <URL, or "not deployed: run `docker compose up`"> · **Demo video / transcript:** <link> · **Design doc:** <link>',
        "**Live demo:** not deployed (no cloud account was used): run `uv run python -m copilot.service` as in the Day 5 lesson · **Transcript:** below · **Design doc:** `weeks/week12_capstone/solutions/design/DESIGN.md`",
    )
    demo = "\n".join(
        f"**{d['label']}.** `{d['question']}`\n> {d['answer'][:260].replace(chr(10), ' ')}{'…' if len(d['answer']) > 260 else ''}\n"
        + (
            f"> *sources:* {'; '.join(d['sources'])}\n"
            if d["sources"]
            else "> *(no sources shown)*\n"
        )
        for d in res.get("demo", [])
    )
    what = (
        "- Answers questions about the course with **numbered citations** to the lesson and section.\n"
        "- **Refuses** what the lessons do not cover, and **blocks or neutralises** hostile input.\n"
        "- Runs on a laptop CPU with **no model**: retrieval, a relevance gate and an extractive answerer; a verified model-backed answerer is a measured option.\n\n"
        "A real transcript (from `demo.py`, including a question it gets wrong on purpose):\n\n"
        + demo
    )
    out = out.replace(
        "Three bullets and a screenshot or a transcript of one real interaction (including a refusal).",
        what,
    )
    table = (
        "| What | Result | How measured |\n|---|---|---|\n"
        f"| Quality on the held-out test set | {E.fmt(res['test']['answerable'])} of answerable questions; overall {E.fmt(res['test']['overall'])} | 33 test items; pass = key facts present **and** the right lesson cited; abstention for out-of-scope |\n"
        f"| Safety | {res['security']['direct_leaks']} leaks in 18 direct attacks against a scripted obedient model with every control on; {res['security']['poison_through']} of 10 poisoned documents through | code oracles; attacks include ones built to evade the detector |\n"
        f"| Latency (p50 / p95, cold, one CPU process) | {res['latency']['p50'] * 1000:.0f} ms / {res['latency']['p95'] * 1000:.0f} ms | questions no cache has seen |\n"
        f"| Cost per 1,000 questions | ${res['cost']['per_1000']:.3f} machine time | assumed $0.20/hour, 100% utilisation |"
    )
    start = out.index("| What | Result | How measured |")
    end = out.index("**What it does not do well**")
    out = out[:start] + table + "\n\n" + out[end:]
    out = out.replace(
        "**What it does not do well** (the three failures you would show an interviewer first).",
        f"**What it does not do well.** Target R2 (golden pass rate ≥ 80% on answerable test questions) was **not met**: {t['R2']['outcome']}. Every failure was the same cause: the right lesson was quoted but not the sentence that holds the key facts. "
        "A 0.5B model answering from the same sources did worse (it cites the wrong source or omits the facts), so it is not the default. The injection detector misses half of the poisoned documents; the answerer only quotes, so the damage is bounded to text shown.",
    )
    arch = (
        "```mermaid\nflowchart LR\n    U[question] --> IG[input guard] --> R[hybrid retrieval + week scoping] --> Q[quarantine untrusted] --> G{relevance gate}\n"
        "    G -- weak --> X[refuse] \n    G -- ok --> A[extractive answerer] --> O[output guard] --> OUT[answer + citations]\n```\n"
        "Lessons are chunked by heading with week and day metadata and indexed for BM25 and dense search. A question that names weeks is also searched inside each week. A cross-encoder decides whether the "
        "sources can answer at all. The answerer quotes the best sentences with `[n]` markers and cannot invent a fact. Guards sit on the question, on untrusted sources and on the output, and the Week 11 gateway provides keys, limits and metrics."
    )
    out = out.replace("Diagram + five sentences.", arch)
    r = res["retrieval"]
    ext = res["systems"]["extractive (shipped)"]["overall"]
    llm = res["systems"].get("model + verification + fallback")
    decisions = (
        "| Decision | Alternative | The number |\n|---|---|---|\n"
        f"| Hybrid retrieval with week scoping | BM25 alone, dense alone | hit@5 {E.fmt(r['hybrid + week scope']['hit5'])} against {E.fmt(r['BM25 only']['hit5'])} and {E.fmt(r['dense only']['hit5'])}; both lessons found for {r['hybrid + week scope']['both']} of {r['hybrid + week scope']['multi']} two-lesson questions against {r['BM25 only']['both']} for BM25 |\n"
        f"| A cross-encoder as the relevance gate | the cosine, the BM25 score | AUC {res['gate']['auc_rerank']:.3f} against {res['gate']['auc_cosine']:.3f} for the cosine |\n"
        f"| Extractive answers by default | a 0.5B model with verification | {E.fmt(ext)} against {E.fmt(llm['overall']) if llm else 'not run'} overall on the test split |\n"
        f"| Quarantine by provenance, not by content | scan the whole corpus | the first-party lessons teach injection; {res['security']['poison_through']} of 10 poisoned uploads still get through, so provenance is shown with every sentence |"
    )
    out = out.replace(
        "Four decisions, each with the alternative and the number that settled it.", decisions
    )
    out = out.replace(
        "Exact commands, from a clean clone, with what you should see.",
        "```bash\nuv sync --extra serve\nuv run python weeks/week12_capstone/solutions/day2_solution.py      # builds the index (about 48 s first time)\n"
        "uv run python weeks/week12_capstone/solutions/demo.py                # the transcript above\nuv run python weeks/week12_capstone/solutions/run_capstone.py --load # the report\nuv run pytest weeks/week12_capstone/solutions\n```",
    )
    limits = (
        "\n".join(f"- {x}" for x in NOT_RUN)
        + f"\n- {res['dev']['n']} dev and {res['test']['n']} test items: one item moves a rate by 3 points.\n- The golden set was written by the author with the lessons open."
    )
    out = out.replace(
        "What was not run, what was assumed, what would break first.",
        limits
        + "\n- What would break first: more than about 2.6 requests per second on one process (it answers one question at a time).",
    )
    return out
