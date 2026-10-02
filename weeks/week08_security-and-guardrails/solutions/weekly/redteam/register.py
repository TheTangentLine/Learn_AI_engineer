"""The findings register: one row per problem, with the evidence that came out of the runs (never typed in by hand), the control, the status and
the test that keeps it fixed. ``build(metrics)`` takes the numbers from ``run_weekly.collect()`` so the register cannot drift from the runs.

Statuses:   fixed        a control removes it and a regression test enforces that
            mitigated    reduced, with a named residual risk
            open         known and not yet addressed (the corpus holds an expected-failure test for it)
            accepted     a design limit with an owner's decision (here: a placeholder owner, because this is a lab)
            verified     not a problem: an existing control was tested and holds
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEVERITIES = ["low", "medium", "high", "critical"]


@dataclass
class Finding:
    id: str
    target: str  # rag | agent | support | privacy | platform
    title: str
    severity: str
    status: str
    evidence: str
    control: str
    residual: str = ""
    tests: list[str] = field(default_factory=list)


def frac(pair) -> str:
    return "not run" if pair is None else f"{pair[0]}/{pair[1]}"


def build(m: dict) -> list[Finding]:
    s, rag, ag, gr = m["support"], m["rag"], m["agent"], m.get("grounding")
    sup = lambda cfg, goal: frac(s["static"][cfg][goal])  # noqa: E731
    f: list[Finding] = []

    f += [
        Finding(
            "F-01",
            "support",
            "Any customer can have any invoice refunded (the invoice tools never ask whose invoice it is)",
            "high",
            "fixed",
            f"unowned_refund {sup('none', 'unowned_refund')} succeed with no new control; {sup('ownership', 'unowned_refund')} with ownership; adaptive: {s['adaptive_ownership']['adaptive']}/{s['adaptive_ownership']['attacks']}",
            "ownership check in the tool layer, keyed on the authenticated customer; other customers' invoices answer like a missing invoice",
            "assumes the web layer supplies a real authenticated customer id (F-13)",
            [
                "test_weekly.py::test_ownership_closes_refund_and_disclosure",
                "corpus: support:unowned_refund/*",
            ],
        ),
        Finding(
            "F-02",
            "support",
            "Any customer can read any invoice's amount and status",
            "medium",
            "fixed",
            f"unowned_disclosure {sup('none', 'unowned_disclosure')} -> {sup('ownership', 'unowned_disclosure')} with ownership",
            "the same ownership check (and the answer cannot be used to enumerate which ids exist)",
            "",
            ["test_weekly.py::test_ownership_closes_refund_and_disclosure"],
        ),
        Finding(
            "F-03",
            "support",
            "A human approver reads text the customer wrote as if the system had written it",
            "high",
            "fixed",
            f"approver_injection {sup('none', 'approver_injection')} undefended. The filter still lets {s['adaptive_filter']['static']}/{s['adaptive_filter']['attacks']} through as written (the human-written rewordings that avoid its trigger words) and {s['adaptive_filter']['adaptive']}/{s['adaptive_filter']['attacks']} after the Day 6 mutators; enum: {s['adaptive_enum']['adaptive']}/{s['adaptive_enum']['attacks']}",
            "reason reduced to one of a few categories (no customer text reaches the approver)",
            "the approver loses the customer's own words; show them as a clearly quoted, untrusted block if the business needs them",
            ["test_weekly.py::test_the_reason_filter_is_probabilistic_and_the_enum_is_structural"],
        ),
        Finding(
            "F-04",
            "support",
            "A knowledge-base article can put an attacker's link or image into a customer-visible reply",
            "medium",
            "fixed",
            f"link_injection {sup('none', 'link_injection')} -> {sup('reply_guard', 'link_injection')} with the reply guard",
            "reply guard (no remote images, links only to our own host)",
            "",
            ["test_weekly.py::test_the_reply_guard_removes_attacker_links_and_images"],
        ),
        Finding(
            "F-05",
            "support",
            "Verified, not a problem: the tech specialist cannot refund; a false 'refund approved' reply is caught",
            "high",
            "verified",
            f"forbidden_tool {sup('none', 'forbidden_tool')} and false_success {sup('none', 'false_success')} with NO new control (least-privilege tools and the Week 6 reply check)",
            "Week 6: per-specialist tools; check_reply against what the tools returned",
            "",
            ["test_weekly.py::test_existing_week6_controls_hold"],
        ),
    ]

    rag_after = rag["all layers"]
    f += [
        Finding(
            "F-06",
            "rag",
            "Instructions in a question or a retrieved document hijack the answer",
            "high",
            "mitigated",
            f"obedient model: {frac(rag['none']['lab'])} lab attacks succeed undefended; real Qwen {frac(rag.get('qwen_none_lab'))}. All layers: lab {frac(rag_after['lab'])}, held-out {frac(rag_after['heldout'])}, adaptive {frac(rag_after['adaptive'])}",
            "input guard, document filter, datamarking, secret isolation, output guard (Day 3)",
            "integrity attacks (print a token, state a false fact) still get through paraphrases: the probabilistic layers recognise phrasings",
            ["corpus: rag:* (blocked) and rag-adaptive:* (expected failures)"],
        ),
        Finding(
            "F-07",
            "rag",
            "The hidden internal code can be leaked, or sent out in an image URL",
            "critical",
            "fixed",
            f"leak + exfil goals succeed {frac(rag['none']['leak_exfil'])} undefended; {frac(rag_after['leak_exfil'])} with all layers INCLUDING the adaptive attacker",
            "the model never sees the code (isolation) and the output guard removes images and blocks the code (structural: independent of the model and of phrasing)",
            "",
            ["test_day3.py structural rows", "corpus: rag:*/leak_secret, rag:*/exfil_url"],
        ),
    ]
    if gr:
        f.append(
            Finding(
                "F-08",
                "rag",
                "The model invents answers to questions the documents do not answer, and repeats false premises",
                "medium",
                "mitigated",
                f"real Qwen: {gr['unanswerable_wrong']}/12 unanswerable answered, {gr['premise_wrong']}/6 false premises accepted; a word-overlap gate at 0.9 lets {gr['lexical_bad_through']}/{gr['bad']} bad answers through and loses {gr['lexical_good_lost']}/{gr['good']} good ones; NLI catches {gr['swaps_caught']}/{gr['swaps']} number swaps",
                "grounding gates (word overlap for invented content, NLI entailment for swaps)",
                "answers that are grounded but do not address the question pass every gate",
                ["test_day6.py::test_real_model_pinned_gate_signals"],
            )
        )
    else:
        f.append(
            Finding(
                "F-08",
                "rag",
                "Hallucination and false premises",
                "medium",
                "mitigated",
                "not run (needs the local models)",
                "grounding gates",
                "",
                [],
            )
        )
    f += [
        Finding(
            "F-09",
            "agent",
            "A poisoned search result makes the research agent write attacker text to disk, send the code out in the report, or report a false fact",
            "critical",
            "mitigated",
            f"{frac(ag['none'])} attacks succeed undefended; {frac(ag['all layers'])} with all layers (lab set: the detector layer was tuned on it)",
            "capabilities, tool policy, confirmation, tool-result filtering, output guard (Day 4)",
            "the filtering layer's 0% is a ceiling (Day 3: held-out recall 77%)",
            ["corpus: agent:*"],
        ),
        Finding(
            "F-10",
            "agent",
            "A tool policy alone leaves the report (the output channel) open",
            "high",
            "fixed",
            f"capabilities + policy: {frac(ag['caps_policy'])}; adding the output guard: {frac(ag['caps_policy_output'])}; the {frac(ag['caps_policy_output'])} left are integrity goals (token, false fact)",
            "output guard on the final report",
            "integrity attacks need the filtering layer or a claim checker (F-06, F-08)",
            ["test_day4.py"],
        ),
        Finding(
            "F-11",
            "platform",
            "A defence measured only on fixed attacks looks perfect: rewording alone defeats the detector",
            "high",
            "open",
            f"static 0/96 -> adaptive {frac(rag_after['adaptive'])} against the same stack after ONE rewrite each; the structural layers are unaffected",
            "report adaptive numbers beside static ones; keep the rewrites as permanent tests",
            "a better detector or a claim checker; until then the integrity risk is real",
            ["corpus: rag-adaptive:* (expected failures, strict)"],
        ),
        Finding(
            "F-12",
            "privacy",
            "The PII detector finds 58% of personal data on realistic text (names 0 of 5)",
            "medium",
            "mitigated",
            "generated set 160/160, realistic set 19/33 [41%, 73%] (Day 5)",
            "validators, pseudonymisation before the model, redaction in logs, retention and erasure",
            "names, spaced cards, obfuscated emails: add an NER model and measure on your own data",
            ["test_day5.py"],
        ),
        Finding(
            "F-13",
            "support",
            "Assumption: the customer id comes from an authenticated session",
            "high",
            "accepted",
            "the ownership check fails closed (an unknown conversation owns nothing), but it trusts the identity it is given",
            "web-layer authentication outside this lab",
            "if identity is taken from message text, F-01 returns",
            ["test_weekly.py::test_an_unknown_conversation_owns_nothing"],
        ),
    ]
    return f


def residual_risks(findings: list[Finding]) -> list[Finding]:
    return sorted(
        (x for x in findings if x.status in ("open", "mitigated", "accepted")),
        key=lambda x: (-SEVERITIES.index(x.severity), x.id),
    )


def markdown(findings: list[Finding]) -> str:
    rows = [
        "| id | target | severity | status | finding | evidence | control | residual |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in findings:
        rows.append(
            f"| {x.id} | {x.target} | {x.severity} | **{x.status}** | {x.title} | {x.evidence} | {x.control} | {x.residual or '-'} |"
        )
    return "\n".join(rows)
