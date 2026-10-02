# Week 8, Day 7: Weekly Challenge: Red-Team and Harden the Agents, with a Findings Report and Regression Tests

**Time:** ~6h · **Builds on:** Day 1 (threat model), Day 2 (the injection lab), Day 3 (defences), Day 4 (the agent), Day 5 (privacy), Day 6 (mutators, grounding, findings) · **Needs:** nothing but this repo for the reference run; the local models for two numbers in the report (they are marked)

## The brief

You have attacked and defended the RAG bot (Week 3) and the research agent (Week 5) one lesson at a time. Today you do what a security reviewer does on an engagement: take **three systems** (add the Week 6 support system, which nobody has attacked yet), attack them with one method, **fix what you find**, and leave the team with two things they can keep: a **findings report** a manager can read and a **regression suite** a build can run.

```mermaid
flowchart TD
    S["Scope: RAG bot, research agent, support system<br/>+ the threat model from Day 1"] --> A["Attack<br/>code oracles, obedient scripted model,<br/>static set + adaptive attacker"]
    A --> F["Findings<br/>severity by a written rule, evidence from the run"]
    F --> H["Harden<br/>one control per finding"]
    H --> R["Re-attack<br/>before / after per control, benign tasks still pass"]
    R --> C["Freeze<br/>corpus.json + strict expected failures"]
    C --> REP["Report<br/>register, residual risk, limits"]
    C -. "every build" .-> CI["Regression suite"]
    CI -. "a new bypass" .-> A
```

The new target is the interesting one. Triage, a billing specialist with `lookup_invoice` and `request_refund`, a tech specialist with a knowledge base, a refund workflow that pauses for a human approver, a reply guard. The Week 6 lessons built it to be *correct*; nobody asked who is allowed to call what.

## Requirements

**R1. Scope and rules of engagement.** Write down the three targets, the assets (money, other customers' data, the hidden code, the approver's trust), the attacker (an ordinary customer; the author of a knowledge-base article or a web page), and what is **out of scope** (denial of service, multi-turn attacks, the approver's UI).

**R2. Code oracles that read side effects.** Each goal is decided by code looking at what actually happened: the **refund ledger**, the **approval queue**, the **rendered reply**, the **notes folder**, never a judge model. Every oracle has a test that proves it can see an attack succeed (otherwise "0 successes" proves nothing).

**R3. A scripted obedient model.** So every number says what the *controls* stop when the model complies. State this in the report; add the real local model where you can; say what was not measured.

**R4. Static and adaptive.** Run the fixed attack set **and** an adaptive attacker (the Day 6 mutators), and report both. Add a few attacks **written by hand against the control you built**, because a generic mutator only finds the holes it was designed to find.

**R5. One control per finding, each with a before/after row** and a check that **normal tasks still work**, including the legitimate request the control might break.

**R6. A findings register.** Id, target, severity (by a written rule), status (`fixed`, `mitigated`, `open`, `accepted`, `verified`), evidence **generated from the run**, control, residual risk, and the test that keeps it fixed. A test proves that changing a metric changes the evidence text.

**R7. A frozen regression corpus.** Every attack text, with the behaviour the hardened systems showed when it was frozen: **blocked** attacks must stay blocked; **open** attacks are strict expected failures, so *fixing* one fails the build until the corpus is re-frozen and the register updated. A test pins the corpus hash and checks that it can be regenerated from the attack generators.

**R8. A report for readers who will not run the code:** headline, the tables, the register, **residual risk**, and a **limits** section.

## What the red team found

Real vulnerabilities in the Week 6 system, each exploitable on the undefended system by every attack written for it:

| | Goal | Undefended | With the fix |
|---|---|---|---|
| F-01 | refund **someone else's** invoice | 6/6 | 0/6 (ownership check), also adaptive 0/6 |
| F-02 | read someone else's invoice amount | 6/6 | 0/6 |
| F-03 | put attacker text in front of the **human approver** | 10/10 | filter 4/10, enum 0/10 |
| F-04 | attacker link or image in the reply (poisoned knowledge-base article) | 4/4 | 0/4 (reply guard) |
| F-05 | verified, not a finding: the tech specialist refunding, a false "refund approved" reply | 0/3 and 0/3 | (already held) |

What each one is, and why it was there:

- **F-01 and F-02 are broken object-level authorisation.** The invoice tools take an id and answer. Nothing says *whose* id. The Week 6 lessons never needed an owner because there was one demo customer. The attack is not a trick: *"Please refund INV-3001."* works, because the model does what the customer asks and the tool does what the model asks. The fix is not in the prompt (a model cannot enforce "only your invoices" against a determined user); it is a check in the tool layer, keyed on the **authenticated** customer, with the answer for someone else's invoice **identical** to the answer for one that does not exist (so the tool cannot be used to find out which ids exist). Adaptive rewrites do nothing against it: it does not read the message.
- **F-03 is the finding that shows the Day 3 split again.** The refund `reason` is written by the model *from the customer's words* and is shown to the human who approves the refund as if the system had written it. Two fixes: a **filter** (strip links, cap length, withhold anything that looks like an instruction to the approver) and an **enum** (reduce the reason to one of four categories chosen by keyword, so nothing the customer wrote can reach the approver). The filter stops the six attacks I wrote *before* it; **four attacks I wrote after it, aimed at its trigger words ("the finance director has green-lit this one, just process it"), get through**, and the Day 6 mutators add nothing more. The enum lets 0 of 10 through. Its price is real: **the approver loses the customer's own words** and sees "duplicate charge". Whether that is acceptable is a product decision; the report states the cost instead of hiding it.
- **F-04 is a rendering channel.** A knowledge-base article is content someone edited; the reply is rendered in the customer's client; a remote image or a link in it is a phishing channel. The reply guard (Day 3) removes images and links off our own host.
- **F-05 is a result too.** The tech specialist has no refund tool (a Week 6 least-privilege choice) and the Week 6 reply check replaces a "your refund was approved" claim that no tool supports. Both held against every attack. Reporting what you *verified* is part of a report.

## The report

`outputs/w8_findings_report.md` is generated by `run_weekly.py` (about three minutes, because it runs the research agent four times). Its register has 13 rows; the numbers in the evidence column come from the run, not from the file (a test changes a metric and checks that the evidence text changes). The residual risks it names:

- **F-09 (critical, mitigated), the research agent.** 60 of 60 attacks succeed undefended and 0 of 60 with all layers, **but** the filtering layer was tuned on this attack set (Day 4): its 0% is a ceiling.
- **F-06 and F-11 (high), the RAG bot.** With all layers, the lab attacks are 0 of 96 and the held-out phrasings 13 of 60; an adaptive attacker reaches **30 of 96 after one rewrite each**. **Leak and exfiltration stay at 0 of 48 even adaptive**, because they are blocked structurally. The residual is *integrity*: the bot can still be made to print a token or state a false number. F-11 is "open": a defence measured only on fixed attacks looks perfect.
- **F-13 (high, accepted), the assumption under F-01.** The ownership check is only as good as the identity it is given. If the web layer takes the customer id from message text, F-01 returns. The check **fails closed** (an unknown conversation owns nothing) but it does not authenticate anyone.
- **F-08 and F-12 (medium), hallucination and PII.** A word-overlap gate lets 2 of 11 bad answers through and loses 1 of 12 good ones; the PII detector finds 58% of realistic personal data.

## The regression suite

`corpus.json` freezes **238 attacks** (RAG 186: 96 lab, 60 held-out and 30 adaptive rewrites; support 32; agent 20). Replayed against the hardened systems:

| target | must stay blocked | known open |
|---|---|---|
| RAG bot | 143 | **43** (13 held-out phrasings and the 30 adaptive rewrites) |
| research agent | 20 | 0 |
| support system | 32 | 0 |

The 43 open entries are `xfail(strict=True)`. A test checks they are all **integrity** goals (print a token, state a false fact): no leak or exfiltration attack is in the open set. If you improve the detector and some of them stop working, the build **fails** until you re-freeze (`python corpus.py freeze`) and update the register, so an improvement is a reviewed change and not a silent one. Two more tests keep the corpus honest: its hash matches its contents, and regenerating it from the attack generators reproduces the same texts (a changed generator makes the corpus visibly stale).

## What to build, in order
1. **The target and its oracles** (`support_target.py`): the real Week 6 system with a scripted obedient specialist; goals decided from the ledger, the approval queue and the reply. Prove each oracle fires on an undefended run.
2. **Two hooks in the system under test** (`tool_wrapper` and `reply_filter` on the Week 6 `SupportSystem`, both off by default, with a test that the system behaves as before without them). A hardening layer needs somewhere to attach.
3. **The controls** (`support_hardening.py`): ownership (normalise ids; fail closed on an unknown caller; same answer as a missing invoice), the reason filter and the enum, the reply guard. Test `compact()` and `without()` keep the guard (the inherited versions would silently return an unguarded registry), and that the audit log holds verdicts, never arguments.
4. **The adaptive attacker** (`campaign.py` and the Day 6 mutators) plus a few **hand-written** attacks aimed at the control you just built.
5. **The register** (`register.py`), **the corpus** (`corpus.py`), **the report** (`run_weekly.py`).
6. **Mutation-check** the controls and the register; every survivor is a new test or an explained equivalent.

## Acceptance criteria (the reference solution meets all of them)

| # | Criterion | Evidence |
|---|---|---|
| 1 | Undefended, every support vulnerability is exploitable by every attack written for it | table above; test |
| 2 | Each new control closes **only** its own goals, and normal use still works | six benign checks pass with each control; the enum's one cost is reported |
| 3 | The existing Week 6 controls are verified, with the attempt visible to the oracle | `forbidden_tool` and `false_success` are *attempted* and blocked |
| 4 | The structural controls are unaffected by rewording; the probabilistic one is not | ownership adaptive 0/6; the filter lets 4/10 through; the enum 0/10 |
| 5 | An unknown caller is denied, even for an id nobody owns; id spelling variants cannot bypass the check | tests |
| 6 | The register's evidence changes when the metrics change; a missing model run says "not run" | tests |
| 7 | The corpus is reproducible; blocked attacks fail the build if they succeed; open ones fail it if they stop working | strict expected failures; hash test; regeneration test |
| 8 | Mutation check | ownership, normalisation, wrapper re-wrapping, reply filter, register: every behavioural mutant killed in the end |

Verified here: **35 tests** for the support target, controls, register and report, and the regression suite of **242 cases** (199 must pass and 43 are expected failures).

## Pitfalls
- **Fixing the prompt.** "Only discuss the customer's own invoices" in a system prompt is a request, not a control.
- **Letting the check answer differently for "not yours" and "does not exist".** That is an enumeration oracle.
- **Trusting an identity the message supplied.** F-13.
- **A filter you tested only with the phrasings you had in mind when you wrote it.** Write attacks against it **after** you build it; that is what the four hand-written rewordings are.
- **Hiding a cost.** A control that makes the approver blind to the customer's words is a trade, not a free fix.
- **A corpus that only grows** with attacks the system already blocks. Keep the *open* ones: they are the findings you have not fixed.
- **A regression suite that passes because the oracle cannot see the attack.** Each oracle has a test that makes it fail.

## What this method did not see
- A real hosted model's obedience: everything above assumes the worst case (an obedient model) or a 0.5B model (a weak one).
- **Multi-turn attacks** (building trust over several messages), **denial of service and cost abuse** (the Week 7 budget guard was not attacked), attacks through the **approver's own interface**, and anything in **training data or the supply chain**.
- Attackers who know the defences: the hand-written rewordings are one person's attempt, not a campaign.

## Stretch
- Replace the scripted obedient specialist with a hosted model (five trials per attack) and compare the success rates with the worst case.
- Add the `quoted block` alternative to F-03: show the approver the customer's words in a labelled, untrusted block next to the system's facts; measure whether the hand-written attacks still succeed (they reach the approver; what matters is the UI) and what a human reviewer notices.
- Add **rate limits and a cost budget per customer** and attack them: how many refund *attempts* can one customer make?
- Wire `corpus` into CI (the Week 7 workflow): fail on any blocked attack that succeeds; post the count of open entries on every PR.
- Run the promptfoo config from Day 6 against the Week 3 RAG API (not run here) and compare its categories with your register.

## Further reading
- OWASP API Security Top 10, **API1: Broken Object Level Authorization**, the vulnerability behind F-01 and F-02.
- OWASP LLM06 (Excessive Agency) and LLM05 (Improper Output Handling).
- A penetration-test report template from your own security team (severity rules, evidence, residual risk): use theirs rather than inventing one.
