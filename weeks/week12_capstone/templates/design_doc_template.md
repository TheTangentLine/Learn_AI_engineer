# Design doc: <product name>

**Author:** · **Status:** draft / reviewed / accepted · **Date:** · **Reviewers:**

## 1. Problem and users
Who has the problem, what they do today, what "better" looks like for them. One paragraph, then the two or three questions the product must answer well (real examples, not categories).

## 2. Goals and non-goals
- **Goals** (each testable): …
- **Non-goals** (what you will *not* do, so nobody assumes you will): …

## 3. Requirements
| # | Requirement | How it is measured | Target |
|---|---|---|---|
| R1 | | | |
(Functional and non-functional: quality, safety, latency, cost, availability. A requirement without a measurement is a wish.)

## 4. Architecture
A diagram (Mermaid) and one paragraph per component: what it does, what it depends on, what happens when it fails.

## 5. Data
What the corpus or dataset is, where it comes from, who may see it, how it is refreshed, what is deliberately excluded and why.

## 6. Evaluation plan
- **Golden set:** how many items, of which kinds, who wrote them, how they are checked, the dev/test split.
- **Metrics and thresholds:** the pass rule for an item, the number that gates a release.
- **Baselines to beat:** the simplest thing that could work, measured first.
- **What would change your mind:** the result that makes you drop a design choice.

## 7. Decisions and alternatives
| Decision | Alternatives considered | Why this one | How we will find out if it was wrong |
|---|---|---|---|

## 8. Risks
| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
(At least five. Include the security risks and the "the model is wrong" risks.)

## 9. Security and privacy
Threat model in a paragraph: assets, who attacks what through which channel, which controls, which residual risks you accept.

## 10. Cost and latency budget
Per-request budget by stage, the price assumptions (as inputs, dated), the volume at which the design stops being the right one.

## 11. Rollout and operations
How it ships, how you know it is healthy (metrics, alerts), how you roll back, what the on-call person needs.

## 12. Open questions
What you do not know yet, and the experiment that would tell you.
