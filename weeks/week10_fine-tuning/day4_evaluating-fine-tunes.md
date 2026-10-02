# Week 10, Day 4: Evaluating a Fine-Tune: Base Against Tuned, Overfitting, Forgetting

**Time:** ~5h (about 20 minutes of that is generation on a CPU) · **Needs:** CPU only; the Day 3 checkpoints · **Run it:** `uv run python weeks/week10_fine-tuning/solutions/day4_solution.py`

Training loss going down means nothing about whether the model is *better*. A fine-tune has three ways to fool you: it can look great on data that resembles its training set and fail on real inputs; it can improve the task while quietly breaking other abilities; and it can look better than the baseline because you compared them on twelve examples. Today you build the evaluation that catches all three, apply it to the Day 3 checkpoints, and write down what the numbers do and do not say.

## Learning objectives
- Evaluate on **two kinds of held-out data** (the easy, in-distribution one and the hard, unseen-phrasing one) and read the **generalisation gap**.
- Compare systems with **confidence intervals** and a **paired test** on the same inputs, and say when a difference is noise.
- Detect **overfitting across epochs** and **catastrophic forgetting** (with unrelated probes, a language-modelling loss and an over-generalisation check).
- Do **error analysis**: which fields fail, and which *kind* of wrong.

---

## 1. What is being compared

All systems are SmolLM2-135M-Instruct with greedy decoding, on the **same emails**, scored by the same code:

| system | prompt | what it is |
|---|---|---|
| base, 3-shot | schema description + 3 worked examples (593 tokens) | the best prompt from Day 1 |
| LoRA, epoch 1 / 2 / 3 | one instruction line (about 85 tokens) | the Day 3 checkpoints, adapters merged for speed |
| a hosted frontier model | n/a | **NOT RUN** (no API key was used): the column exists so a report has a place for it |

Two evaluation sets, with different jobs:

| set | n | what it measures | how it was made |
|---|---|---|---|
| **hand-written** | 38 | generalisation to phrasing the training data never produced | the 8 Week 2 emails plus 30 more, written by me, with answer keys, before any model saw them |
| **synthetic held-out** | 100 | learning of the task as the generator expresses it | the same generator as training, different seeds, filtered the same way |

The **gap between them is the most important number of the day.** A model can score 95%+ on the synthetic set and far less on the hand-written one: that is not overfitting in the sense of memorising; it is the model having learned the *templates* rather than the *task*.

## 2. Reading rates and differences honestly

- **Rates get a Wilson interval.** 21 correct of 38 is 55%, and the 95% interval is roughly **[39%, 70%]**. Two systems whose intervals overlap may still differ, but you cannot tell from the intervals alone.
- **Compare systems on the same inputs with a paired test.** For each email, did system A do better, worse or the same as B? The paired bootstrap resamples *emails* and looks at the distribution of the mean difference: easy and hard emails affect both systems alike, so the pairing cancels the difficulty and the test is much more sensitive than comparing two separate intervals. We report the difference, its 95% interval, a p-value, and the counts *better on / worse on / tied on*.
- **Exact match** (all 8 fields right) is the strict metric; **field accuracy** (the share of the 8 fields right) gives partial credit and separates "almost right" from "useless". Report both.
- **38 emails is small.** A system that is right on 24 against 21 is not demonstrably better. The intervals say so; believe them.

{{RESULTS}}

## 3. Overfitting across epochs

A fine-tune that trains for too long on a narrow dataset *memorises its surface*. The signals, in the order they appear: the training loss keeps falling while the **held-out loss stops falling or rises**; the **synthetic** score keeps rising while the **hand-written** score stalls or falls (the model is getting better at the templates); and the model becomes **brittle** (small changes of phrasing change the answer). Checkpoints per epoch let you see it: pick the epoch from the *hand-written* curve (not the training loss), and say how many emails separate the epochs.

## 4. Forgetting

Fine-tuning on one task moves the weights; some of what the model could do before is overwritten. Three measurements, none of which the training loss shows:

1. **Unrelated probes** (36 prompts: facts, arithmetic, one-line format instructions, small JSON tasks other than orders), asked with the model's ordinary chat prompt and graded by code. The base model is the reference: a *drop* is forgetting.
2. **Over-generalisation:** the share of replies to unrelated prompts that are an *order object*. A fine-tune that answers "What is the capital of France?" with `{"is_order":false,...}` has learned the format too well.
3. **Language-modelling loss on ordinary prose**, with no chat template: has the model's grasp of general text degraded?

An extraction service does not need the model to answer trivia, so some forgetting may be acceptable; the point is to **know**, and to decide on purpose. If it matters, mix some general data into the fine-tuning set, lower the rank, use fewer epochs, or keep the adapters separate and switch them per task (the reason adapters exist).

## 5. Error analysis

After the number, the failures. For each wrong answer ask *what kind* (not JSON; valid JSON that is not a valid order; a valid order with wrong fields) and *which field*. Cluster them. The clusters tell you what to do next, and **what to do next must be to change the training data or the method, never to tune on the evaluation set**: if you fix the generator for the *specific phrasing* of a failing hand-written email, that email is no longer unseen and the score is no longer an honest estimate. Fix the *category* (say, "dates written as '2nd of January'") with fresh data, and re-evaluate on a **new** hand-written set (the stretch exercise).

## 6. Pitfalls
- **Selecting the epoch on the test set.** The hand-written set was used to *choose* between three checkpoints here; with so few emails that choice is itself a (small) source of optimism. Say so.
- **Calling a gap of two emails an effect.**
- **Only the easy test set.**
- **Reporting the average of probes** and missing that one kind collapsed.
- **Comparing against a baseline that was never tuned**, or against one that was handicapped.
- **Forgetting that greedy decoding is one sample** of the model's behaviour; a sampled model has run-to-run variance.
- **Treating the frontier column as optional.** If you cannot run it, say it is missing; do not imply the fine-tuned model "matches" something you did not measure.

---

## Daily challenge: an eval report comparing base, tuned and a frontier model

**Build** (reference: [`solutions/evalrun.py`](solutions/evalrun.py), [`solutions/probes.py`](solutions/probes.py), [`solutions/day4_solution.py`](solutions/day4_solution.py)):
1. An evaluation harness that scores any system on any set with validity, exact match and field accuracy, **Wilson intervals** and per-field accuracy, and records tokens and seconds per request.
2. A **paired comparison** of two systems on the same inputs (difference, 95% interval, p, wins/losses/ties).
3. A **forgetting suite**: unrelated probes with automatic graders (with tests that each grader accepts a right answer and rejects a wrong or order-shaped one), an over-generalisation rate, and a prose loss.
4. Error analysis: error types and the most-failed fields.
5. A frontier-model column: run it if you have a key; otherwise **state clearly that it is not run**.

**Acceptance criteria**
- Every probe has a right answer that passes and an obviously wrong one that fails; the "order JSON" detector does not fire on a normal reply.
- The comparison reports the paired difference with an interval and counts, and says whether the interval excludes zero.
- Hand-written and synthetic results are reported **separately** with their gap.
- The forgetting table includes the base model as the reference.
- No evaluation email appears in any training split (re-check it in the report).

**Stretch**
- Add a **second hand-written set** written after the first training run and report the score on it (the honest estimate after iteration).
- Bootstrap over **training seeds** (train three models) and report the spread of the exact-match score.
- Add a **calibration** check: does the model's token-level confidence on the answer predict whether it is right?
- Score with an **LLM judge** for the fields that are fuzzy (item names) and validate the judge against your exact-match labels (Week 7 Day 2).

## Further reading
- Zhang et al., *A Careful Examination of Large Language Model Performance on Grade School Arithmetic* (contamination and the generalisation gap).
- Luo et al., *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning*.
- Week 7 Day 1 and Day 3 of this course (eval-driven development, paired bootstrap, gating).
