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
| LoRA, epoch 1 / 2 / 3 | one instruction line (about 60 tokens) | the Day 3 checkpoints, adapters merged for speed |
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

## Results (greedy decoding; 95% Wilson intervals; "exact" = all 8 fields right)

| system | set | valid JSON | valid Order | exact | field accuracy | prompt tokens | s / email |
|---|---|---|---|---|---|---|---|
| base, 3-shot | hand-written | 97% | 76% | **3%** [0%, 13%] | 39% | 593 | 1.7 |
| base, 3-shot | synthetic | 95% | 76% | 2% [1%, 7%] | 39% | 598 | 1.9 |
| LoRA, epoch 1 | hand-written | 100% | 95% | 61% [45%, 74%] | 87% | 60 | 1.3 |
| LoRA, epoch 1 | synthetic | 100% | 98% | 83% [74%, 89%] | 96% | 65 | 1.4 |
| LoRA, epoch 2 | hand-written | 100% | 92% | **74%** [58%, 85%] | 89% | 60 | 1.3 |
| LoRA, epoch 2 | synthetic | 100% | 99% | 92% [85%, 96%] | 98% | 65 | 1.3 |
| LoRA, epoch 3 | hand-written | 100% | 92% | **74%** [58%, 85%] | 89% | 60 | 1.2 |
| LoRA, epoch 3 | synthetic | 100% | 99% | 92% [85%, 96%] | 98% | 65 | 1.3 |
| a hosted frontier model, prompted | n/a | **not run** | | | | | |

**The fine-tuned model is far better than the best prompt, and the evidence is not close.** On the same 38 hand-written emails, epoch 2 against the 3-shot prompt (paired bootstrap, 10,000 resamples):

- **exact match: +0.71** (95% interval **[+0.55, +0.84]**, p < 0.0001): better on **27** emails, worse on **0**, tied on 11;
- **per-field accuracy: +0.50** (interval [+0.38, +0.62]): better on 34 emails, worse on 3, tied on 1.

The interval excludes zero by a wide margin and the model never did *worse* on exact match. That is a statement about **these 38 emails** (so it carries the caveats of section 6) and about a **135M-parameter** model against a prompt for the *same* 135M model: it says fine-tuning rescues a model that prompting could not, not that a 135M model rivals a large one.

**The generalisation gap is 18 points** (hand-written 74%, synthetic 92% at epoch 3) and 22 points at epoch 1. The model learned the task well enough to read phrasing it never saw, and not as well as it reads the generator's. The 38-email interval for the hand-written score is wide ([58%, 85%]); the synthetic one (n = 100) is narrower ([85%, 96%]).

**Cost, too:** the prompt is **60 tokens** against **593** (10× fewer input tokens), and the answer arrives in 1.2 to 1.3 s per email against 1.7 s (the 3-shot prompt mostly produces *shorter, wrong* answers, so its decoding is cheaper than a correct JSON object would be).

**Overfitting across epochs: not visible here.** Epoch 1 → 2 raised the hand-written exact match from 61% to 74% (and the synthetic from 83% to 92%); epoch 3 changed nothing (74% and 92% again) while the dev loss kept falling (0.0073 to 0.0056). More epochs bought confidence but no accuracy; there is no sign of the hand-written score *falling*, which would be overfitting. With 38 emails, a one-email change is 2.6 points, so "epoch 2 or 3" is a coin toss: pick the cheaper one.

## Where the remaining errors are (epoch 2, hand-written set)

| field | accuracy |
|---|---|
| is_order | 92% |
| customer_name | 92% |
| order_id | 92% |
| **items** | **82%** |
| urgency | 87% |
| delivery_date | 87% |
| total_amount | 92% |
| currency | 92% |

Error types over the 38 emails: **28 exact**, **7 valid orders with wrong fields** (4 wrong `items`, 2 wrong `urgency`, 2 wrong `delivery_date`), **3 invalid orders**, **0 replies that were not JSON**. Examples from the run (which field, not the detail of the wrong value): the email ending "no rush; delivery by 2026-12-15 would be great. EUR 45" got the wrong **urgency**; "Sam Rivera here ... GBP 23.99 all in. Delivery date TBD." produced an **invalid order** (the schema's own checks rejected it); the long "Good morning ... £2,340.00" email (a thousands separator, a date written as "15 December 2026") got the wrong **delivery date**; and the bullet-list email with `4 x whiteboard markers (black)` got its **items** wrong. These are phrasing and layout the generator did not produce in that form: the generalisation gap, itemised. The right response is to add *that category* of variety to the data and re-evaluate on **fresh** hand-written emails, never to tune against these 38.

## Forgetting (36 unrelated probes; ordinary chat prompt)

| model | probe accuracy | facts (10) | arithmetic (8) | format (10) | json (8) | order JSON for an unrelated prompt | loss on ordinary prose (nats/token) |
|---|---|---|---|---|---|---|---|
| base | 31% | 70% | 25% | 20% | 0% | 0% | 3.880 |
| LoRA, epoch 1 | 39% | 60% | 50% | 30% | 12% | 0% | 4.088 |
| LoRA, epoch 2 | 28% | 40% | 25% | 30% | 12% | 0% | 4.108 |
| LoRA, epoch 3 | 25% | 30% | 25% | 30% | 12% | 0% | 4.117 |

(The "base" row is the untouched model with its ordinary chat prompt; its probe accuracy is low because SmolLM2-135M is a very small model.)

- **Some factual recall was lost**: capitals and similar facts fall from 7 of 10 to 3 of 10 over the three epochs (the loss is monotonic with training length, which is what forgetting looks like). The **prose loss rises by 0.24 nats per token** (6%): the model's general language modelling is measurably worse.
- **No over-generalisation:** none of the 36 unrelated prompts got an order object back (0%): the model still follows an ordinary prompt differently from the task prompt, because the task has its own system message.
- **Arithmetic, format and JSON probes moved by one to three questions each** in either direction: with n = 8 to 10 that is noise (epoch 1's higher total is a few lucky arithmetic answers). Only the factual drop and the prose loss are clear.
- Whether this matters is a **product decision**: an extraction service does not need trivia. If it does, mix general chat data into the fine-tuning set, lower the rank, or keep the adapters separate and load them only for this task.

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
