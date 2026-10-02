# AI engineer interview checklist

How to use it: for each line, **say the answer out loud in two sentences**, then say **where you measured it** (a lesson and a number from your own runs). A line you can only answer from memory is a line to re-run. Weeks refer to this repository; the numbers in the last column are from the runs it documents, on one laptop, and are examples of the kind of evidence to bring, not facts about every system.

## 1. How models behave (Weeks 1, 9, 10)
| topic | you should be able to explain | where you measured it |
|---|---|---|
| tokens and cost | why price, latency and limits are all in tokens; why other languages and code cost more | Week 1 Day 2: token counts across English, Vietnamese and code |
| sampling | what temperature and top-p change; why temperature 0 is not always deterministic | Week 1 Day 3 |
| context and the KV cache | what the cache stores, why a long context is expensive, what prompt caching reuses | Week 1 Day 4; Week 9 Day 6 (cache bytes per token); Week 11 Day 2 |
| the transformer | attention (and why divide by √d), RoPE, RMSNorm, GQA, MoE in a paragraph each | Week 9 (from-scratch decoder checked against the library) |
| fine-tuning | when to tune instead of prompt or retrieve; what LoRA trains; what a loss mask is | Week 10 Days 1-3: fine-tuned 74% exact against 3% for the best prompt on 38 hand-written emails |
| preference tuning | what DPO optimises and how it can satisfy its objective while hurting the task | Week 10 Day 5: DPO made the model worse (74% → 32%) |
| quantisation | what a "4-bit" file contains; why small models break first | Week 11 Day 1: Q4_0 collapsed the 135M extractor to 16% while perplexity moved 11% |

## 2. Building with models (Weeks 2, 5, 6)
| topic | you should be able to explain | where |
|---|---|---|
| structured output | schema, validation, retry; what a schema cannot say | Week 2 Day 3 |
| workflows vs agents | when a fixed flow beats a loop | Week 2 Day 4, Week 6 Day 1 |
| function calling | the model never runs anything; how a loop, a budget and a stop condition make it safe | Week 5 Days 1-2 (a 0.5B agent passed 0 of 7 tasks) |
| tool design | why tool descriptions are prompts | Week 5 Day 3 |
| MCP | the N×M problem it solves; the trust issue | Week 5 Day 4 |
| context engineering | compaction, scratchpads, why prompts grow quadratically | Week 5 Day 5 |
| multi-agent | when a second agent helps and when it hurts; handoffs; human approval | Week 6 Days 3-5 |

## 3. Retrieval (Weeks 3, 4, 12)
| topic | you should be able to explain | where |
|---|---|---|
| embeddings and chunking | what a vector is; the chunk-size trade-off; why heading-aware chunks | Week 3 Days 1, 3 |
| hybrid search | why BM25 and dense fail differently; fusion; re-ranking | Week 3 Day 5; **Week 12 Day 2**: hybrid hit@5 98% against 96% for either alone; both lessons found for 8 of 8 two-lesson questions with week scoping against 4 of 8 for either signal alone |
| evaluation | hit@k, MRR, golden sets, intervals, paired comparisons | Week 4 Day 1; Week 12 Day 1 (a validated golden set) |
| abstention | how a system decides it cannot answer | Week 12 Day 2: cross-encoder AUC 0.996 against 0.962 for cosine and 0.917 for BM25 |
| faithfulness | what a citation proves and does not | Week 4 Day 2; Week 12 Day 3 (support check catches an invented fact) |

## 4. Quality and operations (Weeks 7, 11, 12)
| topic | you should be able to explain | where |
|---|---|---|
| eval-driven development | the loop; dev vs test; why zero difference is a bug | Week 7 Day 1; Week 12 Day 3 |
| LLM judges | rubric, calibration (kappa, TPR/TNR), position bias | Week 7 Day 2 |
| CI gates | what each rule catches; why a gate must be few and trusted | Week 7 Day 3; Week 12 Day 4 (fails on 4 of 4 bad changes) |
| tracing | what to record, what never to record | Week 7 Day 4; Week 12 Days 4, 6 |
| cost and latency | where the time goes; caches; cascades | Week 7 Day 5; Week 12 Day 6 (the gate was 95% of the latency) |
| serving | continuous batching, PagedAttention, memory math, speculative decoding (and when it loses) | Week 11 Days 2-3 (a draft 0.43 of the target's cost made generation 2× slower) |
| a gateway | auth, per-user limits, back-pressure, health vs readiness | Week 11 Day 4; Week 12 Day 6 (admission must come before the work) |
| deployment | secrets, non-root, weights mounted not baked | Week 11 Day 6 |

## 5. Safety (Weeks 8, 12)
| topic | you should be able to explain | where |
|---|---|---|
| the core problem | one channel for instructions and data; no parameterised query for a prompt | Week 8 Day 1 |
| injection | direct vs indirect; why detectors are heuristics | Week 8 Days 2-3 |
| structural vs probabilistic defences | what stops an attack without recognising it | Week 12 Day 4: input guard alone 5 of 18 obeyed; with verification and the output guard 0 of 18 |
| agent security | least privilege; the lethal trifecta | Week 8 Day 4 |
| privacy | redact before logging; retention | Week 8 Day 5; Week 12 Day 4 |
| red teaming | static vs adaptive attackers; findings registers | Week 8 Days 6-7 |

## 6. The questions that are really about judgement
Answer each with a story from your own capstone, including a number and a mistake.
1. Tell me about a system you built. What did you measure first?
2. Your RAG answers are wrong. How do you find out whether retrieval or generation is at fault?
3. Your accuracy went up on the dev set. How do you know it is real?
4. A new model is cheaper. How do you decide whether to switch?
5. What is the first thing you would put in CI for an LLM feature?
6. How would you handle a prompt injection found in production?
7. How do you know your golden set is good?
8. What would you do with a week and no new data to make the system better?
9. When would you not use an LLM?
10. Describe a time a measurement contradicted what you expected. What did you change?

## 7. System-design practice (30 minutes each)
- A support assistant that can issue refunds. (Week 6: approval, idempotency, evaluation of actions, not text.)
- A search-and-answer product over 10 million documents. (Week 3-4, 11: index size, update path, per-tenant filters, cost.)
- A document-extraction pipeline with a human review queue. (Week 2, 10: schema, validation, fine-tune or prompt, error budget.)
- A coding agent with a sandbox. (Week 5-6, 8: execution limits, egress, permissions, evaluation of the final state.)
For each: requirements with numbers, the architecture, the evaluation plan, the three risks, the cost estimate, and what you would not build first.
