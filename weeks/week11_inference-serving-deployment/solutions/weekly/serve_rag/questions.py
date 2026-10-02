"""The labelled question set for the Week 11 weekly challenge: hand-written questions about the repository's own Week 9-11 lessons (the corpus the deployed API retrieves from).

Each in-scope question lists the lesson file(s) that contain the answer; retrieval counts as a hit when ANY of them is among the returned sources. Out-of-scope questions have no
answer in the corpus: the right behaviour is to retrieve nothing relevant and abstain. Questions are paraphrases, not copies of headings (a keyword search that only works on exact
headings would flatter itself). The split alternates (even index = dev, odd = test): choose settings on dev, report on test.
"""

from __future__ import annotations

IN_SCOPE = [
    ("How does rotary position embedding rotate queries and keys?", ["day4_transformer-block.md"]),
    ("Why do modern models use RMSNorm instead of LayerNorm?", ["day4_transformer-block.md"]),
    (
        "What is grouped-query attention and how does it shrink the KV cache?",
        ["day4_transformer-block.md", "day6_modern-architecture.md"],
    ),
    (
        "How is a byte-level BPE tokenizer trained on my own text?",
        ["day2_bpe-tokenizer-from-scratch.md"],
    ),
    (
        "Why does the BPE tokenizer split text with a regular expression first?",
        ["day2_bpe-tokenizer-from-scratch.md"],
    ),
    ("What does the causal mask do in attention?", ["day3_attention-from-scratch.md"]),
    (
        "Why are attention scores divided by the square root of the head dimension?",
        ["day3_attention-from-scratch.md"],
    ),
    (
        "What is a mixture of experts and how many experts run for each token?",
        ["day6_modern-architecture.md"],
    ),
    (
        "How does FlashAttention avoid building the full attention matrix?",
        ["day6_modern-architecture.md"],
    ),
    (
        "What loss does the mini GPT reach, and what are the baselines for it?",
        ["day5_train-a-mini-gpt.md"],
    ),
    ("What is the loss mask in supervised fine-tuning?", ["day1_when-to-fine-tune.md"]),
    (
        "When should I fine-tune instead of using retrieval or prompting?",
        ["day1_when-to-fine-tune.md"],
    ),
    ("How do MinHash and LSH find near-duplicate training examples?", ["day2_data-curation.md"]),
    (
        "Why must training data be decontaminated against the evaluation set?",
        ["day2_data-curation.md"],
    ),
    ("What does the rank control in LoRA?", ["day3_sft-with-lora.md"]),
    ("What is QLoRA and what does the 4-bit base change?", ["day3_sft-with-lora.md"]),
    ("What did DPO do to the order extraction model?", ["day5_preference-tuning.md"]),
    ("How do I merge a LoRA adapter into the base weights?", ["day6_merge-export-publish.md"]),
    (
        "What is forgetting after a fine-tune and how is it measured?",
        ["day4_evaluating-fine-tunes.md"],
    ),
    (
        "What is actually inside a Q4_K_M GGUF file?",
        ["day1_local-inference-and-quantization.md"],
    ),
    (
        "How does continuous batching differ from static batching?",
        ["day2_continuous-batching-and-paged-attention.md"],
    ),
    (
        "How does PagedAttention reduce wasted KV-cache memory?",
        ["day2_continuous-batching-and-paged-attention.md"],
    ),
    (
        "Why did speculative decoding slow down the CPU experiment?",
        ["day3_gpu-math-speculative-decoding-economics.md"],
    ),
    ("What does a token bucket rate limiter protect against?", ["day4_fastapi-llm-backend.md"]),
]

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Give me a recipe for pancakes.",
    "Who won the 2018 football World Cup?",
    "How do I treat a sprained ankle?",
    "What is the boiling point of water at sea level?",
    "Translate good morning into Spanish.",
    "What is the best programming language for building websites?",
    "How tall is Mount Everest?",
]


def questions() -> list[dict]:
    rows = [
        {"id": f"in{i:02d}", "question": q, "expected": exp, "in_scope": True}
        for i, (q, exp) in enumerate(IN_SCOPE)
    ]
    rows += [
        {"id": f"out{i:02d}", "question": q, "expected": [], "in_scope": False}
        for i, q in enumerate(OUT_OF_SCOPE)
    ]
    for i, r in enumerate(rows):
        r["split"] = "dev" if i % 2 == 0 else "test"
    return rows
