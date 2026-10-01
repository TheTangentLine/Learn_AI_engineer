"""A hand-labelled faithfulness set: 18 contexts (verbatim sentences from the lessons), each with one
FAITHFUL answer and one deliberately UNFAITHFUL answer.

Failure types (what a judge must be able to catch):
  contradiction  the answer says the opposite of the context
  numeric        the answer changes a number
  swap           same words, but subject/object (or which-is-better) are swapped
  fabrication    a plausible claim the context does not support

Labels are mine (human-written). Pairs are split by CONTEXT into dev (first 9) and test (last 9) so a
threshold tuned on dev never sees the test contexts.
"""

from __future__ import annotations

PAIRS: list[dict] = [
    {
        "q": "Why do clients need jitter when retrying?",
        "context": 'Jitter is essential. Without it, 100 clients that failed at the same instant all retry at the same instant, forever (the "thundering herd").',
        "good": "Without jitter, clients that fail together all retry together, which is the thundering herd problem.",
        "bad": "Without jitter, failed clients naturally spread their retries over time, which avoids a thundering herd.",
        "type": "contradiction",
    },
    {
        "q": "How does chat cost grow?",
        "context": "Because history is resent, cost per turn grows with conversation length, and the total grows ~quadratically.",
        "good": "Resending the history makes each turn more expensive, so the total cost grows roughly quadratically.",
        "bad": "Since only the newest message is sent, the cost per turn stays constant as the conversation grows.",
        "type": "contradiction",
    },
    {
        "q": "Why can't models count letters?",
        "context": "When a word is one atomic token, the model cannot see the letters inside; it only knows what it memorised about that token.",
        "good": "A single-token word hides its letters from the model, which only knows what it memorised about that token.",
        "bad": "Models spell words reliably because they read every letter of a word separately.",
        "type": "contradiction",
    },
    {
        "q": "What does a higher temperature do?",
        "context": 'Temperature isn\'t "creativity", it\'s "risk". Higher T raises the chance of every wrong token too.',
        "good": "A higher temperature increases the probability of wrong tokens as well.",
        "bad": "A higher temperature reduces hallucinations because it makes the output more diverse.",
        "type": "fabrication",
    },
    {
        "q": "How many validation retries?",
        "context": "Cap attempts (2–3). Persistent failure usually means a bad prompt, schema, or input, not bad luck.",
        "good": "Limit retries to two or three, because repeated failure usually points to a bad prompt, schema or input.",
        "bad": "Keep retrying until it works, since persistent failure is usually just bad luck.",
        "type": "contradiction",
    },
    {
        "q": "Which wins on a small corpus?",
        "context": "At this size numpy wins: an index adds build time and overhead and gains nothing.",
        "good": "On the small corpus, brute-force numpy wins because an index only adds build time and overhead.",
        "bad": "On the small corpus, an HNSW index is much faster than numpy and clearly worth building.",
        "type": "contradiction",
    },
    {
        "q": "When does the embedder truncate?",
        "context": "Past ~2,000 characters the embedder silently truncates the chunk, so big chunks are partly invisible to search.",
        "good": "Beyond about 2,000 characters the embedder silently truncates a chunk, so large chunks are partly invisible to search.",
        "bad": "Beyond about 20,000 characters the embedder silently truncates a chunk, so only huge chunks are affected.",
        "type": "numeric",
    },
    {
        "q": "Why number the sources?",
        "context": "Number the sources so citations are short and checkable.",
        "good": "Numbering the sources keeps citations short and easy to check.",
        "bad": "Numbering the sources lets the model recall them from its training data instead of the prompt.",
        "type": "fabrication",
    },
    {
        "q": "Did RRF beat BM25?",
        "context": "RRF with k=60 (MRR 0.62) did not beat plain BM25 (0.66).",
        "good": "RRF scored an MRR of 0.62, which did not beat the plain BM25 score of 0.66.",
        "bad": "RRF scored an MRR of 0.82, comfortably beating the plain BM25 score of 0.66.",
        "type": "numeric",
    },
    # ---- test split starts here (contexts 9-17) ----
    {
        "q": "What did the voting experiment show about agreement?",
        "context": "But unanimity was 100% accurate and a split vote was 0% accurate.",
        "good": "Unanimous votes were always right, while split votes were always wrong.",
        "bad": "Split votes were always right, while unanimous votes were always wrong.",
        "type": "swap",
    },
    {
        "q": "Does the API remember earlier calls?",
        "context": "The API is stateless. The model remembers nothing between calls.",
        "good": "The API is stateless, so the model retains nothing from one call to the next.",
        "bad": "The API stores the conversation on the server, so the model remembers earlier calls.",
        "type": "contradiction",
    },
    {
        "q": "Why evict memory in batches?",
        "context": "Evict in batches (e.g. 4 messages at once) so you pay for one compression call every few turns, not every turn.",
        "good": "Evicting several messages at once means you pay for a compression call only every few turns.",
        "bad": "Compress after every single message so memory is always up to date.",
        "type": "contradiction",
    },
    {
        "q": "What do embeddings encode?",
        "context": "Opposites look nearly identical: embeddings encode topic strongly and polarity weakly.",
        "good": "Embeddings encode topic strongly but polarity weakly, so opposites can look nearly identical.",
        "bad": "Embeddings encode polarity strongly but topic weakly, so opposites look very different.",
        "type": "swap",
    },
    {
        "q": "Workflow or single call first?",
        "context": "Start with a single well-prompted call. Add a workflow only when it measurably improves results.",
        "good": "Begin with one well-prompted call and add workflow steps only if they measurably improve results.",
        "bad": "Begin with a multi-step workflow, and only simplify to one call if it measurably hurts.",
        "type": "contradiction",
    },
    {
        "q": "How good was the free follow-up baseline?",
        "context": "Simply concatenating the previous question with the follow-up (zero LLM calls, zero latency) scored 4/5, the same as the LLM rewrite.",
        "good": "Concatenating the previous question with the follow-up needs no LLM call and scored 4/5, the same as the LLM rewrite.",
        "bad": "Concatenating the previous question with the follow-up needs no LLM call and scored 1/5, far below the LLM rewrite.",
        "type": "numeric",
    },
    {
        "q": "How much do model prices differ?",
        "context": "Prices differ by ~100× between the top and bottom of this table, so model choice is the single biggest cost lever you have (more than any prompt trick).",
        "good": "Prices vary by roughly 100 times across the model tiers, so choosing the model is the biggest cost lever.",
        "bad": "Prices are nearly identical across model tiers, so the choice of model barely affects cost.",
        "type": "contradiction",
    },
    {
        "q": "How noisy is a 10-example dev set?",
        "context": "With 10 dev examples, one example is 10 points.",
        "good": "With only 10 dev examples, a single example shifts the score by 10 points.",
        "bad": "With only 10 dev examples, a single example shifts the score by 1 point.",
        "type": "numeric",
    },
    {
        "q": "How well do structure-aware chunkers keep answers whole?",
        "context": "Heading-aware and semantic chunkers keep the answer's paragraph whole 93% of the time; naive fixed cuts do so only 40%.",
        "good": "Heading-aware chunkers kept the answer paragraph whole 93% of the time, versus 40% for naive fixed cuts.",
        "bad": "Naive fixed cuts kept the answer paragraph whole 93% of the time, versus 40% for heading-aware chunkers.",
        "type": "swap",
    },
]
N_DEV = 9


def items() -> list[dict]:
    """Flatten to labelled examples: label 1 = faithful, 0 = unfaithful."""
    out = []
    for i, p in enumerate(PAIRS):
        split = "dev" if i < N_DEV else "test"
        out.append(
            {
                "pair": i,
                "split": split,
                "q": p["q"],
                "context": p["context"],
                "answer": p["good"],
                "label": 1,
                "type": "none",
            }
        )
        out.append(
            {
                "pair": i,
                "split": split,
                "q": p["q"],
                "context": p["context"],
                "answer": p["bad"],
                "label": 0,
                "type": p["type"],
            }
        )
    return out
