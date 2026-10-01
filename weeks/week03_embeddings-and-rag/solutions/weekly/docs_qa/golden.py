"""Golden questions with chunker-independent ground truth: (question, lesson, answer phrase)."""

from __future__ import annotations

# Answerable: the phrase (lower-cased, markdown stripped) must appear in a chunk of that lesson.
ANSWERABLE: list[tuple[str, str, str]] = [
    (
        "how do I stop my app from hammering an API that keeps failing",
        "week01/day5",
        "100 clients that failed at the same instant",
    ),
    (
        "why can't a language model count the letters in a word",
        "week01/day2",
        "counting letters is guesswork",
    ),
    (
        "how much GPU memory does remembering a long conversation take",
        "week01/day4",
        "kv bytes = 2 (k and v)",
    ),
    (
        "what setting makes the output more random or more deterministic",
        "week01/day3",
        'temperature isn\'t "creativity", it\'s "risk"',
    ),
    (
        "ways to get machine-readable JSON out of a model without parse errors",
        "week02/day3",
        "constrains decoding",
    ),
    (
        "keep a chat from growing too expensive while still remembering the user's name",
        "week02/day5",
        "enforce the budget at assembly time",
    ),
    (
        "asking the same question several times and voting to be more accurate",
        "week02/day2",
        "agreement is a confidence signal",
    ),
    (
        "send each support ticket to a specialised handler",
        "week02/day4",
        "specialised prompts are shorter and sharper",
    ),
    (
        "how to split data so I don't fool myself when tuning a prompt",
        "week02/day6",
        "selection bias",
    ),
    (
        "picking the cheapest model that is still good enough",
        "week01/day6",
        "pick the cheapest model that clears it",
    ),
    (
        "reuse the start of a prompt across requests to save money",
        "week01/day4",
        "stable first, volatile last",
    ),
    ("what should a well written prompt contain", "week02/day1", "examples beat adjectives"),
    (
        "can I trust the probabilities a model assigns to its words",
        "week01/day2",
        "a confidently wrong model scores high too",
    ),
    (
        "running many requests at once without hitting rate limits",
        "week01/day5",
        "unbounded gather = stampede",
    ),
    (
        "why does a bigger chunk size make retrieval look better than it is",
        "week03/day3",
        "hit@5 rewards large chunks",
    ),
    (
        "what goes wrong when a metadata filter is very selective in hnswlib",
        "week03/day2",
        "makes raw hnswlib raise",
    ),
    (
        "how should I choose the score threshold for saying I don't know",
        "week03/day4",
        "calibrate τ on data",
    ),
    (
        "why do embeddings rate opposite statements as nearly identical",
        "week03/day1",
        "polarity weakly",
    ),
    (
        "why did plain keyword search beat fusing with vectors",
        "week03/day5",
        "rrf's equal-vote averaging dilutes it",
    ),
    (
        "security: where should per-user access filtering happen",
        "week03/day2",
        "never after retrieval in app code",
    ),
]

UNANSWERABLE: list[str] = [
    "what is the capital of Australia",
    "how do I deploy a service to Kubernetes",
    "who won the 2018 football world cup",
    "what is the recipe for sourdough bread",
    "how many moons does Jupiter have",
    "how do I configure a router for port forwarding",
    "what is the boiling point of tungsten",
    "who painted the ceiling of the Sistine Chapel",
]

# (earlier question, ambiguous follow-up, lesson, phrase): the follow-up alone is unretrievable
FOLLOWUPS: list[tuple[str, str, str, str]] = [
    (
        "How do I retry failed API calls?",
        "And how do I limit how many run at once?",
        "week01/day5",
        "unbounded gather = stampede",
    ),
    (
        "Why can't models count letters in a word?",
        "Does the same thing explain mistakes with decimal numbers?",
        "week01/day2",
        "numbers are chopped arbitrarily",
    ),
    (
        "How do I get structured output from a model?",
        "What if its JSON fails my schema?",
        "week02/day3",
        "cap attempts",
    ),
    (
        "What is prompt caching?",
        "What kinds of things break it?",
        "week01/day4",
        "silent invalidator",
    ),
    (
        "How does self-consistency work?",
        "When does it not help?",
        "week02/day2",
        "when it does not help",
    ),
]
