"""A small general-capability probe set to measure FORGETTING: 36 prompts that have nothing to do with order extraction, each with an automatic grader.

A model fine-tuned on one narrow task can lose abilities it had (answering a factual question, following a one-line format instruction) or start to apply
the narrow task everywhere (answering every prompt with an order JSON). These probes are answered with the model's ordinary chat prompt (no task
system message) and graded by a regular expression or a JSON check. They are easy on purpose: they measure whether the fine-tune BROKE something, not
how good the model is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    kind: str  # facts | arithmetic | format | json
    prompt: str
    check: str  # regex the lowercase reply must match (search), "cs:<regex>" for a case-sensitive search, or "json:<key>=<value>" for a JSON check, or "exact:<text>"

    def passes(self, reply: str) -> bool:
        text = reply.strip()
        if self.check.startswith("json:"):
            key, _, want = self.check[5:].partition("=")
            try:
                obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text))
            except ValueError:
                return False
            return isinstance(obj, dict) and str(obj.get(key)).lower() == want.lower()
        if self.check.startswith("exact:"):
            return text.strip(" .!\n") == self.check[6:]
        if self.check.startswith("cs:"):  # case-sensitive search
            return re.search(self.check[3:], text) is not None
        return re.search(self.check, text.lower()) is not None


def emits_order_json(reply: str) -> bool:
    """The signature of an over-fitted fine-tune: a reply to an unrelated prompt that is the order object."""
    return reply.lstrip().startswith('{"is_order"')


PROBES: list[Probe] = [
    Probe("facts", "What is the capital of France?", r"paris"),
    Probe("facts", "What is the capital of Japan?", r"tokyo"),
    Probe("facts", "Which planet is known as the Red Planet?", r"mars"),
    Probe("facts", "How many days are in a week?", r"\b7\b|seven"),
    Probe("facts", "What colour do you get by mixing blue and yellow?", r"green"),
    Probe("facts", "Who wrote the play Romeo and Juliet?", r"shakespeare"),
    Probe("facts", "What is the chemical formula of water?", r"h2o|h₂o"),
    Probe("facts", "What is the largest ocean on Earth?", r"pacific"),
    Probe("facts", "How many legs does a spider have?", r"\b8\b|eight"),
    Probe("facts", "What comes after Monday?", r"tuesday"),
    Probe("arithmetic", "What is 7 + 5?", r"\b12\b"),
    Probe("arithmetic", "What is 9 - 4?", r"\b5\b"),
    Probe("arithmetic", "What is 6 times 3?", r"\b18\b"),
    Probe("arithmetic", "What is 20 divided by 4?", r"\b5\b"),
    Probe("arithmetic", "What is 15 + 27?", r"\b42\b"),
    Probe("arithmetic", "What is 8 * 8?", r"\b64\b"),
    Probe("arithmetic", "What is 100 - 37?", r"\b63\b"),
    Probe("arithmetic", "What is 12 + 12?", r"\b24\b"),
    Probe("format", "Reply with exactly the word YES and nothing else.", "exact:YES"),
    Probe("format", "Reply with exactly the word BANANA and nothing else.", "exact:BANANA"),
    Probe("format", "Write the number 42 in words.", r"forty[- ]two"),
    Probe(
        "format",
        "List three fruits separated by commas, and nothing else.",
        r"^[^,\n]+,[^,\n]+,[^,\n]+\.?$",
    ),
    Probe("format", "Say hello in Spanish.", r"hola"),
    Probe("format", "Say thank you in French.", r"merci"),
    Probe("format", "Spell the word CAT backwards.", r"\btac\b"),
    Probe("format", "Answer with true or false: the sun is a star.", r"^\W*true\W*$"),
    Probe("format", "Complete the line: Roses are red, violets are", r"blue"),
    Probe(
        "format",
        "Write the word 'hello' in capital letters.",
        "cs:HELLO",
    ),
    Probe("json", 'Return JSON with a single key "x" whose value is 1.', "json:x=1"),
    Probe(
        "json",
        "Extract the person's name as JSON {\"name\": ...} from: 'My name is Sara and I like tea.'",
        "json:name=Sara",
    ),
    Probe(
        "json",
        "Return JSON {\"city\": ...} for: 'I live in Madrid and work from home.'",
        "json:city=Madrid",
    ),
    Probe(
        "json",
        'Return JSON {"sentiment": "positive" or "negative"} for: \'I love this, it is wonderful!\'',
        "json:sentiment=positive",
    ),
    Probe(
        "json",
        'Return JSON {"sentiment": "positive" or "negative"} for: \'This is terrible and I hate it.\'',
        "json:sentiment=negative",
    ),
    Probe(
        "json",
        "Return JSON {\"n\": ...} where n is the number of apples in: 'I bought 7 apples today.'",
        "json:n=7",
    ),
    Probe("json", "Return JSON {\"color\": ...} for: 'The car is red.'", "json:color=red"),
    Probe("json", "Return JSON {\"animal\": ...} for: 'A dog barked outside.'", "json:animal=dog"),
]
