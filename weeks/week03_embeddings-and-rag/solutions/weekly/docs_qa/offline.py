"""Scripted stand-ins for the LLM so the whole bot runs (and is tested) with no API keys.

The reader is a *semantic extractive* reader: it sees only the numbered sources in the prompt, embeds
their sentences and answers with the best match (or declines). The rewriter appends the previous
user question to the follow-up. Both exercise the plumbing; neither says anything about answer quality.
"""

from __future__ import annotations

import json
import re

import numpy as np

from common.rag import IDK

READER_THRESHOLD = 0.55


def make_rules(embedder):
    def reader(prompt: str) -> str:
        question = re.search(r"<question>(.*?)</question>", prompt, re.S).group(1)
        cands = []
        for sid, body in re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S):
            for sent in re.split(r"(?<=[.!?])\s+|\n", body):
                sent = re.sub(r"[*`#|]", "", sent).strip()
                if (
                    50 <= len(sent) <= 400
                    and sent[-1] in ".!?"
                    and not sent.startswith(("<", "[", "-", "Week"))
                ):
                    cands.append((int(sid), sent))
        if not cands:
            return json.dumps({"answerable": False, "answer": IDK, "citations": []})
        sims = embedder.embed_documents([c[1] for c in cands]) @ embedder.embed_query(question)
        best = int(np.argmax(sims))
        if sims[best] < READER_THRESHOLD:
            return json.dumps({"answerable": False, "answer": IDK, "citations": []})
        sid, sent = cands[best]
        return json.dumps({"answerable": True, "answer": f"{sent} [{sid}]", "citations": [sid]})

    def rewriter(prompt: str) -> str:
        user_turns = re.findall(r"User: (.*)", prompt)
        follow_up = re.search(r"Follow-up: (.*)", prompt).group(1)
        return f"{user_turns[-1]} {follow_up}" if user_turns else follow_up

    return [
        (r"You rewrite follow-up questions|Rewrite the follow-up", rewriter),
        (r"<question>", reader),
    ]
