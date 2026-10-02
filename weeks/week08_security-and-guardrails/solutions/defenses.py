"""A defence pipeline for the RAG target: five independent layers that can be switched on one at a time.

  input_guard     refuse a user message the detector flags (the user channel)
  doc_filter      remove instruction-like sentences from retrieved text before the model sees it ("sentences"), or drop whole chunks ("chunks")
  spotlight       mark untrusted text so the model can tell data from instructions ("delimit" or "datamark")
  isolate_secret  the model that reads documents never sees the secret (least data: it did not need it)
  output_guard    sanitise what the model wrote: secret scan, no images, no links to unlisted hosts

The first three and spotlighting live BEFORE the model and depend on a detector or on the model's cooperation (probabilistic).
The last two do not depend on the model at all (structural): a fully hijacked model still cannot leak what it never saw or emit
what the output guard removes.
"""

from __future__ import annotations

import dataclasses
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from common import guard, rag  # noqa: E402
from common import redteam as rt  # noqa: E402

ALLOWED_HOSTS = {"docs.acme.example"}
REFUSAL = "I can only answer questions about the product documentation."
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Defenses:
    input_guard: bool = False
    doc_filter: str = ""  # "" | "sentences" | "chunks"
    spotlight: str = ""  # "" | "delimit" | "datamark"
    isolate_secret: bool = False
    output_guard: bool = False
    threshold: float = guard.THRESHOLD
    rng: random.Random | None = (
        None  # seeded ONLY in experiments, so a run is reproducible; None = real randomness
    )

    def label(self) -> str:
        parts = [
            n
            for n, on in (
                ("input", self.input_guard),
                (f"doc:{self.doc_filter}", bool(self.doc_filter)),
                (f"spot:{self.spotlight}", bool(self.spotlight)),
                ("isolate", self.isolate_secret),
                ("output", self.output_guard),
            )
            if on
        ]
        return "+".join(parts) or "none"

    # -- the stages
    def _filter_text(self, text: str) -> tuple[str, bool]:
        """Drop flagged sentences (or lines). Returns the cleaned text and whether anything was removed."""
        pieces = [p for p in SENTENCE.split(text) if p.strip()]
        keep = [p for p in pieces if not guard.detect(p, threshold=self.threshold).flagged]
        return (" ".join(keep) if len(keep) != len(pieces) else text), len(keep) != len(pieces)

    def run(self, target, question: str, hits, system: str, reached: bool) -> rt.Observation:
        canary = target.canary
        blocked = ""
        if self.input_guard and guard.detect(question, threshold=self.threshold).flagged:
            return rt.Observation(output=REFUSAL, reached=reached, blocked_by="input_guard")
        shown = list(hits)
        if self.doc_filter:
            kept = []
            for h in shown:
                if self.doc_filter == "chunks":
                    if guard.detect(h.text, threshold=self.threshold).flagged:
                        blocked = "doc_filter"
                        continue
                    kept.append(h)
                else:
                    cleaned, changed = self._filter_text(h.text)
                    blocked = "doc_filter" if changed else blocked
                    kept.append(dataclasses.replace(h, text=cleaned) if changed else h)
            shown = kept
        if not shown:
            return rt.Observation(output=rag.IDK, reached=reached, blocked_by=blocked)
        if self.isolate_secret:
            system = system.split("\n\nInternal reference code")[0]
        boundary = guard.new_boundary(self.rng) if self.spotlight == "delimit" else ""
        if self.spotlight:
            system = system + "\n" + guard.spotlight_instruction(self.spotlight, boundary=boundary)
            shown = [
                dataclasses.replace(
                    h,
                    text=guard.delimit(h.text, boundary)
                    if self.spotlight == "delimit"
                    else guard.datamark(h.text),
                )
                for h in shown
            ]
        question_for_model = question
        prompt = rag.build_prompt(question_for_model, shown)
        out = target.model(system, prompt)
        if self.output_guard:
            policy = guard.OutputPolicy(allowed_hosts=ALLOWED_HOSTS, secrets=[canary.secret])
            res = guard.guard_output(out, policy)
            if not res.clean:
                blocked = "output_guard"
            out = res.text
        return rt.Observation(output=out, reached=reached, blocked_by=blocked)


CONFIGS: dict[str, Defenses] = {
    "none": Defenses(),
    "input guard": Defenses(input_guard=True),
    "doc filter (sentences)": Defenses(doc_filter="sentences"),
    "doc filter (chunks)": Defenses(doc_filter="chunks"),
    "spotlight: delimit": Defenses(spotlight="delimit"),
    "spotlight: datamark": Defenses(spotlight="datamark"),
    "isolate secret": Defenses(isolate_secret=True),
    "output guard": Defenses(output_guard=True),
    "structural (isolate + output)": Defenses(isolate_secret=True, output_guard=True),
    "all layers": Defenses(
        input_guard=True,
        doc_filter="sentences",
        spotlight="datamark",
        isolate_secret=True,
        output_guard=True,
    ),
}
