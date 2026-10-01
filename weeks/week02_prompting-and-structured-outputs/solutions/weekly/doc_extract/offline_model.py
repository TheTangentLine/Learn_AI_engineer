"""A scripted fake model for the pipeline. It knows the gold answers (via the DOC-nn marker in the
text) and injects specific, realistic failures so every pipeline branch is exercised:

  01 invoice : first extraction has an arithmetic error        -> caught by validator, retry fixes
  03 invoice : vendor silently wrong                           -> NOT caught (shows in the report)
  04 invoice : classified as 'receipt' with high confidence    -> schema mismatch, 3 failures
  06 receipt : first extraction uses an invalid enum value     -> caught, retry fixes
  09 offer   : first extraction has an unparseable date        -> caught, retry fixes
  11 offer   : classifier is unsure (0.55)                     -> gate sends it to review
"""

from __future__ import annotations

import json
import re

FAULTS = {"01", "03", "04", "06", "09", "11"}


def make_rules(gold: dict[str, dict]):
    by_id = {name.removesuffix(".pdf"): g for name, g in gold.items()}

    def doc_id(prompt: str) -> str:
        return re.search(r"DOC-(\d+)", prompt).group(1)

    def classify(prompt: str) -> str:
        i = doc_id(prompt)
        kind, conf = by_id[i]["doc_type"], 0.96
        if i == "04":
            kind, conf = "receipt", 0.9
        if i == "11":
            conf = 0.55
        return json.dumps({"doc_type": kind, "confidence": conf})

    def extract(prompt: str) -> str:
        i = doc_id(prompt)
        data = json.loads(json.dumps(by_id[i]["data"]))
        retry = "failed validation" in prompt
        if not retry:
            if i == "01":
                data["total"] = round(data["total"] + 10, 2)
            elif i == "03":
                data["vendor"] = "Northwind Supply Co"
            elif i == "06":
                data["payment_method"] = "visa"
            elif i == "09":
                data["start_date"] = "sometime in November"
        return json.dumps(data)

    return [(r"Classify the document", classify), (r"(?s).*", extract)]
