"""The Week 6 support system with a privacy layer around it.

``PrivateSupport.handle`` pseudonymises personal data BEFORE anything else sees the message: the model, the conversation history, the audit
log and any trace see ``<EMAIL_1>``, never the address. The reply is restored on the way out, so the customer reads their own values.
Card numbers keep going through the Week 6 card guard (removed, never restored: nothing needs to read a card back).

The vault that maps tokens back lives in this process only. That is the point (the database holds no raw personal data) and the cost
(after a restart the stored history keeps its tokens and they can no longer be restored: persist the vault ONLY if you can protect it
better than you protect the database, for example encrypted with a key held elsewhere).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "week07_evals-observability-llmops" / "solutions"))

import day1_solution  # noqa: E402,F401  (puts the Week 6 support_system package on the path)
from support_system.system import Reply, SupportSystem  # noqa: E402

from common import pii  # noqa: E402

KEEP_FOR_CARD_GUARD = {"CREDIT_CARD"}


class PrivateSupport(SupportSystem):
    def __init__(self, *args, min_score: float = 0.5, **kw):
        super().__init__(*args, **kw)
        self._vaults: dict[str, pii.Pseudonymizer] = {}
        self.min_score = min_score

    def handle(self, conversation_id: str, message: str) -> Reply:
        ps = self._vaults.setdefault(conversation_id, pii.Pseudonymizer(min_score=self.min_score))
        safe = ps.pseudonymize(
            message, skip_types=KEEP_FOR_CARD_GUARD
        )  # cards are left for the card guard, which removes them outright
        reply = super().handle(conversation_id, safe)
        reply.text = ps.restore(reply.text)
        return reply

    def forget(self, conversation_id: str) -> None:
        """Drop the vault: the tokens in the stored history become unreadable for good."""
        self._vaults.pop(conversation_id, None)
