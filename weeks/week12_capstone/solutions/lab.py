"""Shared setup for the Week 12 day scripts: the real embedder, index, reranker and golden set, built once and cached under ``outputs/``.

lab = Lab()                    # loads (or builds) the index of Weeks 1-11 and the models; about a minute the first time, seconds after
lab.retriever(alpha=0.5)       # a Retriever over it
lab.calibrated_gate()          # a RerankGate whose threshold was chosen on the DEV split only
lab.copilot(...)               # the product, assembled
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from copilot import answer as A  # noqa: E402
from copilot import core as C  # noqa: E402
from copilot import gate as GT  # noqa: E402
from copilot import golden as G  # noqa: E402
from copilot import guard as CG  # noqa: E402
from copilot import ingest as I  # noqa: E402
from copilot import retrieve as R  # noqa: E402

OUT = ROOT / "outputs"
INDEX_DIR = OUT / "w12_index"
WEEKS = ROOT / "weeks"


class Lab:
    def __init__(self, index_dir: Path = INDEX_DIR, load_reranker: bool = True):
        from common.embed import get_embedder

        t0 = time.perf_counter()
        self.embedder = get_embedder()
        self.docs = I.load_corpus(WEEKS)
        self.index, self.ingest_report = I.build_index(self.embedder, index_dir, self.docs)
        self.reranker = None
        if load_reranker:
            from common.rerank import get_reranker

            self.reranker = get_reranker()
        self.items = G.load()
        self.idf = C.idf_from_index(self.index)
        self.setup_seconds = time.perf_counter() - t0
        self._cal: dict | None = None

    def retriever(self, **cfg) -> R.Retriever:
        return R.Retriever(
            self.index, R.RetrievalConfig(**cfg), self.reranker if cfg.get("rerank") else None
        )

    def gate_scores(
        self, retriever: R.Retriever, split: str | None = None
    ) -> tuple[list[float], list[float]]:
        """(scores of answerable questions, scores of out-of-scope questions) under the rerank signal."""
        g = GT.RerankGate(self.reranker, 0.0)
        pos, neg = [], []
        for it in self.items:
            if it.kind == "adversarial" or (split and it.split != split):
                continue
            s = g.score(it.question, retriever.retrieve(it.question))
            (pos if it.answerable else neg).append(s)
        return pos, neg

    def calibrated_gate(self, retriever: R.Retriever | None = None) -> tuple[GT.RerankGate, dict]:
        """The gate with its threshold fitted on the dev split only (the test split is never used to choose anything)."""
        if self._cal is None:
            pos, neg = self.gate_scores(retriever or self.retriever(), "dev")
            self._cal = GT.calibrate(pos, neg)
        return GT.RerankGate(self.reranker, self._cal["threshold"]), self._cal

    def copilot(
        self,
        *,
        answerer=None,
        gate="rerank",
        retriever=None,
        guards=True,
        cache=None,
        trusted=("week",),
        **cfg,
    ) -> C.Copilot:
        r = retriever or self.retriever(**cfg)
        g = self.calibrated_gate()[0] if gate == "rerank" else gate
        return C.Copilot(
            r,
            answerer or A.ExtractiveAnswerer(self.idf),
            gate=g,
            input_guard=CG.InputGuard() if guards else None,
            output_guard=CG.OutputGuard() if guards else None,
            cache=cache,
            trusted_prefixes=trusted,
        )
