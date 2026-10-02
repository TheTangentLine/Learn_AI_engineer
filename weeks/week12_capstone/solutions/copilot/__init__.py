"""Course Copilot: the Week 12 reference capstone, a cited, guarded, observable question-answering product over this course's own lessons.

The package assembles what Weeks 1-11 built: ingestion and hybrid retrieval (Weeks 3-4), a core flow with verification (Weeks 2, 4), guards (Week 8), tracing and cost accounting (Week 7),
and the serving gateway (Week 11). Modules: ``ingest``, ``retrieve``, ``answer``, ``guard``, ``core``, ``golden``, ``obs``, ``service``.
"""

from . import (
    _paths,  # noqa: E402,F401  (first: puts the repository root, ``common`` and the Week 11 gateway on sys.path before any submodule imports them)
)
