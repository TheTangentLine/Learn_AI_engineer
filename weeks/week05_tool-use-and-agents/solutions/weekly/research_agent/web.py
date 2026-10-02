"""A tiny local 'web': the course lessons served over real HTTP, with a search endpoint.

Why local: the project runs offline and deterministically, yet the agent's tools make real HTTP requests, so the
fetch layer's safety (allowlist, size caps, redirects, timeouts) is exercised for real. Swap ``LocalWeb`` for a real
search API and ``Fetcher``'s allowlist for your own policy to go live.

    with LocalWeb(docs) as web:
        web.base        # "http://127.0.0.1:PORT"
        # GET /search?q=...&limit=5  -> JSON [{"title", "url", "snippet"}]
        # GET /page/<slug>           -> the lesson as text (mermaid diagrams removed)
        # extra test pages: LocalWeb(docs, extra={"/evil": ("text/plain", body, delay_seconds)})
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")
_MERMAID = re.compile(r"```mermaid\n.*?```\n?", re.S)


def slug_of(short: str) -> str:
    return short.replace("/", "-")


def clean_markdown(text: str) -> str:
    text = _MERMAID.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SearchIndex:
    """BM25 over paragraphs; a page's score is its best paragraph's, its snippet is that paragraph."""

    def __init__(self, pages: dict[str, tuple[str, str]]):  # slug -> (title, text)
        self.pages = pages
        self.paragraphs: list[tuple[str, str]] = []
        for slug, (_title, text) in pages.items():
            for para in re.split(r"\n\s*\n", clean_markdown(text)):
                if len(para.split()) >= 8 and not para.lstrip().startswith("```"):
                    self.paragraphs.append((slug, para.strip()))
        self.bm25 = BM25Okapi([tokens(p) for _, p in self.paragraphs]) if self.paragraphs else None

    def search(self, query: str, limit: int = 5) -> list[dict]:
        q = tokens(query)
        if not q or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(q)
        best: dict[str, tuple[float, str]] = {}
        for (slug, para), score in zip(self.paragraphs, scores, strict=True):
            if score > 0 and score > best.get(slug, (0.0, ""))[0]:
                best[slug] = (float(score), para)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:limit]
        return [
            {"title": self.pages[slug][0], "slug": slug, "snippet": " ".join(para.split())[:240]}
            for slug, (_, para) in ranked
        ]


class LocalWeb:
    def __init__(self, docs, extra: dict[str, tuple[str, str, float]] | None = None):
        self.pages = {slug_of(d.short): (d.title, d.text) for d in docs}
        self.index = SearchIndex(self.pages)
        self.extra = extra or {}
        self.hits: list[str] = []  # every request path, for tests
        self._server: ThreadingHTTPServer | None = None
        self.base = ""

    def __enter__(self) -> LocalWeb:
        web = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, ctype: str, body: str, headers: dict | None = None):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                url = urlparse(self.path)
                web.hits.append(self.path)
                if url.path in web.extra:
                    ctype, body, delay = web.extra[url.path]
                    time.sleep(delay)
                    return self._send(200, ctype, body)
                if url.path == "/redirect":
                    target = parse_qs(url.query).get("to", ["/"])[0]
                    return self._send(302, "text/plain", "", {"Location": target})
                if url.path == "/search":
                    qs = parse_qs(url.query)
                    q = qs.get("q", [""])[0]
                    limit = int(qs.get("limit", ["5"])[0])
                    out = [
                        {**r, "url": f"{web.base}/page/{r['slug']}"}
                        for r in web.index.search(q, limit)
                    ]
                    for r in out:
                        r.pop("slug")
                    return self._send(200, "application/json", json.dumps(out))
                if url.path.startswith("/page/"):
                    slug = url.path[len("/page/") :]
                    if slug in web.pages:
                        return self._send(
                            200, "text/markdown; charset=utf-8", clean_markdown(web.pages[slug][1])
                        )
                return self._send(404, "text/plain", "not found")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    @property
    def host(self) -> str:
        return self.base.removeprefix("http://")
