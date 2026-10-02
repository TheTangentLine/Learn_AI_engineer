"""The agent's web tools. Sources are numbered on FIRST FETCH (not on search), so [n] always means 'a page the
agent actually read'; the report checker relies on that."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import quote

from common.tools import ToolFailure, tool

from .fetch import Fetcher, FetchError

PAGE_CHARS = 3000


@dataclass
class Source:
    n: int
    url: str
    title: str
    text: str


@dataclass
class SourceLog:
    sources: list[Source] = field(default_factory=list)

    def register(self, url: str, text: str) -> Source:
        for s in self.sources:
            if s.url == url:
                return s
        m = re.search(r"^#\s+(.+)$", text, re.M)
        src = Source(len(self.sources) + 1, url, m.group(1).strip() if m else url, text)
        self.sources.append(src)
        return src

    def get(self, n: int) -> Source | None:
        return self.sources[n - 1] if 1 <= n <= len(self.sources) else None


def make_research_tools(base: str, fetcher: Fetcher, log: SourceLog) -> list:
    cache: dict[str, str] = {}

    @tool
    def search_web(query: str, limit: int = 5) -> str:
        """Search the document collection. Returns the best matching pages: title, URL and a snippet. Search results are
        leads, not evidence: open a page with fetch_page before you rely on it or cite it.

        Args:
            query: A few specific words, e.g. "CRAG rescued baseline misses".
            limit: Maximum results (1-10).
        """
        if not 1 <= limit <= 10:
            raise ToolFailure("limit must be between 1 and 10")
        try:
            page = fetcher.get(f"{base}/search?q={quote(query)}&limit={limit}")
        except FetchError as exc:
            raise ToolFailure(str(exc)) from None
        results = json.loads(page.text)
        if not results:
            return "No results. Try different or fewer words."
        return "\n".join(
            f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)
        )

    @tool
    def fetch_page(url: str, start: int = 0) -> str:
        """Read a page. The first time you open a page it gets a source number [n]; cite it in your report as [n].
        Long pages come in parts of about 3000 characters: pass start to read on.

        Args:
            url: A URL returned by search_web.
            start: Character offset to start reading from (0 = the beginning).
        """
        if start < 0:
            raise ToolFailure("start must be >= 0")
        try:
            if url not in cache:
                page = fetcher.get(url)
                cache[page.url] = page.text
                cache[url] = page.text
        except FetchError as exc:
            raise ToolFailure(str(exc)) from None
        text = cache[url]
        src = log.register(url, text)
        body = text[start : start + PAGE_CHARS]
        end = start + len(body)
        more = (
            f"\n[characters {start}-{end} of {len(text)}; call again with start={end} to continue]"
            if end < len(text)
            else f"\n[end of page: {len(text)} characters]"
        )
        if not body:
            return f"Source [{src.n}] {src.title}: nothing at start={start} (the page has {len(text)} characters)."
        return f"Source [{src.n}] {src.title} ({url})\n{body}{more}"

    return [search_web, fetch_page]
