"""A fetcher that treats every URL as hostile: strict allowlist, no surprise redirects, size and time caps.

The URL comes from a MODEL (which may have read a poisoned page), so this is an SSRF boundary: the model must not
be able to reach cloud metadata endpoints (169.254.169.254), other local services, files, or arbitrary hosts.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse


class FetchError(Exception):
    """A refusal or failure whose message tells the model what to do instead."""


@dataclass
class Page:
    url: str
    text: str
    truncated: bool


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None  # we follow redirects ourselves, re-checking the allowlist on every hop


class Fetcher:
    def __init__(
        self,
        allowed_hosts: set[str],
        *,
        max_bytes: int = 300_000,
        timeout_s: float = 5.0,
        max_redirects: int = 3,
    ):
        self.allowed = {h.lower() for h in allowed_hosts}
        self.max_bytes, self.timeout_s, self.max_redirects = max_bytes, timeout_s, max_redirects
        self._opener = urllib.request.build_opener(_NoRedirect)

    def check(self, url: str) -> str:
        try:
            p = urlparse(url)
            port = p.port  # raises ValueError for 'host:12x'
            host = (p.hostname or "").lower()
        except ValueError:
            raise FetchError(f"malformed URL {url!r}. Use a URL returned by search_web.") from None
        if p.scheme not in ("http", "https"):
            raise FetchError(
                f"only http(s) URLs can be fetched, not {p.scheme or 'a bare path'!r}. Use a URL from search_web."
            )
        if p.username or p.password or "@" in p.netloc:
            raise FetchError("URLs with credentials ('user@host') are not allowed.")
        netloc = f"{host}:{port}" if port else host
        if netloc not in self.allowed:
            raise FetchError(
                f"host {netloc!r} is not on the allowlist. Only URLs returned by search_web can be fetched."
            )
        return url

    def get(self, url: str) -> Page:
        for _ in range(self.max_redirects + 1):
            self.check(url)
            req = urllib.request.Request(
                url, headers={"User-Agent": "research-agent/1.0", "Accept": "text/*"}
            )
            try:
                resp = self._opener.open(req, timeout=self.timeout_s)
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    location = exc.headers.get("Location", "")
                    url = urljoin(url, location)  # re-validated at the top of the loop
                    continue
                raise FetchError(
                    f"the server returned HTTP {exc.code} for {url}. Check the URL or search again."
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise FetchError(f"could not fetch {url}: {getattr(exc, 'reason', exc)}") from None
            with resp:
                ctype = resp.headers.get_content_type()
                if not ctype.startswith("text/") and ctype != "application/json":
                    raise FetchError(
                        f"content type {ctype!r} is not text; only text pages can be read."
                    )
                data = resp.read(self.max_bytes + 1)
            return Page(
                url,
                data[: self.max_bytes].decode("utf-8", errors="replace"),
                len(data) > self.max_bytes,
            )
        raise FetchError(f"too many redirects (more than {self.max_redirects}).")
