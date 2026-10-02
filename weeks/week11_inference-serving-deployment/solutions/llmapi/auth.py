"""API keys: generated once, stored only as SHA-256 hashes, compared in constant time. A key identifies a USER, and the user carries the limits."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: str
    rpm: int = 60  # sustained requests per minute
    burst: int = 10  # how many requests may arrive at once (the bucket's capacity)
    daily_tokens: int = 200_000  # prompt + completion tokens per UTC day
    max_concurrent: int = 4  # requests in flight at once
    max_tokens_cap: int = 1024  # the largest ``max_tokens`` this user may ask for


def make_key(prefix: str = "sk-local") -> str:
    """A new random key (shown to the user ONCE; only its hash is stored)."""
    return f"{prefix}-{secrets.token_urlsafe(24)}"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeyStore:
    """key hash -> User. ``authenticate`` compares the hash of the presented key with EVERY stored hash in constant time per comparison, so response time does not reveal
    how much of a key was right or which user it belongs to."""

    def __init__(self) -> None:
        self._by_hash: dict[str, User] = {}

    def add(self, key: str, user: User) -> None:
        self._by_hash[hash_key(key)] = user

    def authenticate(self, presented: str | None) -> User | None:
        if not presented:
            return None
        digest, found = hash_key(presented), None
        for stored, user in self._by_hash.items():
            if hmac.compare_digest(stored, digest):
                found = user
        return found

    def __len__(self) -> int:
        return len(self._by_hash)


def bearer_token(header: str | None) -> str | None:
    """'Bearer abc' -> 'abc' (case-insensitive scheme); anything else -> None."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def load_keys(spec: list[dict]) -> ApiKeyStore:
    """Build a store from a JSON-style list: ``{"user": "alice", "key_sha256": "<hex>", "rpm": 60, ...}``. A plaintext ``"key"`` is also accepted for local demos
    and tests, but a deployment should ship only hashes (a leaked config then leaks nothing usable)."""
    store = ApiKeyStore()
    allowed = {"rpm", "burst", "daily_tokens", "max_concurrent", "max_tokens_cap"}
    for entry in spec:
        unknown = set(entry) - allowed - {"user", "key", "key_sha256"}
        if unknown:
            raise ValueError(f"unknown field(s) in a key entry: {sorted(unknown)}")
        if ("key" in entry) == ("key_sha256" in entry):
            raise ValueError("each entry needs exactly one of 'key' or 'key_sha256'")
        user = User(entry["user"], **{k: entry[k] for k in allowed if k in entry})
        store._by_hash[entry["key_sha256"] if "key_sha256" in entry else hash_key(entry["key"])] = (
            user
        )
    return store
