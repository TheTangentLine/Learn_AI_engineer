"""Per-user rate limiting: a token bucket for request rate, a daily token quota, a concurrency cap. The clock is injected so every behaviour is testable without sleeping."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .auth import User


class TokenBucket:
    """Capacity ``burst``, refilled at ``rate`` tokens per second. ``acquire`` takes one token if there is one, else says how long until there will be."""

    def __init__(self, rate: float, burst: float, clock: Callable[[], float] = time.monotonic):
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be positive and burst at least 1")
        self.rate, self.burst, self.clock = rate, float(burst), clock
        self.tokens, self.updated = float(burst), clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def acquire(self, cost: float = 1.0) -> tuple[bool, float]:
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        return False, (cost - self.tokens) / self.rate


@dataclass
class Decision:
    allowed: bool
    code: str = ""  # rate_limited | quota_exceeded | too_many_concurrent
    retry_after: float = 0.0


@dataclass
class _State:
    bucket: TokenBucket
    day: int = -1
    tokens_today: int = 0
    in_flight: int = 0
    requests: int = 0
    rejected: int = 0


@dataclass
class Limiter:
    """Everything a request must pass before it reaches the model, per user. ``wall_clock`` (seconds since the epoch) decides which UTC day the quota belongs to."""

    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    _states: dict[str, _State] = field(default_factory=dict)

    def _state(self, user: User) -> _State:
        st = self._states.get(user.id)
        if st is None:
            st = self._states[user.id] = _State(TokenBucket(user.rpm / 60, user.burst, self.clock))
        today = int(self.wall_clock() // 86400)
        if st.day != today:
            st.day, st.tokens_today = today, 0
        return st

    def admit(self, user: User) -> Decision:
        """Check in order of cost to the user: quota (their budget), concurrency, then the request rate. Admitting counts the request as in flight."""
        st = self._state(user)
        if st.tokens_today >= user.daily_tokens:
            st.rejected += 1
            return Decision(False, "quota_exceeded", 86400 - self.wall_clock() % 86400)
        if st.in_flight >= user.max_concurrent:
            st.rejected += 1
            return Decision(False, "too_many_concurrent", 1.0)
        ok, wait = st.bucket.acquire()
        if not ok:
            st.rejected += 1
            return Decision(False, "rate_limited", wait)
        st.in_flight += 1
        st.requests += 1
        return Decision(True)

    def release(self, user: User, tokens_used: int) -> None:
        """Called when the request ends, whatever happened: frees the concurrency slot and charges the tokens actually generated and read."""
        st = self._state(user)
        st.in_flight = max(0, st.in_flight - 1)
        st.tokens_today += max(0, tokens_used)

    def usage(self, user: User) -> dict[str, int]:
        st = self._state(user)
        return {
            "tokens_today": st.tokens_today,
            "daily_tokens": user.daily_tokens,
            "in_flight": st.in_flight,
            "requests": st.requests,
            "rejected": st.rejected,
        }
