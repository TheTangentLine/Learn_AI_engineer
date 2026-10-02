"""A tool-permission policy for agents: default-deny, argument rules, data-loss prevention on tool arguments, an egress allowlist,
taint tracking, confirmations that fail closed, and an audit log.

    from common import policy

    engine = policy.PolicyEngine(
        [policy.ToolRule("search_web"), policy.ToolRule("fetch_page"),
         policy.ToolRule("create_note", mode="confirm", args=(policy.ArgRule("content", "max_len", 2000),))],
        secrets=[user_project_code], egress_hosts={"docs.acme.example"}, confirmer=ask_a_human,
    )
    safe_registry = engine.guard(registry)         # the agent loop uses it exactly like the original

Why this exists: the model is not a security boundary. A hijacked model can ask for ANYTHING, so every permission must be decided by
code that does not read the model's reasons. What the policy enforces, in order, for each call the model makes:
  1. the tool is allowed at all (default deny), and the task's capability set includes it;
  2. the call count is within the tool's limit;
  3. the arguments pass the tool's rules (length, pattern, choices, path, host);
  4. no argument carries a protected value (DLP), and any URL in an argument points at an allowed host (egress);
  5. an argument that copies text from untrusted content is not passed to a tool that must not receive it (taint);
  6. a "confirm" tool is approved by a human who sees the real arguments; with no confirmer the answer is no.
The model is told WHY a call was refused (so a legitimate agent can recover) without being told how the check works.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from . import guard
from .chat import ToolCall
from .tools import ToolRegistry, ToolResult

URL_IN_TEXT = re.compile(r"(?:https?:)?//[^\s<>()\[\]\"'`]+", re.I)


def _host_matches(host: str, allowed: Iterable[str]) -> bool:
    """A host matches an allowlist entry exactly, or (for DNS names only) as a subdomain of it. An IP address matches only if that exact
    address is listed: ``10.0.0.5`` listed allows ``10.0.0.5``, never a subdomain trick and never a neighbouring address."""
    if guard.is_ip_host(host):
        return host in set(allowed)
    return any(host == h or host.endswith("." + h) for h in allowed)


# ----------------------------------------------------------------------------- taint tracking


class TaintTracker:
    """Remembers where text came from. ``observe(text, trusted=False)`` records content the model read from an untrusted source
    (a web page, a document, another tool's output); ``observe(text, trusted=True)`` records what the USER said. A value is tainted if
    it shares a run of ``min_len`` characters with untrusted content that the user did not also supply: the model is copying what an
    attacker wrote. This is an approximation of data-flow tracking (a paraphrase escapes it) that costs nothing and catches the
    copy-paste exfiltration and the copied-payload attacks."""

    def __init__(self, min_len: int = 24):
        self.k = min_len
        self._untrusted: dict[str, str] = {}  # shingle -> source label
        self._trusted: set[str] = set()

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", guard.normalize(text).folded).strip()

    def _shingles(self, text: str) -> set[str]:
        t = self._norm(text)
        return {t[i : i + self.k] for i in range(max(len(t) - self.k + 1, 0))}

    def observe(self, text: str, *, trusted: bool, source: str = "") -> None:
        sh = self._shingles(text)
        if trusted:
            self._trusted |= sh
        else:
            for s in sh:
                self._untrusted.setdefault(s, source or "untrusted content")

    def taint_of(self, value: Any) -> str | None:
        """The source label if ``value`` copies untrusted text the user did not provide, else None."""
        for s in self._shingles(value if isinstance(value, str) else str(value)):
            if s in self._untrusted and s not in self._trusted:
                return self._untrusted[s]
        return None


# ----------------------------------------------------------------------------- rules


@dataclass(frozen=True)
class ArgRule:
    arg: str
    kind: str  # max_len | regex | enum | path_under | host_allow | no_injection | min_len
    value: Any = None
    message: str = ""

    def violation(self, v: Any) -> str | None:
        s = v if isinstance(v, str) else str(v)
        msg = self.message
        if self.kind == "max_len" and len(s) > self.value:
            return msg or f"{self.arg} is too long ({len(s)} > {self.value} characters)"
        if self.kind == "min_len" and len(s) < self.value:
            return msg or f"{self.arg} is too short"
        if self.kind == "regex" and not re.fullmatch(self.value, s):
            return msg or f"{self.arg} has an invalid format"
        if self.kind == "enum" and v not in self.value:
            return msg or f"{self.arg} must be one of {sorted(map(str, self.value))}"
        if self.kind == "path_under":
            p = PurePosixPath(s)
            if p.is_absolute() or ".." in p.parts or str(p).startswith("~"):
                return msg or f"{self.arg} must be a relative path inside {self.value}"
        if self.kind == "host_allow":
            host = guard.host_of(s)
            if host is None or not _host_matches(host, self.value):
                return msg or f"{self.arg} must point to one of {sorted(self.value)}"
        if self.kind == "no_injection" and guard.detect(s).flagged:
            return msg or f"{self.arg} contains text that looks like instructions to an AI"
        return None


@dataclass(frozen=True)
class ToolRule:
    name: str
    mode: str = "allow"  # allow | confirm | deny
    args: tuple[ArgRule, ...] = ()
    max_calls: int | None = None
    taint_sensitive: bool = False  # refuse arguments that copy untrusted text


@dataclass
class Decision:
    tool: str
    action: str  # allow | deny | confirm-approved | confirm-denied
    reason: str = ""
    args_digest: str = (
        ""  # a hash of the arguments: the audit log must not become a second copy of the data
    )

    @property
    def allowed(self) -> bool:
        return self.action in ("allow", "confirm-approved")


def digest(args: Mapping[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(args.items())).encode()).hexdigest()[:12]


class PolicyEngine:
    def __init__(
        self,
        rules: Sequence[ToolRule],
        *,
        secrets: Iterable[str] = (),
        egress_hosts: Iterable[str] | None = None,
        taint: TaintTracker | None = None,
        confirmer: Callable[[ToolCall], bool] | None = None,
        default: str = "deny",
    ):
        self.rules = {r.name: r for r in rules}
        self.secrets = [s for s in secrets if s]
        self.egress = None if egress_hosts is None else {h.lower() for h in egress_hosts}
        self.taint, self.confirmer, self.default = taint, confirmer, default
        self.calls: dict[str, int] = {}
        self.audit: list[Decision] = []

    # -- the decision
    def check(self, call: ToolCall) -> Decision:
        d = self._decide(call)
        self.audit.append(d)
        return d

    def _decide(self, call: ToolCall) -> Decision:
        dig = digest(call.args)
        rule = self.rules.get(call.name)
        if rule is None:
            if self.default == "allow":
                rule = ToolRule(call.name)
            else:
                return Decision(
                    call.name, "deny", f"the tool {call.name!r} is not available for this task", dig
                )
        if rule.mode == "deny":
            return Decision(
                call.name, "deny", f"the tool {call.name!r} is not available for this task", dig
            )
        if rule.max_calls is not None and self.calls.get(call.name, 0) >= rule.max_calls:
            return Decision(
                call.name,
                "deny",
                f"{call.name} has already been used the maximum {rule.max_calls} times",
                dig,
            )
        for ar in rule.args:
            if ar.arg in call.args and (why := ar.violation(call.args[ar.arg])):
                return Decision(call.name, "deny", why, dig)
        for key, value in call.args.items():
            for s in self.secrets:
                if guard.contains_secret(str(value), s):
                    return Decision(
                        call.name,
                        "deny",
                        f"argument {key!r} contains a protected value that must not leave the system",
                        dig,
                    )
            if self.egress is not None:
                for url in URL_IN_TEXT.findall(str(value)):
                    host = guard.host_of(url)
                    if host is None or not _host_matches(host, self.egress):
                        return Decision(
                            call.name,
                            "deny",
                            f"argument {key!r} contains a URL outside the allowed hosts",
                            dig,
                        )
            if (
                rule.taint_sensitive
                and self.taint is not None
                and (src := self.taint.taint_of(value))
            ):
                return Decision(
                    call.name,
                    "deny",
                    f"argument {key!r} copies text from {src}, which the user did not provide",
                    dig,
                )
        if rule.mode == "confirm":
            if self.confirmer is None:
                return Decision(
                    call.name,
                    "deny",
                    "this action needs human confirmation and none is available",
                    dig,
                )
            ok = bool(self.confirmer(call))
            return Decision(
                call.name,
                "confirm-approved" if ok else "confirm-denied",
                "" if ok else "the human declined",
                dig,
            )
        self.calls[call.name] = self.calls.get(call.name, 0) + 1
        return Decision(call.name, "allow", "", dig)

    # -- wiring
    def guard(
        self, registry: ToolRegistry, *, result_filter: Callable[[str], str] | None = None
    ) -> GuardedRegistry:
        return GuardedRegistry(registry, self, result_filter=result_filter)

    def allowed_tools(self, names: Iterable[str]) -> list[str]:
        """Which of these tools the model may be shown: those with a rule that is not 'deny' (or, with default 'allow', without one)."""
        if self.default == "deny":
            return [n for n in names if n in self.rules and self.rules[n].mode != "deny"]
        return [n for n in names if self.rules.get(n, ToolRule(n)).mode != "deny"]


class GuardedRegistry(ToolRegistry):
    """A registry whose every call goes through the policy. The model only SEES the tools the policy allows, and a refused call
    returns an error it can read; the underlying tool is never invoked. Untrusted tool results are fed to the taint tracker."""

    def __init__(
        self,
        inner: ToolRegistry,
        engine: PolicyEngine,
        *,
        untrusted_tools: Iterable[str] = ("fetch_page", "search_web", "read_note", "search_notes"),
        result_filter: Callable[[str], str] | None = None,
    ):
        super().__init__(
            [
                t
                for t in inner.tools()
                if t.name in engine.allowed_tools([x.name for x in inner.tools()])
            ],
            compact=getattr(inner, "_compact", False),
        )
        self._inner, self.engine, self.untrusted_tools = inner, engine, set(untrusted_tools)
        self.result_filter = result_filter  # applied to what UNTRUSTED tools return, before the model and the taint tracker see it
        self.filtered = 0  # how many results the filter changed

    def _rewrap(self, inner: ToolRegistry) -> GuardedRegistry:
        return GuardedRegistry(
            inner,
            self.engine,
            untrusted_tools=self.untrusted_tools,
            result_filter=self.result_filter,
        )

    def compact(
        self,
    ) -> GuardedRegistry:  # the inherited versions would silently return an UNGUARDED registry
        return self._rewrap(self._inner.compact())

    def without(self, *names: str) -> GuardedRegistry:
        return self._rewrap(self._inner.without(*names))

    def execute(self, call: ToolCall) -> ToolResult:
        d = self.engine.check(call)
        if not d.allowed:
            return ToolResult(
                call.id,
                call.name,
                f"Blocked by policy: {d.reason}. Do not retry this call; continue the task without it.",
                True,
            )
        res = super().execute(call)
        if (
            self.result_filter is not None
            and call.name in self.untrusted_tools
            and not res.is_error
        ):
            cleaned = self.result_filter(res.content)
            if cleaned != res.content:
                self.filtered += 1
                res = ToolResult(res.call_id, res.name, cleaned, res.is_error, res.seconds)
        if self.engine.taint is not None and call.name in self.untrusted_tools and not res.is_error:
            self.engine.taint.observe(
                res.content, trusted=False, source=f"the result of {call.name}"
            )
        return res


def capabilities_for_task(
    task: str, *, base: Iterable[str] = ("search_web", "fetch_page")
) -> set[str]:
    """Least privilege by task: reading tools always, note tools only when the user asked to save or recall something."""
    caps = set(base)
    if re.search(
        r"\b(?:save|note|notes|remember|write\s+down|record|bookmark|recall)\b", task, re.I
    ):
        caps |= {"create_note", "append_to_note", "read_note", "search_notes", "list_notes"}
    return caps
