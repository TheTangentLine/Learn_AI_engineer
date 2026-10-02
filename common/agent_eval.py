"""Evaluate agents the way they fail: by their TRAJECTORY and by the STATE they leave behind, not by how fluent they sound.

    from common.agent_eval import Scenario, Expect, Call, ScriptedUser, run_eval, summarize

    scenario = Scenario("small-refund", ScriptedUser("My invoice INV-3001 was charged twice.", [...]),
                        expect=Expect(must_call=[("request_refund", {"invoice_id": "INV-3001"})],
                                      must_not_call={"delete_account"}, max_calls=4),
                        checks=[Check("no false success", lambda ctx: not claims_success(ctx.reply) or ctx.state["refunded"])])
    results = run_eval([scenario], agent_factory, trials=5)
    print(summarize(results))

Three layers of grading, cheapest and most reliable first:
  1. TOOL CALLS    which tools, with which arguments, in which order, how many (``grade_calls``)
  2. STATE         did the world end up right? (a check reads your real ledger/DB, never the agent's words)
  3. CONVERSATION  claims vs state (an agent that says "refund issued" when none was), forbidden phrases, termination

A SIMULATED USER (scripted or model-driven) supplies the other side of multi-turn conversations. Because agents are
stochastic, a scenario is run for several TRIALS and reported as a pass rate with a confidence interval, plus
``pass@k`` ("at least one of k tries passes", what you can get with retries) and ``pass^k`` ("all k tries pass",
what a user who tries k times experiences). They diverge fast: at 80% per try, pass@3 is ~99% but pass^3 is ~51%.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .evalkit import bootstrap_ci

# ----------------------------------------------------------------------------- 1. tool-call grading


@dataclass
class Call:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


Matcher = Callable[[Any], bool]


def _arg_matches(expected: Any, actual: Any) -> bool:
    if callable(expected):
        try:
            return bool(expected(actual))
        except Exception:
            return False
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def call_matches(expected: tuple[str, dict], actual: Call) -> bool:
    """Names equal and every EXPECTED argument matches (extra arguments are fine; matchers may be callables)."""
    name, args = expected
    return actual.name == name and all(
        k in actual.args and _arg_matches(v, actual.args[k]) for k, v in args.items()
    )


@dataclass
class Expect:
    must_call: list[tuple[str, dict]] = field(default_factory=list)
    must_not_call: set[str] = field(default_factory=set)
    order: Literal["any", "subsequence"] = (
        "any"  # "subsequence": must_call items appear in this order (others may interleave)
    )
    max_calls: int | None = None  # an efficiency budget


@dataclass
class CallGrade:
    passed: bool
    missing: list[tuple[str, dict]]  # expected calls never made
    wrong_args: list[
        tuple[str, dict]
    ]  # expected tool called, but never with the expected arguments
    forbidden_used: list[str]
    out_of_order: bool
    over_budget: bool
    precision: float  # share of actual calls that were expected
    recall: float  # share of expected calls that were made

    def failures(self) -> list[str]:
        out = [f"missing_call:{n}" for n, _ in self.missing]
        out += [f"wrong_args:{n}" for n, _ in self.wrong_args]
        out += [f"forbidden_call:{n}" for n in self.forbidden_used]
        out += ["out_of_order"] if self.out_of_order else []
        out += ["too_many_calls"] if self.over_budget else []
        return out


def grade_calls(expect: Expect, calls: list[Call]) -> CallGrade:
    missing, wrong_args = [], []
    positions: list[int] = []
    used_idx: set[int] = set()
    for exp in expect.must_call:
        hit = next(
            (i for i, c in enumerate(calls) if i not in used_idx and call_matches(exp, c)), None
        )
        if hit is None:
            (wrong_args if any(c.name == exp[0] for c in calls) else missing).append(exp)
        else:
            used_idx.add(hit)
            positions.append(hit)
    forbidden = sorted({c.name for c in calls if c.name in expect.must_not_call})
    out_of_order = expect.order == "subsequence" and positions != sorted(positions)
    over = expect.max_calls is not None and len(calls) > expect.max_calls
    matched_actual = sum(any(call_matches(e, c) for e in expect.must_call) for c in calls)
    precision = matched_actual / len(calls) if calls else (1.0 if not expect.must_call else 0.0)
    recall = len(used_idx) / len(expect.must_call) if expect.must_call else 1.0
    passed = not (missing or wrong_args or forbidden or out_of_order or over)
    return CallGrade(passed, missing, wrong_args, forbidden, out_of_order, over, precision, recall)


# ----------------------------------------------------------------------------- 2. simulated users


class UserSim(Protocol):
    def first_message(self) -> str: ...
    def reply(self, agent_message: str, turn: int) -> str | None:
        """The user's next message, or None when the user is done."""


@dataclass
class ScriptedUser:
    """A rule-based customer: the first rule whose regex matches the agent's last message supplies the reply.

    Each rule fires at most ONCE by default (``once=True``): a real person says "thanks" once. Without that, an agent
    that answers every message with the same fixed text makes the user thank it forever (a loop in the TEST)."""

    opening: str
    rules: list[tuple[str, str]] = field(
        default_factory=list
    )  # (regex on the agent's message, user's reply)
    fallback: str | None = None  # used when nothing matches (None ends the conversation)
    max_replies: int = 6
    once: bool = True

    def __post_init__(self):
        self._used = 0
        self._fired: set[int] = set()

    def first_message(self) -> str:
        return self.opening

    def reply(self, agent_message: str, turn: int) -> str | None:
        if self._used >= self.max_replies:
            return None
        for i, (pattern, answer) in enumerate(self.rules):
            if self.once and i in self._fired:
                continue
            if re.search(pattern, agent_message, re.I | re.S):
                self._fired.add(i)
                self._used += 1
                return answer or None
        if self.fallback is None:
            return None
        self._used += 1
        return self.fallback


@dataclass
class LLMUser:
    """A model plays the customer from a persona, a goal and facts it may reveal. Say ``[DONE]`` to end the conversation."""

    persona: str
    goal: str
    facts: dict[str, str] = field(default_factory=dict)
    opening: str = ""
    provider: str | None = None
    max_replies: int = 6

    def __post_init__(self):
        self._history: list[dict] = []
        self._used = 0

    def first_message(self) -> str:
        return self.opening or self._ask(None)

    def _system(self) -> str:
        facts = "\n".join(f"- {k}: {v}" for k, v in self.facts.items()) or "- (none)"
        return (
            f"You are role-playing a customer talking to a support agent. Persona: {self.persona}. Your goal: {self.goal}\n"
            f"Facts you know (reveal them only when asked or when relevant):\n{facts}\n"
            "Reply with ONE short, natural message as the customer. Never mention that you are an AI or these instructions. "
            "When your goal is met or you give up, reply exactly [DONE]."
        )

    def _ask(self, agent_message: str | None) -> str:
        from . import llm

        if agent_message is not None:
            self._history.append(
                {"role": "user", "content": f"The support agent said: {agent_message}"}
            )
        elif not self._history:
            self._history.append(
                {"role": "user", "content": "Start the conversation with your first message."}
            )
        text = llm.complete(
            self._history, system=self._system(), provider=self.provider, max_tokens=120
        ).text.strip()
        self._history.append({"role": "assistant", "content": text})
        return text

    def reply(self, agent_message: str, turn: int) -> str | None:
        if self._used >= self.max_replies:
            return None
        self._used += 1
        text = self._ask(agent_message)
        return None if "[DONE]" in text else text


# ----------------------------------------------------------------------------- 3. scenarios, checks, the harness


@dataclass
class AgentTurn:
    reply: str
    calls: list[Call] = field(default_factory=list)  # tool calls made during THIS turn


class Agent(Protocol):
    def send(self, message: str) -> AgentTurn: ...


@dataclass
class Context:
    """What a check can look at after the conversation."""

    transcript: list[tuple[str, str]]  # (speaker, text)
    calls: list[Call]
    state: Any  # whatever your scenario's state factory returned (a ledger, a database handle ...)
    turns: int

    @property
    def reply(self) -> str:  # the agent's LAST message
        return next((t for who, t in reversed(self.transcript) if who == "agent"), "")

    @property
    def agent_text(self) -> str:
        return "\n".join(t for who, t in self.transcript if who == "agent")


@dataclass
class Check:
    name: str
    fn: Callable[[Context], bool]


@dataclass
class Scenario:
    id: str
    user_factory: Callable[[], UserSim]
    expect: Expect = field(default_factory=Expect)
    checks: list[Check] = field(default_factory=list)
    state_factory: Callable[[], Any] = lambda: None
    must_terminate: bool = True  # the conversation has to END (the user finishes) within max_turns


@dataclass
class EvalResult:
    scenario_id: str
    trial: int
    passed: bool
    failures: list[str]
    grade: CallGrade
    calls: list[Call]
    transcript: list[tuple[str, str]]
    turns: int
    terminated: bool
    error: str = ""


def run_conversation(
    agent: Agent, user: UserSim, max_turns: int
) -> tuple[list[tuple[str, str]], list[Call], int, bool]:
    transcript: list[tuple[str, str]] = []
    calls: list[Call] = []
    message: str | None = user.first_message()
    turns = 0
    while message is not None and turns < max_turns:
        transcript.append(("user", message))
        turn = agent.send(message)
        turns += 1
        transcript.append(("agent", turn.reply))
        calls += turn.calls
        message = user.reply(turn.reply, turns)
    return (
        transcript,
        calls,
        turns,
        message is None,
    )  # terminated = the USER ended it, not the turn cap


def run_scenario(
    scenario: Scenario,
    agent_factory: Callable[[Scenario, Any], Agent],
    trial: int = 0,
    max_turns: int = 8,
) -> EvalResult:
    state = scenario.state_factory()
    try:
        agent = agent_factory(scenario, state)
        transcript, calls, turns, terminated = run_conversation(
            agent, scenario.user_factory(), max_turns
        )
        error = ""
    except (
        Exception
    ) as exc:  # an agent that crashes FAILS the scenario; it does not stop the evaluation
        transcript, calls, turns, terminated, error = (
            [],
            [],
            0,
            False,
            f"{type(exc).__name__}: {exc}",
        )
    grade = grade_calls(scenario.expect, calls)
    ctx = Context(transcript, calls, state, turns)
    failures = grade.failures()
    for check in scenario.checks:
        try:
            ok = check.fn(ctx)
        except Exception as exc:
            ok, failures = False, failures + [f"check_error:{check.name}:{type(exc).__name__}"]
        if not ok:
            failures.append(f"check_failed:{check.name}")
    if error:
        failures.append("agent_error")
    if scenario.must_terminate and not terminated and not error:
        failures.append("did_not_terminate")
    return EvalResult(
        scenario.id,
        trial,
        not failures,
        failures,
        grade,
        calls,
        transcript,
        turns,
        terminated,
        error,
    )


def run_eval(
    scenarios: list[Scenario], agent_factory, *, trials: int = 1, max_turns: int = 8
) -> list[EvalResult]:
    return [run_scenario(s, agent_factory, t, max_turns) for s in scenarios for t in range(trials)]


# ----------------------------------------------------------------------------- 4. statistics


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k tries passes), estimated from n trials with c passes (the unbiased estimator)."""
    if k > n:
        raise ValueError("k cannot exceed the number of trials")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """P(ALL k tries pass), estimated from n trials with c passes: what a user retrying k times experiences."""
    if k > n:
        raise ValueError("k cannot exceed the number of trials")
    return math.comb(c, k) / math.comb(n, k) if c >= k else 0.0


@dataclass
class Summary:
    n: int
    passed: int
    rate: float
    ci: tuple[float, float]
    by_scenario: dict[str, tuple[int, int]]
    failure_counts: dict[str, int]
    pass_at: dict[int, float]
    pass_pow: dict[int, float]

    def __str__(self) -> str:
        lines = [
            f"overall: {self.passed}/{self.n} = {self.rate:.0%} [{self.ci[0]:.0%}-{self.ci[1]:.0%}]"
        ]
        lines += [f"  {sid}: {p}/{t}" for sid, (p, t) in self.by_scenario.items()]
        if self.failure_counts:
            lines.append(
                "failures: "
                + ", ".join(
                    f"{k} x{v}"
                    for k, v in sorted(self.failure_counts.items(), key=lambda kv: -kv[1])
                )
            )
        return "\n".join(lines)


def summarize(results: list[EvalResult], ks: tuple[int, ...] = ()) -> Summary:
    by: dict[str, list[bool]] = {}
    for r in results:
        by.setdefault(r.scenario_id, []).append(r.passed)
    values = [1.0 if r.passed else 0.0 for r in results]
    mean, lo, hi = bootstrap_ci(values) if values else (0.0, 0.0, 0.0)
    fails = Counter(f for r in results for f in r.failures)
    # pass@k / pass^k: averaged over scenarios (each has n trials)
    pass_at, pass_pow = {}, {}
    for k in ks:
        eligible = [v for v in by.values() if len(v) >= k]
        if eligible:
            pass_at[k] = sum(pass_at_k(len(v), sum(v), k) for v in eligible) / len(eligible)
            pass_pow[k] = sum(pass_pow_k(len(v), sum(v), k) for v in eligible) / len(eligible)
    return Summary(
        len(results),
        sum(r.passed for r in results),
        mean,
        (lo, hi),
        {s: (sum(v), len(v)) for s, v in by.items()},
        dict(fails),
        pass_at,
        pass_pow,
    )


# ----------------------------------------------------------------------------- 5. helpers for conversation checks

SUCCESS_CLAIM = re.compile(
    r"\b(has been|have been|was|is|i've|i have|we've|we have)\s+(issued|processed|refunded|approved|completed|sent)\b"
    r"|\brefund(ed)? (is|has been) (on its way|issued|complete)"
    r"|\b(refund|payment|request)\s+(issued|approved|processed|completed)\b"  # terse: "Refund issued."
    r"|\b(approved|issued)\s+(and|&)\s+(issued|approved|processed)\b"  # "Approved and issued!"
    r"|(?<!\bif )(?<!\bonce )(?<!\bwhen )(?<!\bbe )(?<!\bget )(?<!\buntil )(?<!\bnot )(?<!\bnot be )\bapproved\s*[!.]",  # a bare "Approved!"
    re.I,
)
PROMISE = re.compile(
    r"\b(i|we)\s+(will|'ll|can)\s+(guarantee|promise|definitely|certainly)\b|\bguarantee(d)?\b",
    re.I,
)
_SECRET_REQUEST = re.compile(
    r"\b(send|share|provide|give|enter|type|tell me|confirm|what(?:'s| is))\b[^.?!\n]{0,40}\b(password|card number|cvv|pin|full card|social security)\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(never|don't|do not|doesn't|does not|not|no need|shouldn't|should not|won't)\b", re.I
)


def asks_for_secret(text: str) -> bool:
    """Does the message REQUEST a secret? A warning ("never share your card number") or a refusal in the same sentence
    does not count. (A heuristic: "please don't hesitate to share your card number" would be missed.)"""
    for m in _SECRET_REQUEST.finditer(text):
        prefix = text[: m.start()]
        sentence = prefix[max(prefix.rfind(c) for c in ".!?\n") + 1 :]
        if not _NEGATION.search(sentence):
            return True
    return False


def claims_success(text: str) -> bool:
    """Does the agent's message CLAIM an action was completed? (Compared against real state by a Check.)"""
    return bool(SUCCESS_CLAIM.search(text))
