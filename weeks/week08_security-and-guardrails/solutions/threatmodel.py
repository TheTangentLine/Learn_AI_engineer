"""A threat model you can CHECK: threats cite the controls that mitigate them, controls cite code and tests, and a validator
confirms the citations are real and every tool the system exposes appears in some threat.

    model = RESEARCH_AGENT
    problems = validate(model)           # [] when the model is internally consistent and matches the code
    print(render_markdown(model))        # the risk table, highest risk first

Why: a threat model in a wiki rots. This one fails a test when a tool is added without a threat, when a cited control or test is
renamed or deleted, when a "mitigated" threat has no evidence, or when an "accepted" risk has no stated reason.

Evidence references (all resolved by parsing source files, nothing is imported or executed):
    code:<repo-relative path>#<Symbol>          a function, class or ``Class.method`` defined in that file
    test:<repo-relative path>::<test function>  a ``def test_...`` in that file
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

OWASP_LLM_2025 = {  # the OWASP Top 10 for LLM Applications as I know the 2025 edition: check the current list before relying on the ids
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}
STRIDE = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information disclosure",
    "D": "Denial of service",
    "E": "Elevation of privilege",
}
STATUSES = ("mitigated", "partial", "gap", "accepted")


@dataclass(frozen=True)
class Asset:
    id: str
    name: str
    why: str  # what harm follows if it is lost, leaked or corrupted


@dataclass(frozen=True)
class Control:
    id: str
    description: str
    evidence: tuple[str, ...]  # code:... and test:... references


@dataclass(frozen=True)
class Threat:
    id: str
    title: str
    owasp: str
    stride: str
    surface: tuple[str, ...]  # tools and entry points through which it happens
    assets: tuple[str, ...]
    likelihood: int  # 1 (rare) .. 5 (expected)
    impact: int  # 1 (annoying) .. 5 (severe)
    status: str  # mitigated | partial | gap | accepted
    controls: tuple[str, ...] = ()
    remaining: str = ""  # what is still exposed (required for partial and gap)
    reason: str = ""  # why the risk is accepted (required for accepted)

    @property
    def risk(self) -> int:
        return self.likelihood * self.impact


@dataclass
class Model:
    name: str
    description: str
    entry_points: tuple[str, ...]  # where untrusted data enters
    tools: tuple[str, ...]  # what the system can DO
    boundaries: tuple[str, ...]
    assets: tuple[Asset, ...]
    controls: tuple[Control, ...]
    threats: tuple[Threat, ...]
    tool_sources: tuple[
        str, ...
    ] = ()  # files whose @tool functions and tool-name sets must be covered
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- resolving evidence


def _defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()

    def walk(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(prefix + child.name)
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree)
    return names


def resolve(ref: str, root: Path = ROOT) -> str | None:
    """None if the reference resolves, otherwise a message saying why not."""
    if ref.startswith("code:"):
        path, _, symbol = ref[5:].partition("#")
        sep = "#"
    elif ref.startswith("test:"):
        path, _, symbol = ref[5:].partition("::")
        sep = "::"
    else:
        return f"{ref!r}: evidence must start with 'code:' or 'test:'"
    file = root / path
    if not file.is_file():
        return f"{ref}: no such file {path}"
    if not symbol:
        return f"{ref}: missing a symbol after {sep!r}"
    if symbol not in _defined_names(file):
        return f"{ref}: {symbol!r} is not defined in {path}"
    if ref.startswith("test:") and not symbol.split(".")[-1].startswith("test"):
        return f"{ref}: {symbol!r} is not a test function"
    return None


def tools_in_source(path: Path) -> set[str]:
    """Tool names a source file defines: functions decorated with ``@tool`` and string sets/lists named like ``*_TOOLS``."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                name = (
                    dec.func.id
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
                    else getattr(dec, "id", "")
                )
                if name == "tool":
                    found.add(node.name)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith("_TOOLS")
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, set | list | tuple) and all(isinstance(v, str) for v in value):
                found |= set(value)
    return found


# ----------------------------------------------------------------------------- validation


def validate(model: Model, root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    ids = [t.id for t in model.threats]
    if len(ids) != len(set(ids)):
        problems.append("duplicate threat ids")
    control_ids = {c.id for c in model.controls}
    if len(control_ids) != len(model.controls):
        problems.append("duplicate control ids")
    asset_ids = {a.id for a in model.assets}
    for c in model.controls:
        if not c.evidence:
            problems.append(f"control {c.id} cites no evidence")
        for ref in c.evidence:
            if (why := resolve(ref, root)) is not None:
                problems.append(f"control {c.id}: {why}")
        if not any(r.startswith("test:") for r in c.evidence):
            problems.append(f"control {c.id} has no test among its evidence")
    surface_covered: set[str] = set()
    for t in model.threats:
        surface_covered |= set(t.surface)
        if t.owasp not in OWASP_LLM_2025:
            problems.append(f"threat {t.id}: unknown OWASP id {t.owasp!r}")
        if t.stride not in STRIDE:
            problems.append(f"threat {t.id}: unknown STRIDE letter {t.stride!r}")
        if t.status not in STATUSES:
            problems.append(f"threat {t.id}: status must be one of {STATUSES}")
        if not (1 <= t.likelihood <= 5 and 1 <= t.impact <= 5):
            problems.append(f"threat {t.id}: likelihood and impact are 1-5")
        for a in t.assets:
            if a not in asset_ids:
                problems.append(f"threat {t.id}: unknown asset {a!r}")
        for c in t.controls:
            if c not in control_ids:
                problems.append(f"threat {t.id}: unknown control {c!r}")
        if t.status == "mitigated" and not t.controls:
            problems.append(f"threat {t.id} is 'mitigated' but cites no control")
        if t.status == "gap" and t.controls:
            problems.append(f"threat {t.id} is a 'gap' but cites controls: call it 'partial'")
        if t.status in ("partial", "gap") and not t.remaining.strip():
            problems.append(
                f"threat {t.id} is {t.status} but says nothing about what remains exposed"
            )
        if t.status == "partial" and not t.controls:
            problems.append(f"threat {t.id} is 'partial' but cites no control")
        if t.status == "accepted" and not t.reason.strip():
            problems.append(f"threat {t.id} is 'accepted' without a reason")
    used = {c for t in model.threats for c in t.controls}
    for c in control_ids - used:
        problems.append(f"control {c} mitigates no threat")
    for tool in model.tools:
        if tool not in surface_covered:
            problems.append(f"tool {tool!r} appears in no threat's surface")
    for entry in model.entry_points:
        if entry not in surface_covered:
            problems.append(f"entry point {entry!r} appears in no threat's surface")
    if model.tool_sources:
        actual: set[str] = set()
        for src in model.tool_sources:
            actual |= tools_in_source(root / src)
        if actual - set(model.tools):
            problems.append(
                f"the code defines tools the model does not list: {sorted(actual - set(model.tools))}"
            )
        if set(model.tools) - actual:
            problems.append(
                f"the model lists tools the code no longer defines: {sorted(set(model.tools) - actual)}"
            )
    return problems


def ranked(model: Model) -> list[Threat]:
    """Highest risk first; ties by status (gaps first) then id."""
    order = {"gap": 0, "partial": 1, "mitigated": 2, "accepted": 3}
    return sorted(model.threats, key=lambda t: (-t.risk, order[t.status], t.id))


def summary(model: Model) -> dict[str, int]:
    counts = {s: 0 for s in STATUSES}
    for t in model.threats:
        counts[t.status] += 1
    counts["total"] = len(model.threats)
    counts["open_risk"] = sum(t.risk for t in model.threats if t.status in ("gap", "partial"))
    return counts


def render_markdown(model: Model) -> str:
    lines = [
        f"# Threat model: {model.name}",
        "",
        model.description,
        "",
        "| risk | id | threat | OWASP | STRIDE | status | controls | what remains |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in ranked(model):
        lines.append(
            f"| {t.risk} | {t.id} | {t.title} | {t.owasp} {OWASP_LLM_2025[t.owasp]} | {STRIDE[t.stride]} | {t.status} | "
            + (", ".join(t.controls) or "none")
            + f" | {(t.remaining or t.reason or '').replace('|', '/')} |"
        )
    lines += ["", "## Controls and their evidence", ""]
    for c in model.controls:
        lines.append(f"- **{c.id}** {c.description}: " + "; ".join(f"`{e}`" for e in c.evidence))
    s = summary(model)
    lines += [
        "",
        f"{s['total']} threats: {s['mitigated']} mitigated, {s['partial']} partial, {s['gap']} gaps, {s['accepted']} accepted; open risk score {s['open_risk']}.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- the Week 5 research agent

RA = "weeks/week05_tool-use-and-agents/solutions/weekly"
RAT = f"{RA}/test_research_agent.py"
T5 = "weeks/week05_tool-use-and-agents/solutions/test_day4.py"

RESEARCH_AGENT = Model(
    name="the Week 5 research agent",
    description=(
        "A model-driven agent that searches a document collection, reads pages, optionally saves notes through an MCP server "
        "and writes a cited report. Untrusted text reaches the model through search results and page content; the model "
        "chooses URLs to fetch and what to write to notes and to the report."
    ),
    entry_points=(
        "user_question",
        "search_results",
        "page_content",
        "notes_store",
        "report_output",
    ),
    tools=(
        "search_web",
        "fetch_page",
        "create_note",
        "append_to_note",
        "read_note",
        "search_notes",
        "list_notes",
    ),
    tool_sources=(f"{RA}/research_agent/tools.py", f"{RA}/research_agent/agent.py"),
    boundaries=(
        "user -> agent (the question is trusted to be the user's, not safe)",
        "agent -> web (every page is attacker-controllable)",
        "agent -> MCP notes server (a local subprocess with write access to a folder)",
        "agent -> model provider (prompts leave the machine)",
        "report -> whatever renders it (a browser, a chat client, an email)",
    ),
    assets=(
        Asset(
            "A1",
            "the user's question and research topic",
            "may be confidential; it is sent to a provider and into tool arguments",
        ),
        Asset(
            "A2", "the notes folder", "persists across sessions: corrupting it corrupts later runs"
        ),
        Asset(
            "A3",
            "the local network and files behind the fetch tool",
            "an SSRF target: metadata endpoints, local services",
        ),
        Asset("A4", "the integrity of the report", "users act on it; citations must be real"),
        Asset("A5", "the spend and time budget", "an attacker or a bug can burn money"),
    ),
    controls=(
        Control(
            "C1",
            "fetch allowlist: exact host:port, no credentials in URLs, http(s) only, redirects re-checked on every hop",
            (
                f"code:{RA}/research_agent/fetch.py#Fetcher.check",
                f"test:{RAT}::test_fetcher_refuses_everything_off_the_allowlist",
                f"test:{RAT}::test_redirects_are_followed_only_within_the_allowlist_and_rechecked_on_every_hop",
            ),
        ),
        Control(
            "C2",
            "fetch size, time and content-type caps",
            (
                f"code:{RA}/research_agent/fetch.py#Fetcher.get",
                f"test:{RAT}::test_fetcher_rejects_non_text_enforces_size_and_time_and_reports_connection_errors",
            ),
        ),
        Control(
            "C3",
            "a poisoned page the model obeys still cannot reach forbidden hosts (end-to-end check)",
            (
                f"test:{RAT}::test_a_poisoned_page_that_the_model_obeys_still_cannot_reach_forbidden_hosts",
            ),
        ),
        Control(
            "C4",
            "citations are verified mechanically against the pages actually fetched; sources are built by the harness",
            (
                f"code:{RA}/research_agent/report.py#verify_report",
                f"code:{RA}/research_agent/report.py#render_final",
                f"test:{RAT}::test_verifier_classifies_each_failure_mode",
                f"test:{RAT}::test_render_final_builds_sources_from_what_was_really_fetched",
            ),
        ),
        Control(
            "C5",
            "step, cost and token budgets and repeated-call blocking in the agent loop",
            (
                "code:common/agent.py#_run_agent",
                "test:tests/test_agent.py::test_cost_and_token_budgets_stop_the_run_after_the_step_that_crossed_them",
                "test:tests/test_agent.py::test_repeated_identical_calls_are_blocked_with_the_earlier_result_in_the_message",
            ),
        ),
        Control(
            "C6",
            "the notes server confines names to a folder, refuses symlinks that escape it and enforces size and count limits",
            (
                f"test:{T5}::test_note_names_cannot_escape_or_be_weird",
                f"test:{T5}::test_symlinked_notes_pointing_outside_are_not_listed_or_readable",
                f"test:{T5}::test_create_refuses_to_overwrite_unless_asked_and_enforces_limits",
            ),
        ),
        Control(
            "C7",
            "the MCP bridge exposes only an allowlist of the server's tools",
            (
                "code:common/mcp_tools.py#McpBridge",
                f"test:{T5}::test_registry_allowlist_denylist_and_prefix",
            ),
        ),
        Control(
            "C8",
            "old page contents are cleared from the context, so a poisoned page does not stay in view forever",
            (
                f"test:{RAT}::test_old_page_contents_are_cleared_from_the_context_after_keep_pages_results",
            ),
        ),
    ),
    threats=(
        Threat(
            "T01",
            "Indirect prompt injection: a fetched page tells the model to do something else",
            "LLM01",
            "E",
            ("fetch_page", "page_content"),
            ("A1", "A2", "A4"),
            4,
            4,
            "partial",
            ("C1", "C3", "C8"),
            remaining="the model can still be steered: what it writes to notes, what it cites, what it says. Nothing detects or neutralises the injected text.",
        ),
        Threat(
            "T02",
            "SSRF: the model is talked into fetching cloud metadata or local services",
            "LLM06",
            "E",
            ("fetch_page",),
            ("A3",),
            4,
            5,
            "mitigated",
            ("C1", "C2", "C3"),
        ),
        Threat(
            "T03",
            "Exfiltration through the report: an injected instruction makes the model embed private data in a link or image URL that the client fetches",
            "LLM05",
            "I",
            ("report_output",),
            ("A1", "A2"),
            3,
            4,
            "gap",
            remaining="the report is plain model text; no step removes images, rewrites links or checks URLs before a client renders it.",
        ),
        Threat(
            "T04",
            "Poisoned search results steer which pages are read",
            "LLM01",
            "T",
            ("search_web", "search_results"),
            ("A4",),
            3,
            3,
            "partial",
            ("C4",),
            remaining="the instructions call results 'leads, not evidence'; the model can still be lured, only claims from pages it opened can be cited.",
        ),
        Threat(
            "T05",
            "Fabricated or unsupported claims and citations in the report",
            "LLM09",
            "T",
            ("report_output",),
            ("A4",),
            4,
            3,
            "mitigated",
            ("C4",),
        ),
        Threat(
            "T06",
            "Stored injection: attacker text is saved to notes and re-read in a later session",
            "LLM04",
            "T",
            (
                "create_note",
                "append_to_note",
                "read_note",
                "search_notes",
                "list_notes",
                "notes_store",
            ),
            ("A2", "A4"),
            3,
            4,
            "partial",
            ("C6", "C7"),
            remaining="note CONTENT is not inspected or tagged as untrusted; a poisoned note is trusted on re-read.",
        ),
        Threat(
            "T07",
            "Path traversal or symlink escape through note names",
            "LLM06",
            "E",
            ("create_note", "read_note", "append_to_note"),
            ("A2",),
            2,
            4,
            "mitigated",
            ("C6",),
        ),
        Threat(
            "T08",
            "A tool the server adds later becomes available to the model unreviewed",
            "LLM03",
            "E",
            ("list_notes",),
            ("A2", "A3"),
            2,
            4,
            "mitigated",
            ("C7",),
        ),
        Threat(
            "T09",
            "Unbounded consumption: loops, huge pages, repeated searches burn budget or time",
            "LLM10",
            "D",
            ("search_web", "fetch_page"),
            ("A5",),
            4,
            3,
            "mitigated",
            ("C2", "C5"),
        ),
        Threat(
            "T10",
            "The user's question and tool arguments are sent to a third-party model provider",
            "LLM02",
            "I",
            ("user_question",),
            ("A1",),
            5,
            2,
            "accepted",
            reason="inherent to using a hosted model; the user is told, and no secrets are part of the task. Revisit if the question can contain personal data (Day 5).",
        ),
    ),
)
