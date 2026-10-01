"""Tests for Week 5 Day 4: the notes logic, then the REAL server over stdio through our MCP bridge."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import day4_notes_server as ns  # noqa: E402

from common.agent import run_agent  # noqa: E402
from common.chat import ToolCall  # noqa: E402
from common.fake import fake_llm, tool_calls  # noqa: E402
from common.mcp_tools import McpBridge, McpError, stdio  # noqa: E402

SERVER = str(Path(__file__).parent / "day4_notes_server.py")

NOTES = {
    "meeting-2026-03-14": "# Platform sync\ntags: meeting, platform\nDecided to upgrade kubernetes to 1.31 in April.\nOwner: Dana.\n",
    "reading-list": "# Reading list\ntags: personal\nDesigning Data-Intensive Applications\nThe kubernetes book\n",
    "caching-ideas": "# Caching ideas\ntags: platform, design\nUse a write-through cache. Cache invalidation is hard.\nCache keys include tenant id.\n",
    "empty": "",
}


@pytest.fixture()
def root(tmp_path):
    for name, text in NOTES.items():
        (tmp_path / f"{name}.md").write_text(text)
    return tmp_path


@pytest.fixture()
def store(root):
    return ns.NotesStore(root)


# ----------------------------------------------------------------------------- NotesStore


def test_list_notes_with_titles_tags_and_a_tag_filter(store):
    infos = {i.name: i for i in store.list_notes()}
    assert set(infos) == set(NOTES) and infos["meeting-2026-03-14"].title == "Platform sync"
    assert (
        infos["meeting-2026-03-14"].tags == ["meeting", "platform"]
        and infos["empty"].title == "empty"
    )
    assert [i.name for i in store.list_notes("platform")] == ["caching-ideas", "meeting-2026-03-14"]
    assert [i.name for i in store.list_notes(" PLATFORM ")] == [
        "caching-ideas",
        "meeting-2026-03-14",
    ]
    assert store.list_notes("nope") == []
    assert ns.fmt_list([]) == "No notes."
    assert "caching-ideas | Caching ideas | tags: platform, design | modified " in ns.fmt_list(
        store.list_notes()
    )


def test_search_ranks_by_matches_and_name_and_returns_snippets(store):
    hits = store.search("kubernetes upgrade")
    assert [h.name for h in hits] == ["meeting-2026-03-14", "reading-list"]
    assert hits[0].snippets == ["Decided to upgrade kubernetes to 1.31 in April."]
    assert store.search("cache")[0].name == "caching-ideas"
    assert [h.name for h in store.search("reading")][0] == "reading-list", "the note NAME counts"
    assert len(store.search("kubernetes", limit=1)) == 1 and store.search("zzzzqq") == []
    assert "No notes matched" in ns.fmt_hits([]) and "score" in ns.fmt_hits(hits)


def test_search_validates_its_input(store):
    with pytest.raises(ns.NoteError, match="searchable words"):
        store.search("a ? !")
    with pytest.raises(ns.NoteError, match="between 1 and 20"):
        store.search("cache", limit=0)
    with pytest.raises(ns.NoteError, match="between 1 and 20"):
        store.search("cache", limit=21)


def test_read_accepts_case_and_md_suffix_and_missing_notes_list_what_exists(store):
    assert store.read("Reading-List.md").startswith("# Reading list")
    with pytest.raises(ns.NoteError) as e:
        store.read("nope")
    assert (
        "no note named 'nope'" in str(e.value)
        and "caching-ideas" in str(e.value)
        and "search_notes" in str(e.value)
    )


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "a/b",
        "a b",
        "",
        "x" * 65,
        ".hidden",
        "note.txt",
        "-leading",
        "naïve",
        "a\x00b",
    ],
)
def test_note_names_cannot_escape_or_be_weird(store, bad):
    for op in (
        lambda: store.read(bad),
        lambda: store.create(bad, "x"),
        lambda: store.append(bad, "x"),
    ):
        with pytest.raises(ns.NoteError, match="invalid note name"):
            op()


def test_symlinked_notes_pointing_outside_are_not_listed_or_readable(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.md"
    outside.write_text("# Secret\nTOP SECRET")
    (root / "leak.md").symlink_to(outside)
    store = ns.NotesStore(root)
    assert "leak" not in [i.name for i in store.list_notes()]
    assert store.search("secret") == []
    with pytest.raises(ns.NoteError, match="not inside"):
        store.read("leak")
    with pytest.raises(ns.NoteError, match="not inside"):
        store.create("leak", "overwrite attempt", overwrite=True)
    assert outside.read_text() == "# Secret\nTOP SECRET"


def test_create_refuses_to_overwrite_unless_asked_and_enforces_limits(store, root):
    assert store.create("new-note", "# New\nhello") == "Created note 'new-note' (11 characters)."
    with pytest.raises(ns.NoteError, match="append_to_note.*overwrite=true"):
        store.create("new-note", "other")
    assert (root / "new-note.md").read_text() == "# New\nhello"
    assert store.create("new-note", "replaced", overwrite=True).startswith("Replaced note")
    assert (root / "new-note.md").read_text() == "replaced"
    assert store.create("fresh", "x", overwrite=True).startswith("Created"), (
        "overwrite on a new name is just a create"
    )
    with pytest.raises(ns.NoteError, match="too long"):
        store.create("big", "x" * (ns.MAX_NOTE_CHARS + 1))
    assert not (root / "big.md").exists()


def test_note_count_limit(root, monkeypatch):
    monkeypatch.setattr(ns, "MAX_NOTES", 4)
    store = ns.NotesStore(root)
    with pytest.raises(ns.NoteError, match="full"):
        store.create("one-more", "x")
    store.create("empty", "still allowed to replace", overwrite=True)


def test_append_adds_a_newline_boundary_and_checks_existence_and_size(store, root):
    store.append("reading-list", "Another book")
    assert (root / "reading-list.md").read_text().endswith("The kubernetes book\nAnother book\n")
    (root / "no-newline.md").write_text("first")
    store.append("no-newline", "second\n\n")
    assert (root / "no-newline.md").read_text() == "first\nsecond\n"
    store.append("empty", "start")
    assert (root / "empty.md").read_text() == "start\n"
    with pytest.raises(ns.NoteError, match="create it first"):
        store.append("missing", "x")
    with pytest.raises(ns.NoteError, match="exceed"):
        store.append("reading-list", "x" * ns.MAX_NOTE_CHARS)


def test_store_requires_an_existing_folder(tmp_path):
    with pytest.raises(ns.NoteError, match="does not exist"):
        ns.NotesStore(tmp_path / "nope")


# ----------------------------------------------------------------------------- the real server over stdio


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    root = tmp_path_factory.mktemp("live-notes")
    for name, text in NOTES.items():
        (root / f"{name}.md").write_text(text)
    env = {**os.environ, "NOTES_DIR": str(root)}
    with McpBridge(stdio(sys.executable, SERVER, env=env)) as bridge:
        yield bridge, root


def test_server_advertises_tools_with_schemas_and_descriptions(live):
    bridge, _ = live
    specs = {s["name"]: s for s in bridge.list_tools()}
    assert set(specs) == {
        "list_notes",
        "search_notes",
        "read_note",
        "create_note",
        "append_to_note",
    }
    q = specs["search_notes"]["parameters"]
    assert (
        q["required"] == ["query"] and "limit" in q["properties"] and "Words to look for" in str(q)
    )
    assert "DELETES the old text" in specs["create_note"]["description"]


def test_tool_calls_work_end_to_end_and_errors_reach_the_model_as_text(live):
    bridge, _ = live
    assert "meeting-2026-03-14" in bridge.call_tool("search_notes", {"query": "kubernetes"})
    assert bridge.call_tool("read_note", {"name": "reading-list"}).startswith("# Reading list")
    reg = bridge.registry()
    r = reg.execute(ToolCall("1", "read_note", {"name": "nope"}))
    assert r.is_error and "no note named 'nope'" in r.content and "Existing notes" in r.content
    r = reg.execute(ToolCall("2", "read_note", {"name": "../../etc/passwd"}))
    assert r.is_error and "invalid note name" in r.content
    r = reg.execute(ToolCall("3", "search_notes", {"query": 123}))
    assert r.is_error and "validation" in r.content.lower() or "string" in r.content.lower()
    r = reg.execute(ToolCall("4", "no_such_tool", {}))
    assert r.is_error and "Unknown tool" in r.content


def test_write_tools_change_the_folder_and_overwrite_needs_the_flag(live):
    bridge, root = live
    reg = bridge.registry()
    assert not reg.execute(
        ToolCall("1", "create_note", {"name": "from-mcp", "content": "# Hi\nx"})
    ).is_error
    assert (root / "from-mcp.md").read_text() == "# Hi\nx"
    dup = reg.execute(ToolCall("2", "create_note", {"name": "from-mcp", "content": "y"}))
    assert (
        dup.is_error
        and "already exists" in dup.content
        and (root / "from-mcp.md").read_text() == "# Hi\nx"
    )
    assert not reg.execute(
        ToolCall("3", "append_to_note", {"name": "from-mcp", "text": "more"})
    ).is_error
    assert (root / "from-mcp.md").read_text().endswith("more\n")


def test_resources_and_prompts(live):
    bridge, _ = live
    assert (
        "notes://index" in bridge.list_resources()
        and "notes://{name}" in bridge.list_resource_templates()
    )
    assert "reading-list | Reading list" in bridge.read_resource("notes://index")
    assert bridge.read_resource("notes://caching-ideas").startswith("# Caching ideas")
    assert set(bridge.list_prompts()) == {"summarize_note", "weekly_review"}
    p = bridge.get_prompt("summarize_note", {"name": "reading-list"})
    assert p.startswith("user: ") and "reading-list" in p and "read_note" in p
    assert "last 7 days" in bridge.get_prompt("weekly_review")
    with pytest.raises(Exception, match="(?i)nope|not found|no note"):
        bridge.read_resource("notes://nope")


def test_an_agent_can_use_the_server_through_the_bridge(live):
    bridge, _ = live
    reg = bridge.registry(allow={"search_notes", "read_note"})
    script = [
        (
            r"tags: meeting, platform",
            "Dana owns the kubernetes upgrade.",
        ),  # only the FULL note has this line
        (r"(?s)search_notes.*", tool_calls(("read_note", {"name": "meeting-2026-03-14"}))),
        (r"(?s).*", tool_calls(("search_notes", {"query": "kubernetes upgrade owner"}))),
    ]
    with fake_llm(script):
        run = run_agent("Who owns the kubernetes upgrade?", reg, provider="anthropic")
    assert run.ok and run.tool_names == ["search_notes", "read_note"] and "Dana" in run.answer
    assert run.errors == 0 and "Owner: Dana" in run.results[1].content


# ----------------------------------------------------------------------------- the bridge itself


def test_registry_allowlist_denylist_and_prefix(live):
    bridge, _ = live
    assert sorted(bridge.registry(allow={"read_note"}).names()) == ["read_note"]
    assert "create_note" not in bridge.registry(deny={"create_note", "append_to_note"}).names()
    prefixed = bridge.registry(allow={"read_note"}, prefix="notes__")
    assert prefixed.names() == ["notes__read_note"]
    assert not prefixed.execute(
        ToolCall("1", "notes__read_note", {"name": "reading-list"})
    ).is_error
    unknown = bridge.registry(allow={"read_note"}).execute(
        ToolCall("1", "create_note", {"name": "x", "content": "y"})
    )
    assert unknown.is_error and "Unknown tool" in unknown.content, (
        "an allowlist hides tools from the model AND blocks calls"
    )
    with pytest.raises(McpError, match="not offered by the server"):
        bridge.registry(allow={"read_note", "delete_everything"})


def test_parallel_calls_through_one_connection_return_in_order(live):
    bridge, _ = live
    reg = bridge.registry()
    calls = [
        ToolCall(str(i), "read_note", {"name": n})
        for i, n in enumerate(["reading-list", "caching-ideas", "reading-list"])
    ]
    out = reg.execute_all(calls)
    assert [r.content.splitlines()[0] for r in out] == [
        "# Reading list",
        "# Caching ideas",
        "# Reading list",
    ]


def test_bridge_errors_are_clear():
    bridge = McpBridge(stdio(sys.executable, "-c", "import sys; sys.exit(3)"), startup_timeout_s=20)
    with pytest.raises(McpError, match="failed to start"):
        bridge.start()
    with pytest.raises(McpError, match="not running"):
        McpBridge(stdio("x")).list_tools()
    never = McpBridge(
        stdio(sys.executable, "-c", "import time; time.sleep(30)"), startup_timeout_s=1.0
    )
    t0 = time.perf_counter()
    with pytest.raises(McpError, match="did not start within"):
        never.start()
    assert time.perf_counter() - t0 < 10
    never.close()


def test_bridge_closes_its_subprocess_and_can_not_be_reused(root):
    env = {**os.environ, "NOTES_DIR": str(root)}
    bridge = McpBridge(stdio(sys.executable, SERVER, env=env)).start()
    assert "list_notes" in [s["name"] for s in bridge.list_tools()]
    thread = bridge._thread
    bridge.close()
    assert thread is not None and not thread.is_alive()
    with pytest.raises(McpError, match="not running"):
        bridge.call_tool("list_notes", {})


def test_server_without_a_notes_dir_exits_with_a_message():
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "NOTES_DIR"}
    r = subprocess.run(
        [sys.executable, SERVER], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode != 0 and "NOTES_DIR" in r.stderr and r.stdout == ""


def test_search_ordering_name_boost_ties_and_snippet_cap(root, store):
    (root / "cache.md").write_text("# x\nnothing relevant\n")
    (root / "other.md").write_text("cache\ncache\nlots of cache\n")
    scores = {h.name: h.score for h in store.search("cache")}
    assert scores["cache"] == 5 and scores["other"] == 3, (
        "a name match adds 5; each body occurrence adds 1"
    )
    assert store.search("cache")[0].name == "cache", "the name boost outranks 3 body mentions"
    (root / "bbb-note.md").write_text("zebra\n")
    (root / "aaa-note.md").write_text("zebra\n")
    assert [h.name for h in store.search("zebra")] == ["aaa-note", "bbb-note"], (
        "equal scores sort by name"
    )
    (root / "many.md").write_text("\n".join(f"line {i} quokka" for i in range(6)))
    assert len(store.search("quokka")[0].snippets) == 3


def test_higher_score_beats_alphabetical_order(root, store):
    (root / "aaa.md").write_text("walrus\n")
    (root / "zzz.md").write_text("walrus walrus walrus walrus\n")
    hits = store.search("walrus")
    assert [h.name for h in hits] == ["zzz", "aaa"] and [h.score for h in hits] == [4, 1]


def test_tool_errors_are_passed_through_verbatim_not_wrapped_in_exception_noise(live):
    bridge, _ = live
    r = bridge.registry().execute(ToolCall("1", "read_note", {"name": "nope"}))
    assert r.content.startswith("Error executing tool read_note: no note named 'nope'")
    assert "RuntimeError" not in r.content and "Tool read_note failed" not in r.content


def test_tool_list_is_fetched_once_and_cached(live):
    bridge, _ = live
    bridge._tools_cache = None
    calls = []
    real = bridge._call
    bridge._call = lambda f: calls.append(1) or real(f)
    try:
        first = bridge.list_tools()
        assert bridge.list_tools() is first and bridge.registry().names() and len(calls) == 1
    finally:
        bridge._call = real


def test_text_flattening_of_results():
    from types import SimpleNamespace as NS

    from common.mcp_tools import _text

    text = NS(type="text", text="hello")
    image = NS(type="image")
    assert (
        _text(NS(content=[text, image, text], structured_content=None))
        == "hello\n[image content omitted]\nhello"
    )
    assert _text(NS(content=[], structured_content={"result": 3})) == '{"result": 3}'
    assert _text(NS(content=[text], structured_content={"result": 3})) == "hello", (
        "text wins over structured data"
    )
    assert _text(NS(content=None, structured_content=None)) == ""


def test_demo_checker_and_a_full_scripted_run_of_the_client_demo(monkeypatch, capsys):
    import re

    import day4_client_demo as demo

    pattern = demo.QUESTIONS[3][1]
    assert not re.search(pattern, "Let me check whether a note exists", re.I), (
        "'no' must not match inside 'note'"
    )
    assert re.search(pattern, "I couldn't find anything about that.", re.I)
    assert re.search(pattern, "There is no information about a favourite colour.", re.I)

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    rules = [
        (
            r"kubernetes upgrade\?",
            [tool_calls(("search_notes", {"query": "kubernetes upgrade"})), "Dana owns it."],
        ),
        (
            r"flight to Hanoi",
            [tool_calls(("search_notes", {"query": "hanoi flight"})), "It lands at 14:30."],
        ),
        (
            r"cache keys",
            [tool_calls(("search_notes", {"query": "cache keys"})), "Keys include the tenant id."],
        ),
        (
            r"favourite colour",
            [
                tool_calls(("search_notes", {"query": "colour"})),
                "I couldn't find a note about that.",
            ],
        ),
    ]
    with fake_llm(rules):
        demo.main()
    out = capsys.readouterr().out
    assert (
        "passed 4/4" in out
        and "server tools: ['list_notes', 'search_notes'" in out
        and out.count("==> PASS") == 4
    )
    # an agent that answers the colour question WITHOUT searching must not pass
    with fake_llm([(r"(?s).*", "I couldn't say.")]):
        demo.main()
    assert "passed 0/4" in capsys.readouterr().out
