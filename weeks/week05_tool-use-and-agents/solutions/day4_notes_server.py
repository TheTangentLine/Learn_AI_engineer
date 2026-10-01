"""Week 5 Day 4 - Solution: an MCP server for a folder of markdown notes.

Exposes the three MCP primitives:
  tools      search_notes, read_note, list_notes   (read-only)  and  create_note, append_to_note  (write)
  resources  notes://index  and  notes://{name}    (data the HOST decides to load into context)
  prompts    summarize_note(name), weekly_review(days)   (reusable, user-triggered templates)

NOTE: this SDK does not turn docstring ``Args:`` sections into parameter descriptions, so each
parameter is described with ``Annotated[..., Field(description=...)]`` (tested: no parameter is undocumented).

The logic lives in ``NotesStore`` (plain Python, unit-testable); the MCP layer is thin.

Run it:
  NOTES_DIR=~/notes uv run python weeks/week05_tool-use-and-agents/solutions/day4_notes_server.py      # stdio
Use it from Claude Code:
  claude mcp add notes -e NOTES_DIR=$HOME/notes -- uv run python /abs/path/to/day4_notes_server.py
Use it from Claude Desktop / any host: add to its config
  {"mcpServers": {"notes": {"command": "uv", "args": ["run", "python", "/abs/path/day4_notes_server.py"],
                            "env": {"NOTES_DIR": "/abs/path/to/notes"}}}}
Poke at it interactively:  uv run mcp dev weeks/week05_tool-use-and-agents/solutions/day4_notes_server.py
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_NOTE_CHARS = 50_000
MAX_NOTES = 500
SNIPPET_CHARS = 160


class NoteError(Exception):
    """A problem the MODEL can fix; the message says what to do instead."""


@dataclass
class NoteInfo:
    name: str
    title: str
    tags: list[str]
    modified: str
    chars: int


@dataclass
class Hit:
    name: str
    score: int
    snippets: list[str]


class NotesStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NoteError(f"notes folder {str(self.root)!r} does not exist")

    # ------------------------------------------------------------------ helpers

    def _path(self, name: str) -> Path:
        n = name.strip().lower().removesuffix(".md")
        if not NAME_RE.match(n):
            raise NoteError(
                f"invalid note name {name!r}. Use 1-64 characters: lowercase letters, digits, '-' and '_' "
                "(e.g. 'meeting-2026-03-14'); no spaces, slashes or dots."
            )
        p = (self.root / f"{n}.md").resolve()
        if p.parent != self.root:  # a symlink pointing elsewhere
            raise NoteError(f"note {n!r} is not inside the notes folder")
        return p

    def _all(self) -> list[Path]:
        return sorted(
            p for p in self.root.glob("*.md") if p.is_file() and p.resolve().parent == self.root
        )

    @staticmethod
    def _title(text: str, fallback: str) -> str:
        for line in text.splitlines():
            if line.startswith("#"):
                return line.lstrip("# ").strip() or fallback
        return fallback

    @staticmethod
    def _tags(text: str) -> list[str]:
        for line in text.splitlines()[:6]:
            if line.lower().startswith("tags:"):
                return [t.strip().lower() for t in line.split(":", 1)[1].split(",") if t.strip()]
        return []

    def _info(self, p: Path) -> NoteInfo:
        text = p.read_text(errors="replace")
        return NoteInfo(
            p.stem, self._title(text, p.stem), self._tags(text),
            datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d"), len(text),
        )  # fmt: skip

    # ------------------------------------------------------------------ operations

    def list_notes(self, tag: str | None = None) -> list[NoteInfo]:
        infos = [self._info(p) for p in self._all()]
        if tag:
            infos = [i for i in infos if tag.strip().lower() in i.tags]
        return infos

    def search(self, query: str, limit: int = 5) -> list[Hit]:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        if not terms:
            raise NoteError("query has no searchable words; use at least one word of 2+ characters")
        if not 1 <= limit <= 20:
            raise NoteError("limit must be between 1 and 20")
        hits: list[Hit] = []
        for p in self._all():
            text = p.read_text(errors="replace")
            low = text.lower()
            score = sum(low.count(t) for t in terms) + 5 * sum(t in p.stem.lower() for t in terms)
            if not score:
                continue
            snippets = []
            for line in text.splitlines():
                if any(t in line.lower() for t in terms) and len(snippets) < 3:
                    snippets.append(line.strip()[:SNIPPET_CHARS])
            hits.append(Hit(p.stem, score, snippets))
        hits.sort(key=lambda h: (-h.score, h.name))
        return hits[:limit]

    def read(self, name: str) -> str:
        p = self._path(name)
        if not p.is_file():
            names = ", ".join(i.name for i in self.list_notes()[:15]) or "none yet"
            raise NoteError(
                f"no note named {p.stem!r}. Existing notes: {names}. Use search_notes to find one."
            )
        return p.read_text(errors="replace")

    def create(self, name: str, content: str, overwrite: bool = False) -> str:
        p = self._path(name)
        if len(content) > MAX_NOTE_CHARS:
            raise NoteError(
                f"note too long ({len(content)} > {MAX_NOTE_CHARS} characters); split it into several notes"
            )
        if p.exists() and not overwrite:
            raise NoteError(
                f"note {p.stem!r} already exists. Use append_to_note to add to it, or pass overwrite=true to replace it."
            )
        if not p.exists() and len(self._all()) >= MAX_NOTES:
            raise NoteError(f"the notes folder is full ({MAX_NOTES} notes)")
        existed = p.exists()
        p.write_text(content)
        return (
            f"{'Replaced' if existed else 'Created'} note {p.stem!r} ({len(content)} characters)."
        )

    def append(self, name: str, text: str) -> str:
        p = self._path(name)
        if not p.is_file():
            raise NoteError(f"no note named {p.stem!r}; create it first with create_note.")
        old = p.read_text(errors="replace")
        new = old + ("" if old.endswith("\n") or not old else "\n") + text.rstrip("\n") + "\n"
        if len(new) > MAX_NOTE_CHARS:
            raise NoteError(
                f"appending would exceed {MAX_NOTE_CHARS} characters; start a new note instead"
            )
        p.write_text(new)
        return f"Appended {len(text)} characters to {p.stem!r}."


# ----------------------------------------------------------------------------- formatting


def fmt_list(infos: list[NoteInfo]) -> str:
    if not infos:
        return "No notes."
    return "\n".join(
        f"{i.name} | {i.title} | tags: {', '.join(i.tags) or '-'} | modified {i.modified}"
        for i in infos
    )


def fmt_hits(hits: list[Hit]) -> str:
    if not hits:
        return "No notes matched. Try different or fewer words, or list_notes to see what exists."
    return "\n\n".join(
        f"{h.name} (score {h.score})\n" + "\n".join(f"  > {s}" for s in h.snippets) for h in hits
    )


# ----------------------------------------------------------------------------- the MCP server

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
)
OVERWRITE_RISK = ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
)


def build_server(root: str | Path) -> MCPServer:
    store = NotesStore(root)
    mcp = MCPServer(
        "notes",
        instructions=(
            "Personal markdown notes. Search first (search_notes), then read what you need (read_note). "
            "Note names are lowercase-with-dashes. Only write when the user asks you to."
        ),
    )

    def guard(fn):
        try:
            return fn()
        except NoteError as exc:
            raise ToolError(
                str(exc)
            ) from None  # ToolError text reaches the model; other exceptions are hidden

    @mcp.tool(annotations=READ_ONLY)
    def list_notes(
        tag: Annotated[
            str | None,
            Field(
                description='Only notes whose "tags:" header line contains this tag, e.g. "meeting".'
            ),
        ] = None,
    ) -> str:
        """List notes, one per line: name | title | tags | last-modified date. Optionally only those with a tag."""
        return fmt_list(guard(lambda: store.list_notes(tag)))

    @mcp.tool(annotations=READ_ONLY)
    def search_notes(
        query: Annotated[
            str, Field(description='Words to look for, e.g. "kubernetes upgrade plan".')
        ],
        limit: Annotated[int, Field(description="Maximum notes to return (1-20).")] = 5,
    ) -> str:
        """Search all notes for words (case-insensitive); best matches first, each with matching lines.
        Use this before read_note when you do not know the exact note name.
        """
        return fmt_hits(guard(lambda: store.search(query, limit)))

    @mcp.tool(annotations=READ_ONLY)
    def read_note(
        name: Annotated[
            str,
            Field(
                description='The note name as shown by list_notes or search_notes, e.g. "meeting-2026-03-14".'
            ),
        ],
    ) -> str:
        """Return the full text of one note."""
        return guard(lambda: store.read(name))

    @mcp.tool(annotations=WRITE)
    def append_to_note(
        name: Annotated[str, Field(description="Existing note name.")],
        text: Annotated[str, Field(description="The text to add (markdown).")],
    ) -> str:
        """Add text to the end of an existing note. Prefer this over create_note with overwrite."""
        return guard(lambda: store.append(name, text))

    @mcp.tool(annotations=OVERWRITE_RISK)
    def create_note(
        name: Annotated[
            str,
            Field(
                description='New note name: lowercase letters, digits, "-" and "_" only, e.g. "reading-list".'
            ),
        ],
        content: Annotated[
            str,
            Field(
                description='Full markdown text. Put a line "tags: a, b" near the top to tag it.'
            ),
        ],
        overwrite: Annotated[
            bool,
            Field(
                description="Replace an existing note. Only set true when the user asked to replace it."
            ),
        ] = False,
    ) -> str:
        """Create a new note. Fails if the name exists unless overwrite is true (which DELETES the old text)."""
        return guard(lambda: store.create(name, content, overwrite))

    @mcp.resource("notes://index", mime_type="text/plain")
    def index() -> str:
        """The list of all notes (same as the list_notes tool)."""
        return fmt_list(store.list_notes())

    @mcp.resource("notes://{name}", mime_type="text/markdown")
    def note(name: str) -> str:
        """One note's markdown."""
        return guard(lambda: store.read(name))

    @mcp.prompt()
    def summarize_note(name: Annotated[str, Field(description="Note name.")]) -> str:
        """Summarize one note in five bullet points."""
        return f"Read the note '{name}' with read_note, then summarize it in at most five bullet points."

    @mcp.prompt()
    def weekly_review(
        days: Annotated[str, Field(description="How many days back to look.")] = "7",
    ) -> str:
        """Review what changed in the notes recently."""
        return (
            f"Use list_notes to find notes modified in the last {days} days, read the relevant ones, and write "
            "a short weekly review: what I worked on, open questions, and next steps."
        )

    return mcp


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NOTES_DIR")
    if not root:
        sys.exit("Set NOTES_DIR (or pass the folder as the first argument).")
    build_server(root).run()  # stdio: never print() to stdout here, it is the protocol channel


if __name__ == "__main__":
    main()
