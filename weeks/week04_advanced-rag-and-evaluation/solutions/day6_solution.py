"""Week 4 Day 6 - Solution: table-aware ingestion, and images via a pluggable vision describer.

We generate a real PDF report (narrative + two tables + a chart), then compare four ways of turning it
into chunks, by whether retrieval returns a chunk that can ANSWER a cell-lookup question:

  naive-300     pypdf text (one cell per line), 300-char chunks            (the default many tutorials use)
  naive-1000    pypdf text, 1000-char chunks                                (bigger chunks "fix" it...)
  tables-md     pdfplumber tables -> one Markdown table per chunk, caption kept
  row-sentences pdfplumber tables -> one self-describing sentence per row  ("EMEA: Q3 2024 = 4,210; ...")

A chunk ANSWERS (row, column, value) iff it contains the row label, the column header AND the value: the
minimum evidence an LLM needs. (That is generous to naive text, which can still scramble which number
belongs to which column.)

Images: charts are extracted from the PDF and turned into text by a VISION describer; here a scripted
stand-in so the pipeline is testable. Plug in a real vision model with `describe_with_llm`.

  uv run python weeks/week04_advanced-rag-and-evaluation/solutions/day6_solution.py
"""

from __future__ import annotations

import io
import random
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pdfplumber  # noqa: E402
from day3_solution import MiniIndex  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from common.chunking import recursive  # noqa: E402
from common.embed import get_embedder  # noqa: E402
from common.evalkit import bootstrap_ci, fmt_ci  # noqa: E402

SPECS = [  # (title, short name, row labels, column headers, value formatter)
    (
        "Revenue by region (USD thousands)",
        "revenue",
        ["North America", "EMEA", "APAC", "LATAM", "Nordics", "Iberia"],
        ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
        lambda r: f"{r.randint(900, 9000):,}",
    ),
    (
        "Headcount by department",
        "headcount",
        [
            "Engineering",
            "Sales",
            "Support",
            "Finance",
            "Legal",
            "Marketing",
            "Design",
            "Operations",
        ],
        ["Jan 2024", "Apr 2024", "Jul 2024", "Oct 2024"],
        lambda r: str(r.randint(12, 280)),
    ),
    (
        "Product pricing (USD)",
        "pricing",
        ["Anvil", "Beacon", "Cobalt", "Delta", "Ember", "Fjord", "Garnet", "Helix", "Iris", "Juno"],
        ["List price", "Unit cost", "Margin %"],
        lambda r: f"{r.uniform(5, 400):.2f}",
    ),
    (
        "Operating expenses (USD thousands)",
        "expenses",
        [
            "Salaries",
            "Cloud hosting",
            "Travel",
            "Legal fees",
            "Marketing",
            "Facilities",
            "Training",
            "Insurance",
        ],
        ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
        lambda r: f"{r.randint(40, 4800):,}",
    ),
    (
        "Customers by tier",
        "customers",
        ["Free", "Starter", "Team", "Business", "Enterprise"],
        ["2022", "2023", "2024"],
        lambda r: f"{r.randint(120, 90000):,}",
    ),
    (
        "Support tickets by type",
        "tickets",
        [
            "Billing",
            "Login",
            "Outage",
            "Feature request",
            "Bug report",
            "Refund",
            "Onboarding",
            "Security",
        ],
        ["Opened", "Closed", "Avg hours to close"],
        lambda r: f"{r.randint(8, 2400):,}",
    ),
    (
        "Warehouse inventory by site",
        "inventory",
        ["Rotterdam", "Memphis", "Osaka", "Santos", "Pune", "Leeds", "Dubai", "Lagos"],
        ["Units in stock", "Units on order", "Days of cover"],
        lambda r: f"{r.randint(30, 52000):,}",
    ),
    (
        "Employee satisfaction by team",
        "satisfaction",
        ["Platform", "Mobile", "Growth", "Data", "Security", "Billing", "Support", "Docs"],
        ["Q1 score", "Q2 score", "Q3 score", "Q4 score"],
        lambda r: f"{r.uniform(3.1, 4.9):.1f}",
    ),
]
FILLER = [
    "The company continued to execute against its strategic plan during the year, balancing investment in new products with disciplined cost control.",
    "Management believes the current operating model positions the business well for the next planning cycle, subject to the risks described elsewhere in this report.",
    "Customer demand remained broadly stable across segments, with notable strength in enterprise accounts and moderate softness in the self-serve segment.",
    "The leadership team reviewed hiring plans each quarter and adjusted priorities in line with revenue expectations and product milestones.",
    "Operational resilience improved following investments in monitoring, incident response and capacity planning across all regions.",
]


def build_tables(seed: int = 5) -> list[dict]:
    rng = random.Random(seed)
    tables = []
    for n, (title, metric, rows, cols, fmt) in enumerate(SPECS, 1):
        tables.append(
            {
                "title": f"Table {n}: {title}",
                "short": title,
                "metric": metric,
                "header": [metric.title().split()[0] if False else "Item", *cols],
                "rows": [[r, *[fmt(rng) for _ in cols]] for r in rows],
            }
        )
    return tables


TABLES = build_tables()
CHART_VALUES = {"Q1": 13010, "Q2": 13695, "Q3": 14510, "Q4": 15390}
CHART_TITLE = "Total revenue by quarter (USD k)"


# ----------------------------------------------------------------- build the PDF


def chart_png() -> bytes:
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(list(CHART_VALUES), list(CHART_VALUES.values()))
    ax.set_title(CHART_TITLE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def make_report(path: Path) -> None:
    from reportlab.platypus import KeepTogether

    st = getSampleStyleSheet()
    rng = random.Random(1)
    flow = [Paragraph("Annual Report 2024", st["Title"])]
    for t in TABLES:
        for _ in range(3):  # narrative between tables: distractor text that is NOT the answer
            flow.append(Paragraph(" ".join(rng.sample(FILLER, 3)), st["Normal"]))
        flow.append(Spacer(1, 8))
        tbl = Table([t["header"], *t["rows"]])
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]
            )
        )
        flow.append(KeepTogether([Paragraph(t["title"], st["Heading3"]), tbl, Spacer(1, 10)]))
    flow += [
        Paragraph("Figure 1: Quarterly trend", st["Heading3"]),
        Image(io.BytesIO(chart_png()), width=280, height=180),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(flow)


# ----------------------------------------------------------------- four ingestion strategies


def naive_chunks(path: Path, size: int) -> list[str]:
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return recursive(text, size, size // 8)


def extract_tables(path: Path) -> list[dict]:
    """[{caption, header, rows}] using pdfplumber; the caption is the text just above the table."""
    out = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for t in page.find_tables():
                x0, top, x1, _ = t.bbox
                above = page.crop((0, max(0, top - 26), page.width, top)).extract_text() or ""
                data = t.extract()
                out.append(
                    {
                        "caption": above.strip().splitlines()[-1] if above.strip() else "",
                        "header": data[0],
                        "rows": data[1:],
                    }
                )
    return out


def markdown_table_chunks(tables: list[dict]) -> list[str]:
    chunks = []
    for t in tables:
        md = [
            f"{t['caption']}",
            "| " + " | ".join(t["header"]) + " |",
            "|" + "---|" * len(t["header"]),
        ]
        md += ["| " + " | ".join(r) + " |" for r in t["rows"]]
        chunks.append("\n".join(md))
    return chunks


def row_sentence_chunks(tables: list[dict]) -> list[str]:
    """One self-describing chunk per row: caption + 'row label: column = value; ...'. Header repeated."""
    chunks = []
    for t in tables:
        for r in t["rows"]:
            cells = "; ".join(f"{h} = {v}" for h, v in zip(t["header"][1:], r[1:], strict=True))
            chunks.append(f"{t['caption']}. {t['header'][0]}: {r[0]}. {cells}")
    return chunks


# ----------------------------------------------------------------- questions and the answerability test


def cell_questions(n_per_table: int = 5, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    qs = []
    for t in TABLES:
        cells = [(r, c) for r in range(len(t["rows"])) for c in range(1, len(t["header"]))]
        for r, c in rng.sample(cells, n_per_table):
            row, col, value = t["rows"][r][0], t["header"][c], t["rows"][r][c]
            qs.append(
                {
                    "q": f"In the {t['short'].lower()} table, what was {col} for {row}?",
                    "row": row,
                    "col": col,
                    "value": value,
                    "table": t["title"],
                }
            )
    return qs


def answers(chunk: str, q: dict) -> bool:
    """Does this chunk contain the row label, the column header and the value (the minimum evidence)?"""
    return q["row"] in chunk and q["col"] in chunk and q["value"] in chunk


def evaluate_strategy(name: str, chunks: list[str], questions: list[dict], emb) -> dict:
    index = MiniIndex(emb, ["report"] * len(chunks), chunks, chunks)
    h1, h3, rr = [], [], []
    for q in questions:
        res = [t for _, t in index.search(q["q"], 10)]
        ok = [answers(t, q) for t in res]
        h1.append(float(ok[0]))
        h3.append(float(any(ok[:3])))
        rr.append(next((1 / i for i, o in enumerate(ok, 1) if o), 0.0))
    return {
        "name": name,
        "n": len(chunks),
        "mean_len": float(np.mean([len(c) for c in chunks])),
        "hit1": h1,
        "hit3": h3,
        "mrr": rr,
        "any": sum(any(answers(c, q) for c in chunks) for q in questions),
    }


# ----------------------------------------------------------------- images


def extract_images(path: Path) -> list[tuple[int, bytes]]:
    return [
        (i, img.data) for i, page in enumerate(PdfReader(str(path)).pages, 1) for img in page.images
    ]


def scripted_describer(png: bytes) -> str:
    """Stand-in for a vision model (we generated the chart, so we know what it shows)."""
    vals = ", ".join(f"{q} = {v:,}" for q, v in CHART_VALUES.items())
    return f"Bar chart titled '{CHART_TITLE}'. Bars: {vals}. Revenue rises every quarter."


def image_chunks(path: Path, describe: Callable[[bytes], str]) -> list[str]:
    """Each image becomes a text chunk: description + provenance, so it can be retrieved and cited."""
    return [f"[Figure on page {page}] {describe(png)}" for page, png in extract_images(path)]


def image_message(provider: str, png: bytes, question: str) -> list[dict]:
    """Provider-specific message with an image (NOT executed in this lesson: needs a vision API key)."""
    import base64

    b64 = base64.standard_b64encode(png).decode()
    if provider == "anthropic":
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": question},
        ]
    else:  # openai Responses API
        content = [
            {"type": "input_text", "text": question},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ]
    return [{"role": "user", "content": content}]


def approx_image_tokens(width: int, height: int) -> int:
    """Rough rule of thumb used by some providers: tokens ~ width*height/750 (check your provider's docs)."""
    return round(width * height / 750)


# ----------------------------------------------------------------- main


def main() -> None:
    emb = get_embedder()
    path = Path(tempfile.mkdtemp()) / "annual_report.pdf"
    make_report(path)
    tables = extract_tables(path)
    qs = cell_questions()
    pages = len(PdfReader(str(path)).pages)
    print(
        f"report: {pages} pages | tables found by pdfplumber: {len(tables)} (of {len(TABLES)} in the PDF) | "
        f"{len(qs)} cell-lookup questions\n"
    )
    lines = [
        ln for pg in PdfReader(str(path)).pages for ln in (pg.extract_text() or "").splitlines()
    ]
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "Item")
    print("pypdf output for the start of table 1 (every cell on its own line):")
    print("   " + " | ".join(ln.strip() for ln in lines[start : start + 14]) + " ...\n")

    strategies = {
        "naive-300 (pypdf)": naive_chunks(path, 300),
        "naive-1000 (pypdf)": naive_chunks(path, 1000),
        "tables-md (pdfplumber)": markdown_table_chunks(tables),
        "row-sentences": row_sentence_chunks(tables),
    }
    print(
        f"{'strategy':<24}{'chunks':>7}{'mean len':>9}{'answerable':>11}{'hit@1':>18}{'hit@3':>18}"
    )
    for name, chunks in strategies.items():
        r = evaluate_strategy(name, chunks, qs, emb)
        print(
            f"{name:<24}{r['n']:>7}{r['mean_len']:>9.0f}{r['any']:>8}/{len(qs)}"
            f"{fmt_ci(bootstrap_ci(r['hit1'])):>18}{fmt_ci(bootstrap_ci(r['hit3'])):>18}"
        )
    print(
        "(answerable = the question can be answered from SOME chunk at all: the ceiling for retrieval)\n"
    )

    imgs = extract_images(path)
    print(f"images found in the PDF: {len(imgs)} (page {imgs[0][0]}, {len(imgs[0][1]):,} bytes)")
    chunks = [*row_sentence_chunks(tables), *image_chunks(path, scripted_describer)]
    index = MiniIndex(emb, ["report"] * len(chunks), chunks, chunks)
    q = "According to the figure, what was total revenue in Q3?"
    top = index.search(q, 1)[0][1]
    print(f"\nchart question: {q!r}\n   top chunk: {top[:110]!r}")
    print(
        f"   an image of 400x260 px costs roughly {approx_image_tokens(400, 260)} input tokens per call; "
        "describing it ONCE at ingest and indexing the text is far cheaper than re-sending it per question."
    )


if __name__ == "__main__":
    main()
