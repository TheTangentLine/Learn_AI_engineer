"""Tests for the dashboard and the deployment files (parsed, not just present)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from copilot import dashboard as D

HERE = Path(__file__).resolve().parent
DEPLOY = HERE / "deploy"
sys.path.insert(0, str(DEPLOY))

import build_context as BC  # noqa: E402

SECRET_SHAPES = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"\b[0-9a-f]{64}\b"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def metrics():
    return {
        "title": "Course <Copilot>",
        "subtitle": "a & b",
        "split": "test",
        "pass_by_kind": {
            "single": {
                "n": 22,
                "passed": 15,
                "rate": (15 / 22, 0.47, 0.84),
                "interval": "[47%, 84%]",
            },
            "adversarial": {
                "n": 3,
                "passed": 3,
                "rate": (1.0, 0.44, 1.0),
                "interval": "[44%, 100%]",
            },
        },
        "stages_ms": {"retrieve": {"p50": 28.0, "p95": 60.0}, "gate": {"p50": 420.0, "p95": 700.0}},
        "load_closed": [
            {"users": 1, "rps": 2.1, "lat50": 0.45, "lat95": 0.8, "errors": 0},
            {"users": 8, "rps": 2.3, "lat50": 3.4, "lat95": 5.0, "errors": 2},
        ],
        "load_open": [{"rate": 1.0, "n": 20, "lat50": 0.5, "lat95": 0.9, "errors": 0}],
        "cost": [{"name": "extractive", "seconds": 0.45, "tokens": 0, "per_1000": 0.025}],
        "notes": ["one process"],
    }


def test_the_dashboard_renders_every_section_with_the_numbers_and_escapes_everything():
    h = D.render(metrics())
    assert (
        h.startswith("<!doctype html>")
        and "Course &lt;Copilot&gt;" in h
        and "a &amp; b" in h
        and "<Copilot>" not in h
    )
    for needle in (
        "Golden set (test split)",
        "15/22",
        "68%",
        "[47%, 84%]",
        "Where the time goes",
        "420.0 ms",
        "Load: concurrent users",
        "3.40 s",
        "Load: arrival rate",
        "Cost per 1,000",
        "$0.025",
        "one process",
    ):
        assert needle in h, needle
    assert "class='bad'>2<" in h  # the error count of the overloaded row is flagged
    assert (
        "<script" not in h and "http://" not in h and "https://" not in h
    )  # no JavaScript and no external request


def test_missing_sections_are_left_out_and_the_bar_widths_are_relative_to_the_slowest():
    h = D.render({"title": "T", "stages_ms": {"a": {"p50": 50.0, "p95": 100.0}}})
    assert "Golden set" not in h and "Load:" not in h and "Cost per" not in h
    assert "width:50.0%" in h and "width:100.0%" in h  # p50 is half of the largest p95
    assert D.pct((float("nan"), 0, 0)) == "n/a"


def dockerfile_lines():
    text = (DEPLOY / "Dockerfile").read_text().replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_the_dockerfile_bakes_in_no_weights_index_or_secrets_and_runs_unprivileged():
    lines = dockerfile_lines()
    text = (DEPLOY / "Dockerfile").read_text()
    assert (
        lines[0] == "FROM python:3.12-slim"
        and "USER app" in lines
        and lines[-1].startswith('CMD ["python", "-m", "copilot.service"]')
    )
    assert any(line.startswith("HEALTHCHECK") for line in lines)
    assert not any(
        line.startswith("COPY") and any(w in line for w in ("index", "models", ".gguf", ".env"))
        for line in lines
    )
    env = next(line for line in lines if line.startswith("ENV") and "HF_HOME" in line)
    assert (
        "HF_HUB_OFFLINE=1" in env and "/models/hf" in env
    )  # the model cache is mounted, and the container never downloads
    assert "COPILOT_KEYS" not in text.replace(
        "(COPILOT_KEYS, COPILOT_ADMIN_TOKEN)", ""
    ) and not any(s.search(text) for s in SECRET_SHAPES)
    pip_lines = [line for line in lines if "pip install" in line]
    assert (
        "download.pytorch.org/whl/cpu" in pip_lines[0]
    )  # the CPU wheel, not several gigabytes of CUDA libraries
    assert lines.index(
        next(line for line in lines if line.startswith("COPY requirements"))
    ) < lines.index(next(line for line in lines if line.startswith("COPY copilot")))


def test_the_requirements_list_what_the_service_imports():
    reqs = {
        re.split(r"[<>=]", line)[0].lower()
        for line in (DEPLOY / "requirements-copilot.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "rank-bm25",
        "numpy",
        "torch",
        "transformers",
        "python-dotenv",
        "opentelemetry-api",
    } <= reqs


def test_compose_mounts_the_index_and_the_model_cache_read_only_and_requires_secrets():
    c = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())["services"]["copilot"]
    assert (
        all(v.endswith(":ro") for v in c["volumes"])
        and any("/data/index" in v for v in c["volumes"])
        and any("/models/hf" in v for v in c["volumes"])
    )
    assert c["environment"]["COPILOT_KEYS"].startswith("${COPILOT_KEYS:?") and c["environment"][
        "COPILOT_ADMIN_TOKEN"
    ].startswith("${COPILOT_ADMIN_TOKEN:?")
    assert (
        c["ports"][0].startswith("127.0.0.1:")
        and c["mem_limit"]
        and c["build"].startswith("${API_CONTEXT:?")
    )


def test_the_env_example_is_json_with_a_placeholder_and_no_secret_shaped_literal():
    text = (DEPLOY / ".env.example").read_text()
    line = next(x for x in text.splitlines() if x.startswith("COPILOT_KEYS="))
    assert (
        json.loads(line.split("=", 1)[1])[0]["key_sha256"].startswith("<")
        and "NEVER commit" in text
    )
    for p in DEPLOY.iterdir():
        if p.is_file():
            assert not any(s.search(p.read_text()) for s in SECRET_SHAPES), p.name


def test_the_build_context_has_the_code_and_nothing_else(tmp_path):
    info = BC.build(tmp_path / "ctx")
    ctx = tmp_path / "ctx"
    assert {p.name for p in ctx.iterdir()} == {
        "copilot",
        "common",
        "llmapi",
        "Dockerfile",
        "requirements-copilot.txt",
        ".dockerignore",
    }
    files = [p for p in ctx.rglob("*") if p.is_file()]
    assert not [
        p
        for p in files
        if p.suffix in {".gguf", ".safetensors", ".npy", ".sqlite", ".pyc", ".jsonl"}
        or p.name == ".env"
        or "__pycache__" in p.parts
    ]
    assert (
        info["bytes"] < 3_000_000
        and (ctx / "copilot" / "service.py").exists()
        and (ctx / "llmapi" / "app.py").exists()
    )
    assert not (ctx / "copilot" / "golden").exists()


def test_the_service_imports_in_the_container_layout(tmp_path):
    """The container has /app/{copilot,common,llmapi} and no repository: the package must import there (the ``_paths`` module is what makes both layouts work)."""
    import subprocess

    BC.build(tmp_path / "app")
    code = "import copilot.service as s, copilot.core, copilot.evaluate, copilot.ingest, copilot.guard; print(s._paths.ROOT if hasattr(s, '_paths') else 'ok')"
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path / "app",
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""} | _site_env(),
    )
    assert r.returncode == 0, r.stderr[-800:]


def _site_env() -> dict:
    """Keep the interpreter able to find installed packages (the venv's site-packages) without the repository on the path."""
    import site

    return {"PYTHONPATH": ":".join(site.getsitepackages())}
