"""Tests for Week 11 Day 6: the build context, the Dockerfile and platform files (parsed, not just present), the docker helpers against a fake CLI, and one live build-and-run when Docker is available."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "deploy"))

import build_context as BC  # noqa: E402
import dockerlab as D  # noqa: E402

DEPLOY = HERE / "deploy"
SECRET_SHAPES = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"\b[0-9a-f]{64}\b"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


# ----------------------------------------------------------------------------- build context


def test_the_build_context_holds_the_package_docs_and_dockerfile_and_nothing_secret(tmp_path):
    info = BC.build(tmp_path / "ctx", limit_weeks=11)
    ctx = tmp_path / "ctx"
    names = {p.name for p in ctx.iterdir()}
    assert {"llmapi", "docs", "Dockerfile", "requirements-api.txt", ".dockerignore"} <= names
    assert info["docs"] == len(list((ctx / "docs").rglob("*.md"))) > 20
    files = [p for p in ctx.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix in {".gguf", ".safetensors", ".env", ".pyc"}]
    assert not [p for p in files if "__pycache__" in p.parts or p.name == ".env"]
    assert info["bytes"] < 5_000_000  # no weights, no outputs


def test_limit_weeks_and_a_custom_docs_directory(tmp_path):
    few = BC.build(tmp_path / "a", limit_weeks=1)
    many = BC.build(tmp_path / "b", limit_weeks=11)
    assert 0 < few["docs"] < many["docs"]
    docs = tmp_path / "mydocs"
    docs.mkdir()
    (docs / "x.md").write_text("# X\n\ntext\n")
    custom = BC.build(tmp_path / "c", docs=docs)
    assert custom["docs"] == 1 and (tmp_path / "c/docs/x.md").exists()


def test_rebuilding_into_an_existing_directory_replaces_it(tmp_path):
    out = tmp_path / "ctx"
    BC.build(out, limit_weeks=1)
    (out / "stale.txt").write_text("old")
    BC.build(out, limit_weeks=1)
    assert not (out / "stale.txt").exists()


# ----------------------------------------------------------------------------- the Dockerfile


def dockerfile_lines():
    text = (DEPLOY / "Dockerfile").read_text().replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_the_dockerfile_runs_unprivileged_with_a_healthcheck_and_the_dependency_layer_first():
    lines = dockerfile_lines()
    instr = [line.split()[0] for line in lines]
    assert lines[0].startswith("FROM python:3.12-slim") and "latest" not in lines[0]
    assert instr.index("USER") > instr.index("RUN")  # root only to set things up
    user = next(line for line in lines if line.startswith("USER")).split()[1]
    assert user not in ("root", "0")
    assert "HEALTHCHECK" in instr and instr[-1] == "CMD"
    copy_req = next(i for i, line in enumerate(lines) if "requirements-api.txt" in line)
    copy_code = next(i for i, line in enumerate(lines) if line.startswith("COPY llmapi"))
    pip = next(i for i, line in enumerate(lines) if "pip install" in line)
    assert copy_req < pip < copy_code  # editing code does not reinstall packages


def test_no_secret_or_weights_are_baked_into_the_dockerfile():
    text = (DEPLOY / "Dockerfile").read_text()
    for w in ("LLMAPI_KEYS", "ADMIN_TOKEN", "PASSWORD", ".gguf"):
        assert w not in text.replace(
            "# secrets (LLMAPI_KEYS, LLMAPI_ADMIN_TOKEN) are NOT baked in", ""
        )
    assert not any(s.search(text) for s in SECRET_SHAPES)
    env_lines = [line for line in dockerfile_lines() if line.startswith("ENV")]
    assert env_lines and all(
        "KEY" not in line.upper().replace("LLMAPI_DOCS_DIR", "") for line in env_lines
    )


def test_the_dockerignore_excludes_env_files_models_and_vcs():
    ignore = (DEPLOY / ".dockerignore").read_text().split()
    assert {".env", "*.gguf", ".git"} <= set(ignore)


def test_the_runtime_requirements_do_not_pull_in_the_ml_stack():
    reqs = "\n".join(
        line
        for line in (DEPLOY / "requirements-api.txt").read_text().lower().splitlines()
        if line.strip() and not line.startswith("#")
    )
    for heavy in ("torch", "transformers", "tensorflow", "llama"):
        assert heavy not in reqs
    for needed in ("fastapi", "uvicorn", "httpx", "rank-bm25"):
        assert needed in reqs


# ----------------------------------------------------------------------------- compose and platform files


def test_compose_orders_startup_by_health_mounts_models_read_only_and_requires_secrets():
    c = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())
    s = c["services"]
    assert set(s) == {"llama-extractor", "llama-chat", "api"}
    for name in ("llama-extractor", "llama-chat"):
        assert s[name]["healthcheck"]["test"] and any(v.endswith(":ro") for v in s[name]["volumes"])
        assert "--metrics" in s[name]["command"]
        assert s[name]["restart"] == "unless-stopped"
    api = s["api"]
    assert all(
        api["depends_on"][m]["condition"] == "service_healthy"
        for m in ("llama-extractor", "llama-chat")
    )
    # secrets come from the environment and the compose file refuses to start without them
    assert api["environment"]["LLMAPI_KEYS"].startswith("${LLMAPI_KEYS:?")
    assert api["environment"]["LLMAPI_ADMIN_TOKEN"].startswith("${LLMAPI_ADMIN_TOKEN:?")
    backends = json.loads(api["environment"]["LLMAPI_BACKENDS"])
    assert set(backends) == {"order-extractor", "qwen-chat"}
    assert {u.split("//")[1].split(":")[0] for u in backends.values()} <= set(
        s
    )  # service DNS names


def test_the_fly_render_and_modal_files_parse_keep_secrets_out_and_say_they_were_not_run():
    fly = tomllib.loads((DEPLOY / "fly.toml").read_text())
    assert fly["http_service"]["internal_port"] == 8000 and fly["http_service"]["checks"]
    assert "LLMAPI_KEYS" not in fly["env"] and "LLMAPI_ADMIN_TOKEN" not in fly["env"]
    render = yaml.safe_load((DEPLOY / "render.yaml").read_text())
    env = {e["key"]: e for e in render["services"][0]["envVars"]}
    assert env["LLMAPI_KEYS"]["sync"] is False and env["LLMAPI_ADMIN_TOKEN"]["sync"] is False
    compile((DEPLOY / "modal_app.py").read_text(), "modal_app.py", "exec")
    for f in ("fly.toml", "render.yaml", "modal_app.py"):
        assert "NOT RUN" in (DEPLOY / f).read_text()


def test_the_env_example_is_valid_json_with_a_placeholder_and_no_real_secret():
    text = (DEPLOY / ".env.example").read_text()
    line = next(x for x in text.splitlines() if x.startswith("LLMAPI_KEYS="))
    entries = json.loads(line.split("=", 1)[1])
    assert entries[0]["key_sha256"].startswith("<")  # a placeholder, not a hash
    assert not any(s.search(text) for s in SECRET_SHAPES)
    assert "NEVER commit" in text


def test_no_secret_shaped_literal_in_any_deploy_file():
    for p in DEPLOY.iterdir():
        if p.is_file():
            text = p.read_text()
            assert not any(s.search(text) for s in SECRET_SHAPES), p.name


# ----------------------------------------------------------------------------- the docker helpers against a fake CLI


class FakeCLI:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, *args, check=True, timeout=600):
        self.calls.append(args)
        for prefix, out in self.replies.items():
            if args[: len(prefix)] == prefix:
                code, text = out if isinstance(out, tuple) else (0, out)
                return subprocess.CompletedProcess(args, code, text, "")
        return subprocess.CompletedProcess(args, 0, "", "")


def test_container_run_publishes_on_loopback_only_and_passes_env_and_extra_args(monkeypatch):
    fake = FakeCLI({})
    monkeypatch.setattr(D, "docker", fake)
    D.Container(
        "n", "img:1", 9000, env={"A": "1"}, extra_args=["--memory", "256m"], container_port=8000
    ).run()
    run = next(c for c in fake.calls if c[0] == "run")
    assert (
        "127.0.0.1:9000:8000" in run and ("-e", "A=1") == run[run.index("-e") : run.index("-e") + 2]
    )
    assert run[-3:] == ("--memory", "256m", "img:1") or run[-1] == "img:1"
    assert fake.calls[0][:3] == ("rm", "-f", "n")  # a stale container of that name is removed first


def test_health_state_logs_exit_code_and_stop_use_the_cli_outputs(monkeypatch):
    fake = FakeCLI(
        {
            (
                "inspect",
                "c",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            ): "healthy\n",
            ("inspect", "c", "--format", "{{.State.Status}}"): "running\n",
            ("inspect", "c", "--format", "{{.State.ExitCode}}"): "0\n",
        }
    )
    monkeypatch.setattr(D, "docker", fake)
    c = D.Container("c", "i", 1)
    assert c.health() == "healthy" and c.state() == "running" and c.exit_code() == 0
    assert c.stop(3) >= 0 and ("stop", "-t", "3", "c") in fake.calls
    monkeypatch.setattr(D, "docker", FakeCLI({}))
    assert D.Container("c", "i", 1).health() == "gone"  # an empty answer means no such container
    assert D.Container("c", "i", 1).exit_code() == -1


def test_wait_healthy_returns_when_healthy_raises_on_exit_and_on_timeout(monkeypatch):
    c = D.Container("c", "i", 1)
    states = iter(["starting", "starting", "healthy"])
    monkeypatch.setattr(c, "health", lambda: next(states))
    monkeypatch.setattr(c, "state", lambda: "running")
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    assert c.wait_healthy(5) >= 0
    monkeypatch.setattr(c, "health", lambda: "starting")
    monkeypatch.setattr(c, "state", lambda: "exited")
    monkeypatch.setattr(c, "logs", lambda: "boom")
    with pytest.raises(RuntimeError, match="boom"):
        c.wait_healthy(5)
    monkeypatch.setattr(c, "state", lambda: "running")
    clock = iter([0, 1, 2, 100, 101])
    monkeypatch.setattr(D.time, "time", lambda: next(clock))
    with pytest.raises(TimeoutError):
        c.wait_healthy(10)


def test_build_reports_seconds_and_raises_with_the_tail_of_the_output_on_failure(monkeypatch):
    monkeypatch.setattr(D, "docker", FakeCLI({("build",): "Successfully built"}))
    secs, out = D.build("ctx", "t")
    assert secs >= 0 and "Successfully" in out
    monkeypatch.setattr(D, "docker", FakeCLI({("build",): (1, "step 3 failed: no such file")}))
    with pytest.raises(RuntimeError, match="no such file"):
        D.build("ctx", "t")


def test_image_helpers_parse_the_cli_output(monkeypatch):
    fake = FakeCLI(
        {
            ("image", "inspect", "t", "--format", "{{.Size}}"): "228000000\n",
            ("image", "inspect", "t", "--format", "{{.Config.User}}"): "app\n",
            ("image", "inspect", "t", "--format", "{{json .Config.Env}}"): '["PORT=8000","X=1"]',
        }
    )
    monkeypatch.setattr(D, "docker", fake)
    assert D.image_size_mb("t") == 228.0 and D.image_user("t") == "app"
    assert D.env_in_image("t") == ["PORT=8000", "X=1"]


def test_available_is_false_without_a_docker_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(D, "docker", boom)
    assert D.available() is False


# ----------------------------------------------------------------------------- live (skipped without Docker)

live = pytest.mark.skipif(not D.available(), reason="docker is not available")


@live
def test_live_build_run_health_and_clean_logs(tmp_path):
    BC.build(tmp_path / "ctx", limit_weeks=2)
    D.build(str(tmp_path / "ctx"), "order-api:w11-test")
    assert D.image_user("order-api:w11-test") == "app"
    names = [e.split("=", 1)[0] for e in D.env_in_image("order-api:w11-test")]
    assert not [n for n in names if "KEY" in n.upper() and n != "GPG_KEY"]
    secret = "tok-live-test-0001"
    keys = json.dumps([{"user": "u", "key": "sk-live-test-user"}])
    c = D.Container(
        "w11-test-api",
        "order-api:w11-test",
        __import__("llamacpp").free_port(),
        env={
            "LLMAPI_KEYS": keys,
            "LLMAPI_ADMIN_TOKEN": secret,
            "LLMAPI_BACKEND_URL": "http://127.0.0.1:1",
        },
    )
    try:
        c.run()
        c.wait_healthy(60)
        import httpx

        assert httpx.get(f"{c.url}/healthz").status_code == 200
        assert httpx.get(f"{c.url}/readyz").status_code == 503  # alive but no model behind it
        assert c.health() == "healthy"
        assert secret not in c.logs()
        assert "tok-live-test-0001" not in D.image_history("order-api:w11-test")
    finally:
        c.remove()
