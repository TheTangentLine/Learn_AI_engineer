"""Bring the whole deployment up with ``docker compose`` (two llama.cpp containers + the API container built from this repository), wait until it is ready, change one
setting of the API, and tear it down. The compose file is ``solutions/deploy/docker-compose.yml``; nothing here is specific to a machine except the models directory."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import httpx

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
PROJECT = "w11weekly"
KEY = "sk-weekly-demo-alice"
ADMIN = "adm-weekly-demo"
USERS = [
    {
        "user": "alice",
        "key": KEY,
        "rpm": 6000,
        "burst": 500,
        "daily_tokens": 50_000_000,
        "max_concurrent": 64,
        "max_tokens_cap": 400,
    }
]


def compose_env(
    models_dir: Path, context: Path, min_score: float = 0.0, port: int = 8000, abstain: bool = False
) -> dict:
    return {
        **os.environ,
        "MODELS_DIR": str(models_dir),
        "API_CONTEXT": str(context),
        "LLMAPI_KEYS": json.dumps(USERS),
        "LLMAPI_ADMIN_TOKEN": ADMIN,
        "LLMAPI_MIN_SCORE": str(min_score),
        "LLMAPI_ABSTAIN": "1" if abstain else "0",
        "COMPOSE_PROJECT_NAME": PROJECT,
        "API_PORT": str(port),
    }


def compose(*args: str, env: dict, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-f", str(DEPLOY / "docker-compose.yml"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def up(env: dict, *services: str) -> float:
    """``docker compose up -d --build`` (all services, or just the named ones); returns the seconds it took. Raises with the output on failure."""
    t0 = time.perf_counter()
    p = compose("up", "-d", "--build", *services, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"docker compose up failed:\n{(p.stdout + p.stderr)[-2000:]}")
    return time.perf_counter() - t0


def down(env: dict) -> None:
    compose("down", "--volumes", "--remove-orphans", env=env, timeout=120)


def wait_ready(url: str, timeout: float = 120) -> float:
    """Poll ``/readyz`` until every model server answers; returns the seconds waited."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{url}/readyz", timeout=2)
            last = r.text
            if r.status_code == 200:
                return time.time() - t0
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"{url} was not ready after {timeout}s: {last}")


def container_stats(env: dict) -> list[dict]:
    """``docker stats --no-stream`` for the project's containers: name, memory, CPU."""
    ids = compose("ps", "-q", env=env).stdout.split()
    if not ids:
        return []
    p = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids],
        capture_output=True,
        text=True,
        timeout=60,
    )
    rows = []
    for line in p.stdout.splitlines():
        d = json.loads(line)
        rows.append({"name": d["Name"], "memory": d["MemUsage"], "cpu": d["CPUPerc"]})
    return rows
