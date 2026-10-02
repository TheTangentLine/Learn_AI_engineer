"""Start the whole serving stack as real processes: llama-server (the model) and the ``llmapi`` FastAPI app in front of it. Used by the Day 4, 5, 6 and 7 solutions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import llamacpp as L

HERE = Path(__file__).resolve().parent

DEMO_USERS = [
    {
        "user": "alice",
        "key": "sk-demo-alice",
        "rpm": 600,
        "burst": 100,
        "daily_tokens": 5_000_000,
        "max_concurrent": 32,
        "max_tokens_cap": 400,
    },
    {
        "user": "bob",
        "key": "sk-demo-bob",
        "rpm": 30,
        "burst": 3,
        "daily_tokens": 5_000,
        "max_concurrent": 2,
        "max_tokens_cap": 200,
    },
]


class ApiServer:
    """``python -m llmapi`` in a subprocess on a free port, configured only through the environment (exactly as it will run in a container)."""

    def __init__(
        self,
        backend_url: str,
        *,
        users: list[dict] | None = None,
        max_inflight: int = 4,
        max_queue: int = 8,
        admin_token: str | None = "adm-demo",
        extra_env: dict | None = None,
        log: Path | None = None,
    ):
        self.port = L.free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = log or (L.OUT / f"api_{self.port}.log")
        self.env = {
            **os.environ,
            "PYTHONPATH": str(HERE),
            "PORT": str(self.port),
            "HOST": "127.0.0.1",
            "LLMAPI_BACKEND_URL": backend_url,
            "LLMAPI_KEYS": json.dumps(users or DEMO_USERS),
            "LLMAPI_MAX_INFLIGHT": str(max_inflight),
            "LLMAPI_MAX_QUEUE": str(max_queue),
            **({"LLMAPI_ADMIN_TOKEN": admin_token} if admin_token else {}),
            **(extra_env or {}),
        }
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> ApiServer:
        L.OUT.mkdir(parents=True, exist_ok=True)
        self._logf = open(self.log, "w")  # noqa: SIM115
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "llmapi"],
            cwd=HERE,
            env=self.env,
            stdout=self._logf,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"the API exited early; see {self.log}")
            try:
                if httpx.get(f"{self.url}/healthz", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.2)
        raise TimeoutError("the API did not start")

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._logf.close()


class Stack:
    """llama-server + the API. ``with Stack(model) as s:`` gives ``s.llama`` (direct model server) and ``s.api`` (the public endpoint)."""

    def __init__(self, model: Path, *, parallel: int = 4, ctx: int = 4096, **api_kw):
        self.llama = L.LlamaServer(model, parallel=parallel, ctx=ctx)
        self.api_kw = {"max_inflight": parallel, **api_kw}
        self.api: ApiServer | None = None

    def __enter__(self) -> Stack:
        self.llama.__enter__()
        try:
            self.api = ApiServer(self.llama.url, **self.api_kw).__enter__()
        except BaseException:
            self.llama.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self.api:
            self.api.__exit__(*exc)
        self.llama.__exit__(*exc)
