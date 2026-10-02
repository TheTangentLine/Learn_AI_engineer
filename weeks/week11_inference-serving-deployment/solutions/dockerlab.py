"""Thin, tested helpers around the ``docker`` CLI for Day 6 and Day 7: build, run with environment secrets, wait for health, read logs, measure stop time, clean up."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field

import httpx


def docker(*args: str, check: bool = True, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check, timeout=timeout
    )


def available() -> bool:
    try:
        return docker("info", check=False, timeout=15).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build(context: str, tag: str) -> tuple[float, str]:
    """(seconds, the build output). Raises with the output's tail if the build fails."""
    t0 = time.perf_counter()
    p = docker("build", "-t", tag, context, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"docker build failed:\n{(p.stdout + p.stderr)[-1500:]}")
    return time.perf_counter() - t0, p.stdout + p.stderr


def image_size_mb(tag: str) -> float:
    return int(docker("image", "inspect", tag, "--format", "{{.Size}}").stdout.strip()) / 1e6


def image_history(tag: str) -> str:
    return docker("history", "--no-trunc", tag, "--format", "{{.CreatedBy}}").stdout


def image_user(tag: str) -> str:
    return docker("image", "inspect", tag, "--format", "{{.Config.User}}").stdout.strip()


@dataclass
class Container:
    name: str
    image: str
    port: int
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    container_port: int = 8000

    def run(self) -> Container:
        docker("rm", "-f", self.name, check=False)
        args = [
            "run",
            "-d",
            "--name",
            self.name,
            "-p",
            f"127.0.0.1:{self.port}:{self.container_port}",
        ]
        for k, v in self.env.items():
            args += ["-e", f"{k}={v}"]
        docker(*args, *self.extra_args, self.image)
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def health(self) -> str:
        """'starting' | 'healthy' | 'unhealthy' | 'none' (as reported by the image's HEALTHCHECK)."""
        out = docker(
            "inspect",
            self.name,
            "--format",
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            check=False,
        ).stdout.strip()
        return out or "gone"

    def wait_healthy(self, timeout: float = 60) -> float:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.health() == "healthy":
                return time.time() - t0
            if self.state() in ("exited", "dead"):
                raise RuntimeError(f"the container exited:\n{self.logs()[-800:]}")
            time.sleep(0.5)
        raise TimeoutError(
            f"{self.name} did not become healthy in {timeout}s (state {self.health()})\n{self.logs()[-500:]}"
        )

    def state(self) -> str:
        return docker(
            "inspect", self.name, "--format", "{{.State.Status}}", check=False
        ).stdout.strip()

    def logs(self) -> str:
        p = docker("logs", self.name, check=False)
        return p.stdout + p.stderr

    def stop(self, timeout: int = 10) -> float:
        t0 = time.perf_counter()
        docker("stop", "-t", str(timeout), self.name, check=False)
        return time.perf_counter() - t0

    def exit_code(self) -> int:
        return int(
            docker(
                "inspect", self.name, "--format", "{{.State.ExitCode}}", check=False
            ).stdout.strip()
            or -1
        )

    def remove(self) -> None:
        docker("rm", "-f", self.name, check=False)


def env_in_image(tag: str) -> list[str]:
    """The ENV baked into the image (a leaked secret would be visible here)."""
    out = docker("image", "inspect", tag, "--format", "{{json .Config.Env}}").stdout
    return json.loads(out)


def wait_http(url: str, timeout: float = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False
