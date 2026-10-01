"""Run model-written Python somewhere it cannot hurt you (much). Two backends, honest about what each stops.

    from common.sandbox import SubprocessSandbox, DockerSandbox
    res = SubprocessSandbox(timeout_s=10).run("print(sum(range(10)))", files={"data.csv": "a,b\\n1,2\\n"})
    res.stdout, res.stderr, res.returncode, res.timed_out, res.reason, res.files

SubprocessSandbox (no Docker needed; development and trusted-ish code)
  stops:        runaway CPU (wall timeout + RLIMIT_CPU), output floods (RLIMIT_FSIZE on the output files), huge
                files, leftover child processes (the whole process group is killed), leaking your environment
                variables (a minimal env), littering your project (a throw-away working directory)
  does NOT stop: reading any file you can read, using the network, using lots of memory on macOS (RLIMIT_AS is
                not enforced there), a fork bomb. It is NOT a security boundary. tests/test_sandbox.py proves each gap.

DockerSandbox (needs a running Docker daemon; use this for untrusted code)
  --network none, --memory/--memory-swap, --cpus, --pids-limit, --read-only root, a small tmpfs, all capabilities
  dropped, no-new-privileges, an unprivileged user, only the working directory mounted (read-only), and a hard
  wall-clock timeout that kills the CONTAINER (killing the docker CLI alone would leave it running).

Both return the same ``SandboxResult`` so the agent tool does not care which one it is given.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

MAX_OUTPUT_BYTES = 20_000


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    reason: str = (
        ""  # "", "timeout", "cpu limit", "file size limit", "killed (signal N)", "out of memory"
    )
    seconds: float = 0.0
    truncated: bool = False
    files: dict[str, int] = field(default_factory=dict)  # files the code created -> size in bytes

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def summary(self, max_chars: int = 4000, stderr_lines: int = 12) -> str:
        """What an agent should see: stdout, then a SHORT error tail, then any limit that was hit."""
        parts = []
        if self.stdout:
            out = (
                self.stdout
                if len(self.stdout) <= max_chars
                else self.stdout[:max_chars] + "\n[output truncated]"
            )
            parts.append(out)
        if self.stderr and not self.ok:
            tail = "\n".join(
                self.stderr.strip().splitlines()[-stderr_lines:]
            )  # the traceback's end is what matters
            parts.append("stderr:\n" + tail)
        if self.reason:
            parts.append(f"[stopped: {self.reason}]")
        if self.truncated and "[output truncated]" not in "".join(parts):
            parts.append("[output truncated]")
        if self.files:
            parts.append(
                "files created: "
                + ", ".join(f"{n} ({s} bytes)" for n, s in sorted(self.files.items()))
            )
        return "\n".join(parts) if parts else "(no output; use print() to see results)"


def _write_inputs(workdir: Path, code: str, files: Mapping[str, str | bytes | Path] | None) -> None:
    (workdir / "main.py").write_text(code)
    for name, content in (files or {}).items():
        if "/" in name or name.startswith(".") or name == "main.py":
            raise ValueError(f"invalid input file name {name!r}: plain file names only")
        target = workdir / name
        if isinstance(content, Path):
            shutil.copyfile(content, target)
        elif isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)


def _created_files(workdir: Path, inputs: set[str]) -> dict[str, int]:
    out = {}
    for p in sorted(workdir.iterdir()):
        if p.is_file() and p.name not in inputs and p.name != "main.py":
            out[p.name] = p.stat().st_size
    return out


def _read_capped(path: Path, limit: int) -> tuple[str, bool]:
    size = path.stat().st_size
    with open(path, "rb") as f:
        data = f.read(limit)
    return data.decode("utf-8", errors="replace"), size > limit


def _describe_signal(returncode: int | None) -> str:
    if returncode is None or returncode >= 0:
        return ""
    sig = -returncode
    return {
        signal.SIGXCPU: "cpu limit",
        signal.SIGXFSZ: "file size limit",
        signal.SIGKILL: "killed (SIGKILL: out of memory or killed by a limit)",
        signal.SIGSEGV: "crashed (segmentation fault)",
    }.get(sig, f"killed (signal {sig})")


# The launcher runs INSIDE the child: it imposes the limits on itself, then runs main.py. Doing it there (instead of
# subprocess's preexec_fn) is safe when the parent has threads, which an agent's tool executor does.
_LAUNCHER = """
import os, resource, sys, traceback
for name, value in {limits!r}.items():
    if value is not None and hasattr(resource, name):
        resource.setrlimit(getattr(resource, name), (value, value))
os.chdir({workdir!r})
sys.path.insert(0, {workdir!r})
try:
    exec(compile(open("main.py").read(), "main.py", "exec"), {{"__name__": "__main__", "__file__": "main.py"}})
except SystemExit:
    raise
except BaseException:
    etype, err, tb = sys.exc_info()
    traceback.print_exception(etype, err, tb.tb_next)  # drop THIS launcher's frame: only the code's own frames
    sys.exit(1)
"""


class SubprocessSandbox:
    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        cpu_s: int | None = None,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        max_file_bytes: int = 5_000_000,
        max_open_files: int = 64,
        memory_mb: int | None = None,
        python: str | None = None,
    ):
        if resource is None:
            raise RuntimeError(
                "SubprocessSandbox needs a POSIX system (use DockerSandbox on Windows)"
            )
        self.timeout_s = timeout_s
        self.cpu_s = cpu_s if cpu_s is not None else max(1, int(timeout_s))
        self.max_output_bytes = max_output_bytes
        self.max_file_bytes = max_file_bytes
        self.max_open_files = max_open_files
        self.memory_mb = memory_mb  # enforced on Linux only
        self.python = python or sys.executable

    def run(
        self, code: str, files: Mapping[str, str | bytes | Path] | None = None
    ) -> SandboxResult:
        root = Path(tempfile.mkdtemp(prefix="sandbox-"))
        work = root / "work"
        work.mkdir()
        try:
            _write_inputs(work, code, files)
            limits = {
                "RLIMIT_CPU": self.cpu_s,
                "RLIMIT_FSIZE": self.max_file_bytes,
                "RLIMIT_NOFILE": self.max_open_files + 16,
                "RLIMIT_CORE": 0,
                "RLIMIT_AS": self.memory_mb * 1024 * 1024 if self.memory_mb else None,
            }
            launcher = _LAUNCHER.format(limits=limits, workdir=str(work))
            out_path, err_path = root / "stdout", root / "stderr"
            env = {
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "HOME": str(work),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            t0 = time.perf_counter()
            res = SandboxResult()
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                # stdout/stderr go to FILES whose growth RLIMIT_FSIZE bounds: a flood cannot fill memory or disk
                proc = subprocess.Popen(
                    [self.python, "-I", "-c", launcher],
                    stdout=out, stderr=err, stdin=subprocess.DEVNULL, cwd=str(work), env=env,
                    start_new_session=True,
                )  # fmt: skip
                # RLIMIT_FSIZE applies to the files the CHILD writes; ours are the same inode, so it bounds them
                try:
                    proc.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    res.timed_out, res.reason = True, f"timeout after {self.timeout_s:g}s"
                finally:
                    try:  # also reaps children the code started and left running
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    proc.wait()
            res.seconds = time.perf_counter() - t0
            res.returncode = None if res.timed_out else proc.returncode
            if not res.reason:
                res.reason = _describe_signal(proc.returncode)
            err_text = err_path.read_bytes()[-4000:].decode("utf-8", errors="replace")
            if (
                not res.reason and "File too large" in err_text
            ):  # Python turns SIGXFSZ on a write into an OSError
                res.reason = "file size limit"
            res.stdout, t1 = _read_capped(out_path, self.max_output_bytes)
            res.stderr, t2 = _read_capped(err_path, self.max_output_bytes)
            res.truncated = t1 or t2
            res.files = _created_files(work, set(files or {}))
            return res
        finally:
            shutil.rmtree(root, ignore_errors=True)


class DockerSandbox:
    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        timeout_s: float = 20.0,
        memory: str = "128m",
        cpus: str = "1",
        pids: int = 64,
        network: str = "none",
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ):
        self.image, self.timeout_s, self.memory, self.cpus = image, timeout_s, memory, cpus
        self.pids, self.network, self.max_output_bytes = pids, network, max_output_bytes

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return (
                subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
            )
        except (subprocess.TimeoutExpired, OSError):
            return False

    def command(self, name: str, workdir: Path) -> list[str]:
        cap = self.max_output_bytes
        script = (
            f"python -I main.py >/tmp/o 2>/tmp/e; rc=$?; head -c {cap} /tmp/o; "
            f"printf '\\n===STDERR===\\n'; head -c {cap} /tmp/e; printf '\\n===RC=%s' $rc"
        )
        return [
            "docker", "run", "--rm", "--name", name,
            "--network", self.network,
            "--memory", self.memory, "--memory-swap", self.memory,
            "--cpus", self.cpus, "--pids-limit", str(self.pids),
            "--read-only", "--tmpfs", "/tmp:rw,size=16m,noexec",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "-v", f"{workdir}:/work:ro", "-w", "/work",
            self.image, "sh", "-c", script,
        ]  # fmt: skip

    def run(
        self, code: str, files: Mapping[str, str | bytes | Path] | None = None
    ) -> SandboxResult:
        root = Path(tempfile.mkdtemp(prefix="sandbox-"))
        os.chmod(root, 0o755)
        name = f"sandbox-{uuid.uuid4().hex[:12]}"
        try:
            _write_inputs(root, code, files)
            for p in root.iterdir():
                os.chmod(p, 0o644)
            t0 = time.perf_counter()
            res = SandboxResult()
            try:
                proc = subprocess.run(
                    self.command(name, root), capture_output=True, timeout=self.timeout_s
                )
                raw = proc.stdout.decode("utf-8", errors="replace")
                docker_err = proc.stderr.decode("utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["docker", "kill", name], capture_output=True, timeout=30
                )  # the CLI dying is not enough
                res.timed_out, res.reason = True, f"timeout after {self.timeout_s:g}s"
                raw, docker_err = "", ""
            res.seconds = time.perf_counter() - t0
            if not res.timed_out:
                head, _, rest = raw.partition("\n===STDERR===\n")
                err, _, rc = rest.rpartition("\n===RC=")
                res.stdout, res.stderr = head, err
                res.returncode = int(rc) if rc.strip().lstrip("-").isdigit() else None
                if res.returncode is None:
                    res.stderr = docker_err or raw
                    res.returncode = proc.returncode or 1
                if res.returncode == 137:
                    res.reason = "killed (SIGKILL: out of memory or killed by a limit)"
                elif res.returncode == 139:
                    res.reason = "crashed (segmentation fault)"
            res.truncated = len(res.stdout) >= self.max_output_bytes
            return res
        finally:
            shutil.rmtree(root, ignore_errors=True)
