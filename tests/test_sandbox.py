"""Tests for common/sandbox.py. The SubprocessSandbox tests include the things it does NOT stop: they document
the gaps, and fail loudly if a change quietly closes (or widens) one. Docker tests skip without a daemon."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from common.sandbox import DockerSandbox, SandboxResult, SubprocessSandbox

SB = SubprocessSandbox(timeout_s=3)


# ----------------------------------------------------------------------------- basics


def test_stdout_stderr_returncode_and_exceptions():
    r = SB.run("print('hi', 1 + 1)")
    assert (r.stdout, r.returncode, r.ok, r.reason, r.timed_out) == ("hi 2\n", 0, True, "", False)
    r = SB.run("import sys\nprint('out')\nprint('err', file=sys.stderr)\nsys.exit(3)")
    assert r.returncode == 3 and not r.ok and r.stdout == "out\n" and r.stderr == "err\n"
    r = SB.run("def f():\n    raise ValueError('boom')\nf()")
    assert r.returncode == 1 and "ValueError: boom" in r.stderr and "line 2" in r.stderr
    assert SB.run("x = ").returncode == 1 and "SyntaxError" in SB.run("x = ").stderr


def test_input_files_are_copied_and_created_files_are_reported():
    r = SB.run(
        "print(open('data.csv').read().strip().splitlines()[-1])\nopen('out.txt','w').write('hello')",
        files={"data.csv": "a,b\n1,2\n"},
    )
    assert r.stdout == "1,2\n" and r.files == {"out.txt": 5}
    assert (
        SB.run("print(open('b.bin','rb').read())", files={"b.bin": b"\x00\x01"}).stdout
        == "b'\\x00\\x01'\n"
    )


def test_file_names_are_validated():
    for bad in ("../x", "a/b", ".hidden", "main.py"):
        with pytest.raises(ValueError, match="invalid input file name"):
            SB.run("pass", files={bad: "x"})


def test_each_run_starts_fresh_in_a_throwaway_directory():
    SB.run("open('state.txt', 'w').write('1')")
    r = SB.run(
        "import os; print(os.path.exists('state.txt'), os.getcwd() != os.path.expanduser('~'))"
    )
    assert r.stdout == "False True\n", "no state survives between runs"
    cwd = SB.run("import os; print(os.getcwd())").stdout.strip()
    assert "sandbox-" in cwd and not Path(cwd).exists(), "the directory is removed afterwards"


def test_pandas_and_numpy_are_importable_inside():
    r = SB.run(
        "import pandas as pd, numpy as np\nprint(pd.DataFrame({'a': [1, 2]}).a.sum(), np.arange(3).sum())"
    )
    assert r.stdout == "3 3\n", r.stderr


# ----------------------------------------------------------------------------- what it DOES stop


def test_infinite_loop_is_stopped_quickly():
    t0 = time.perf_counter()
    r = SubprocessSandbox(timeout_s=1).run("while True: pass")
    assert time.perf_counter() - t0 < 5 and not r.ok and (r.timed_out or r.reason == "cpu limit")


def test_wall_clock_timeout_for_code_that_sleeps_rather_than_burns_cpu():
    r = SubprocessSandbox(timeout_s=0.5, cpu_s=100).run("import time; time.sleep(30)")
    assert (
        r.timed_out
        and r.returncode is None
        and r.reason == "timeout after 0.5s"
        and 0.4 < r.seconds < 5
    )


def test_cpu_limit_kills_a_busy_loop_even_when_the_wall_clock_is_generous():
    r = SubprocessSandbox(timeout_s=30, cpu_s=1).run("while True: pass")
    assert r.returncode == -24 and r.reason == "cpu limit" and not r.timed_out and r.seconds < 10


def test_output_flood_is_bounded_and_flagged():
    t0 = time.perf_counter()
    r = SubprocessSandbox(timeout_s=5, max_output_bytes=1000, max_file_bytes=200_000).run(
        "while True: print('x' * 1000)"
    )
    assert len(r.stdout) == 1000 and r.truncated and not r.ok and time.perf_counter() - t0 < 5
    assert r.reason == "file size limit"
    assert "[output truncated]" in r.summary()


def test_big_file_writes_hit_the_file_size_limit():
    r = SubprocessSandbox(timeout_s=5, max_file_bytes=100_000).run(
        "open('big.bin', 'wb').write(b'x' * 10_000_000)"
    )
    assert not r.ok and r.reason == "file size limit" and r.files.get("big.bin", 0) <= 100_000


def test_child_processes_are_killed_with_the_group(tmp_path):
    marker = tmp_path / "pid"
    code = f"import subprocess, time\np = subprocess.Popen(['sleep', '60'])\nopen({str(marker)!r}, 'w').write(str(p.pid))\ntime.sleep(60)"
    r = SubprocessSandbox(timeout_s=1, cpu_s=50).run(code)
    pid = int(marker.read_text())
    time.sleep(0.3)
    assert r.timed_out
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the grandchild is gone too


def test_a_child_that_outlives_a_successful_run_is_reaped(tmp_path):
    marker = tmp_path / "pid"
    code = f"import subprocess\np = subprocess.Popen(['sleep', '60'])\nopen({str(marker)!r}, 'w').write(str(p.pid))\nprint('done')"
    assert SB.run(code).ok
    pid = int(marker.read_text())
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_environment_variables_are_not_inherited(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_API_KEY", "sk-test-123")
    r = SB.run("import os; print(sorted(os.environ))")
    assert "SUPER_SECRET_API_KEY" not in r.stdout and "sk-test-123" not in r.stdout
    assert "PATH" in r.stdout and "HOME" in r.stdout
    exact = SB.run("import os; print(os.environ['PATH'])")
    assert exact.stdout.strip() == "/usr/bin:/bin", (
        "a minimal PATH, not yours (which may contain secrets-bearing tool dirs)"
    )


def test_many_sandboxes_in_parallel_threads_do_not_interfere():
    results = [None] * 6

    def work(i):
        results[i] = SB.run(
            f"import time; time.sleep(0.1); print({i} * 7)\nopen('f{i}.txt','w').write('x')"
        )

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [r.stdout.strip() for r in results] == [str(i * 7) for i in range(6)]
    assert all(set(r.files) == {f"f{i}.txt"} for i, r in enumerate(results))


# ----------------------------------------------------------------------------- what it does NOT stop (documented gaps)


def test_GAP_code_can_read_any_file_the_user_can_read(tmp_path):
    canary = tmp_path / "canary.txt"
    canary.write_text("TOP-SECRET-CANARY")
    r = SB.run(f"print(open({str(canary)!r}).read())")
    assert r.stdout.strip() == "TOP-SECRET-CANARY", (
        "a subprocess sandbox is NOT a confidentiality boundary"
    )


def test_GAP_code_can_open_network_connections():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        r = SB.run(
            f"import socket\ns = socket.create_connection(('127.0.0.1', {port}), timeout=2)\nprint('connected')"
        )
        assert r.stdout.strip() == "connected", "no network isolation without a container"
    finally:
        server.close()


def test_GAP_code_can_write_outside_its_working_directory(tmp_path):
    target = tmp_path / "escaped.txt"
    r = SB.run(f"open({str(target)!r}, 'w').write('x')")
    assert r.ok and target.read_text() == "x"


# ----------------------------------------------------------------------------- the result type


def test_summary_is_what_an_agent_should_see():
    assert SandboxResult(stdout="3\n", returncode=0).summary() == "3\n"
    assert SandboxResult(returncode=0).summary() == "(no output; use print() to see results)"
    tb = "\n".join(f"line {i}" for i in range(30)) + "\nValueError: boom"
    s = SandboxResult(stderr=tb, returncode=1).summary(stderr_lines=3)
    assert s == "stderr:\nline 28\nline 29\nValueError: boom" or s.endswith("ValueError: boom")
    assert "line 5" not in s, "only the END of a traceback is shown"
    assert (
        "[stopped: timeout after 2s]"
        in SandboxResult(timed_out=True, reason="timeout after 2s").summary()
    )
    long = SandboxResult(stdout="y" * 5000, returncode=0).summary(max_chars=100)
    assert len(long) < 140 and long.endswith("[output truncated]")
    assert "out.txt (5 bytes)" in SandboxResult(returncode=0, files={"out.txt": 5}).summary()
    assert "stderr:" not in SandboxResult(stderr="warning", returncode=0, stdout="ok").summary(), (
        "stderr only on failure"
    )


# ----------------------------------------------------------------------------- Docker (skipped without a daemon)

docker = pytest.mark.skipif(not DockerSandbox.available(), reason="Docker daemon not running")
DK = DockerSandbox(timeout_s=60)


@docker
def test_docker_runs_code_and_reports_stdout_stderr_and_exit_code():
    r = DK.run("print(6 * 7)")
    assert r.ok and r.stdout.strip() == "42", r.stderr
    r = DK.run("import sys\nprint('err', file=sys.stderr)\nsys.exit(4)")
    assert r.returncode == 4 and "err" in r.stderr
    r = DK.run("raise ValueError('boom')")
    assert r.returncode == 1 and "ValueError: boom" in r.stderr


@docker
def test_docker_has_no_network():
    r = DK.run(
        "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n    print('CONNECTED')\nexcept OSError as e:\n    print('blocked', type(e).__name__)"
    )
    assert r.stdout.startswith("blocked"), r.stdout + r.stderr


@docker
def test_docker_cannot_see_the_host_filesystem_and_cannot_write(tmp_path):
    canary = tmp_path / "canary.txt"
    canary.write_text("TOP-SECRET-CANARY")
    r = DK.run(
        f"import os\nprint(os.path.exists({str(canary)!r}))\ntry:\n    open('/work/x', 'w')\nexcept OSError as e:\n    print('read-only', type(e).__name__)\ntry:\n    open('/etc/x', 'w')\nexcept OSError as e:\n    print('root fs read-only', type(e).__name__)"
    )
    assert r.stdout.splitlines() == ["False", "read-only OSError", "root fs read-only OSError"], (
        r.stdout + r.stderr
    )


@docker
def test_docker_runs_unprivileged_and_reads_input_files():
    r = DK.run(
        "import os\nprint(os.getuid())\nprint(open('data.csv').read().strip())",
        files={"data.csv": "a,b\n1,2\n"},
    )
    assert r.stdout.splitlines()[0] == "65534" and "1,2" in r.stdout, r.stderr


@docker
def test_docker_memory_limit_kills_a_memory_bomb():
    r = DockerSandbox(timeout_s=60, memory="64m").run(
        "x = bytearray(400 * 1024 * 1024)\nprint('allocated')"
    )
    assert (
        not r.ok
        and "allocated" not in r.stdout
        and ("killed" in r.reason or r.returncode in (137, -9))
    ), (r.returncode, r.stderr)


@docker
def test_docker_pids_limit_contains_a_fork_bomb():
    code = "import os, time\nn = 0\nfor _ in range(500):\n    try:\n        if os.fork() == 0:\n            time.sleep(5); os._exit(0)\n        n += 1\n    except OSError:\n        break\nprint('forked', n)"
    r = DockerSandbox(timeout_s=60, pids=16).run(code)
    assert r.stdout.startswith("forked") and int(r.stdout.split()[1]) < 30, (r.stdout, r.stderr)


@docker
def test_docker_timeout_kills_the_container():
    import subprocess

    r = DockerSandbox(timeout_s=3).run("import time; time.sleep(120)")
    assert r.timed_out and "timeout" in r.reason
    time.sleep(1)
    running = subprocess.run(
        ["docker", "ps", "--filter", "name=sandbox-", "-q"], capture_output=True, text=True
    ).stdout.strip()
    assert running == "", "the container must not be left running"


@docker
def test_docker_output_is_capped():
    r = DockerSandbox(timeout_s=60, max_output_bytes=500).run("print('x' * 100000)")
    assert len(r.stdout) <= 600 and r.truncated


def test_docker_command_contains_every_hardening_flag():
    cmd = DockerSandbox(memory="64m", cpus="0.5", pids=10).command("n", Path("/tmp/w"))
    joined = " ".join(cmd)
    for flag in (
        "--network none",
        "--memory 64m",
        "--memory-swap 64m",
        "--cpus 0.5",
        "--pids-limit 10",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        "--user 65534:65534",
        "/tmp/w:/work:ro",
        "--rm",
    ):
        assert flag in joined, flag
    assert sys.executable not in joined


def test_tracebacks_show_only_the_codes_own_frames_not_the_launchers():
    r = SB.run("def f():\n    return 1 / 0\nf()")
    assert 'File "main.py", line 3' in r.stderr and 'File "main.py", line 2, in f' in r.stderr
    assert "runpy" not in r.stderr and "<string>" not in r.stderr and "exec(" not in r.stderr
    syntax = SB.run("x = ")
    assert (
        "SyntaxError" in syntax.stderr
        and "<string>" not in syntax.stderr
        and syntax.returncode == 1
    )
    assert SB.run("raise SystemExit('bye')").stderr == "bye\n"
    assert SB.run("import sys; sys.exit(3)").returncode == 3, (
        "SystemExit passes through with its code"
    )
