"""A thin, tested wrapper around the llama.cpp tools installed with ``brew install llama.cpp``: quantise, benchmark, and serve a GGUF model.

    gguf_dir = ensure_ggufs()                                  # (re)build f16/Q8_0/Q4_0/Q4_K_M/Q2_K from the Week 10 fine-tune
    rows = bench(gguf_dir / "order-extractor-Q8_0.gguf")      # llama-bench: prompt and generation tokens/second
    with LlamaServer(model, parallel=4) as srv:               # llama-server: an OpenAI-compatible endpoint with continuous batching
        reply = srv.chat([{"role": "user", "content": "..."}])

Everything here shells out to real binaries; the parsing helpers are pure functions with tests. Nothing in this module needs a GPU: llama.cpp uses
Metal on Apple silicon and the CPU elsewhere (``ngl=0`` forces CPU).
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "w11"
GGUF_DIR = OUT / "gguf"
QUANTS = ("Q8_0", "Q4_0", "Q4_K_M", "Q2_K")
STEM = "order-extractor"


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------- building the files


def quantize(src: Path, dst: Path, kind: str) -> Path:
    """``llama-quantize src dst kind``; skips the work if ``dst`` already exists."""
    if dst.exists():
        return dst
    subprocess.run(["llama-quantize", str(src), str(dst), kind], check=True, capture_output=True)
    return dst


LLAMACPP_SRC = (
    OUT / "llama.cpp-b9690"
)  # the source tree of the same release as the installed binaries (the converter script is not in the brew bottle)
CHAT_STEM = "qwen2.5-0.5b-instruct"


def convert_hf(model_dir: Path, dst: Path, outtype: str = "f16") -> Path:
    """``convert_hf_to_gguf.py model_dir --outfile dst``; skips the work if ``dst`` exists. Needs the llama.cpp source tree (``LLAMACPP_SRC``) and ``gguf``/``torch``/``transformers``."""
    if dst.exists():
        return dst
    script = LLAMACPP_SRC / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(
            f"{script} is missing: download the llama.cpp source for your installed release (see the Day 1 lesson)"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(script), str(model_dir), "--outfile", str(dst), "--outtype", outtype],
        check=True,
        capture_output=True,
    )
    return dst


def ensure_chat_gguf(gguf_dir: Path = GGUF_DIR) -> Path:
    """Qwen2.5-0.5B-Instruct (a general chat model, from the local Hugging Face cache) as a Q8_0 GGUF: the model that answers /v1/ask."""
    from huggingface_hub import snapshot_download

    q8 = gguf_dir / f"{CHAT_STEM}-Q8_0.gguf"
    if q8.exists():
        return q8
    src = Path(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct"))
    f16 = convert_hf(src, gguf_dir / f"{CHAT_STEM}-f16.gguf")
    return quantize(f16, q8, "Q8_0")


def ensure_ggufs(gguf_dir: Path = GGUF_DIR) -> Path:
    """The f16 file must exist (built once by ``convert_hf_to_gguf.py`` from the Week 10 model directory: see the Day 1 lesson); the quantised ones are derived."""
    f16 = gguf_dir / f"{STEM}-f16.gguf"
    if not f16.exists():
        raise FileNotFoundError(
            f"{f16} is missing: convert outputs/w10_model with llama.cpp's convert_hf_to_gguf.py first (Day 1)"
        )
    for q in QUANTS:
        quantize(f16, gguf_dir / f"{STEM}-{q}.gguf", q)
    return gguf_dir


# ----------------------------------------------------------------------------- llama-bench


@dataclass
class BenchRow:
    model: str
    test: str  # pp128 (prompt processing) or tg64 (token generation)
    tokens_per_second: float
    stdev: float


_ROW = re.compile(
    r"\|\s*(?P<model>[^|]+?)\s*\|\s*(?P<size>[\d.]+ \w+)\s*\|\s*(?P<params>[\d.]+ \w)\s*\|\s*(?P<backend>[^|]+?)\s*\|\s*(?P<threads>\d+)\s*\|\s*(?P<test>(?:pp|tg)\d+)\s*\|\s*(?P<tps>[\d.]+)\s*±\s*(?P<sd>[\d.]+)\s*\|"
)


def parse_bench(text: str) -> list[BenchRow]:
    """The table ``llama-bench`` prints: one row per test with 'mean ± stdev' tokens per second."""
    return [
        BenchRow(m["model"], m["test"], float(m["tps"]), float(m["sd"]))
        for m in _ROW.finditer(text)
    ]


def bench(
    model: Path,
    *,
    prompt: int = 128,
    gen: int = 64,
    threads: int = 4,
    ngl: int = 99,
    repeats: int = 3,
) -> dict[str, float]:
    """{'pp': prompt tokens/s, 'tg': generated tokens/s}. ``ngl=0`` keeps every layer on the CPU; 99 offloads all of them (Metal on a Mac)."""
    out = subprocess.run(
        [
            "llama-bench",
            "-m",
            str(model),
            "-p",
            str(prompt),
            "-n",
            str(gen),
            "-t",
            str(threads),
            "-ngl",
            str(ngl),
            "-r",
            str(repeats),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = parse_bench(out)
    got = {r.test[:2]: r.tokens_per_second for r in rows}
    if {"pp", "tg"} - set(got):
        raise RuntimeError(f"could not read the benchmark table:\n{out[-600:]}")
    return got


# ----------------------------------------------------------------------------- llama-perplexity


_PPL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def parse_perplexity(text: str) -> tuple[float, float]:
    """(perplexity, standard error) from the last line ``llama-perplexity`` prints."""
    m = _PPL.search(text)
    if not m:
        raise RuntimeError(f"no perplexity in the output:\n{text[-500:]}")
    return float(m.group(1)), float(m.group(2))


def perplexity(
    model: Path, text_file: Path, *, ctx: int = 512, ngl: int = 99, threads: int = 4
) -> tuple[float, float]:
    out = subprocess.run(
        [
            "llama-perplexity",
            "-m",
            str(model),
            "-f",
            str(text_file),
            "-c",
            str(ctx),
            "-ngl",
            str(ngl),
            "-t",
            str(threads),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_perplexity(out.stdout + out.stderr)


# ----------------------------------------------------------------------------- llama-server


class LlamaServer:
    """Runs ``llama-server`` on a free port and exposes its OpenAI-compatible API.

    ``parallel`` is the number of slots (sequences decoded together: continuous batching); the context ``ctx`` is divided among them, so each slot
    gets ``ctx // parallel`` tokens. ``extra`` is appended to the command line.
    """

    def __init__(
        self,
        model: Path,
        *,
        parallel: int = 1,
        ctx: int = 4096,
        ngl: int = 99,
        threads: int = 4,
        extra: list[str] | None = None,
        log: Path | None = None,
    ):
        self.model, self.parallel, self.ctx, self.ngl, self.threads = (
            Path(model),
            parallel,
            ctx,
            ngl,
            threads,
        )
        self.extra = extra or []
        self.port = free_port()
        self.log = log or (OUT / f"server_{self.port}.log")
        self._proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [
            "llama-server",
            "-m",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.ctx),
            "-np",
            str(self.parallel),
            "-ngl",
            str(self.ngl),
            "-t",
            str(self.threads),
            "--metrics",
            *self.extra,
        ]

    def __enter__(self) -> LlamaServer:
        OUT.mkdir(parents=True, exist_ok=True)
        self._logf = open(self.log, "w")  # noqa: SIM115 - closed in __exit__
        self._proc = subprocess.Popen(self.command(), stdout=self._logf, stderr=subprocess.STDOUT)
        self.wait_ready()
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._logf.close()

    def wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log}")
            try:
                if httpx.get(f"{self.url}/health", timeout=1).json().get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.2)
        raise TimeoutError(f"llama-server did not become ready; see {self.log}")

    def chat(
        self, messages: list[dict], *, max_tokens: int = 160, temperature: float = 0.0, **kw
    ) -> dict:
        r = httpx.post(
            f"{self.url}/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **kw},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def metrics(self) -> dict[str, float]:
        return parse_metrics(httpx.get(f"{self.url}/metrics", timeout=5).text)


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text format -> {name: value} (labels are not used by llama-server's metrics)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name.strip()] = float(value)
            except ValueError:
                continue
    return out


def reply_text(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


def timings(resp: dict) -> dict[str, float]:
    t = resp.get("timings", {})
    return {
        k: float(t[k]) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms") if k in t
    }


def file_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def python_exe() -> str:
    return sys.executable


_ = json


# ----------------------------------------------------------------------------- inspecting a GGUF file


def gguf_summary(path: Path) -> dict:
    """What is inside a GGUF file: the architecture metadata and, per tensor type, how many tensors and how many bytes. A 'Q4_K_M' file is NOT all Q4_K: the
    quantiser picks a type per tensor, and falls back to a different one when a tensor's width is not a multiple of the block size (256 for the k-quants)."""
    from gguf import GGUFReader

    r = GGUFReader(str(path))

    def scalar(key: str):
        f = r.fields.get(key)
        return None if f is None else f.contents() if hasattr(f, "contents") else None

    by: dict[str, list[int]] = {}
    for t in r.tensors:
        row = by.setdefault(t.tensor_type.name, [0, 0])
        row[0] += 1
        row[1] += int(t.n_bytes)
    return {
        "architecture": scalar("general.architecture"),
        "context_length": scalar("llama.context_length"),
        "layers": scalar("llama.block_count"),
        "width": scalar("llama.embedding_length"),
        "has_chat_template": "tokenizer.chat_template" in r.fields,
        "n_tensors": len(r.tensors),
        "by_type": {
            k: {"tensors": v[0], "mb": v[1] / 1e6}
            for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])
        },
        "tensors": {t.name: t.tensor_type.name for t in r.tensors},
    }
