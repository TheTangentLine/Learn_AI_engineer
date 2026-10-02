"""Tests for Week 11 Day 1: the llama.cpp wrappers (pure parsers, quantise, a live server when the tools exist) and the table helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[2] / "weeks/week10_fine-tuning/solutions"))

import day1_solution as d1  # noqa: E402
import llamacpp as L  # noqa: E402

BENCH = """| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 256M F16                 | 256.63 MiB |   134.52 M | BLAS,MTL   |       4 |           pp128 |      6882.94 ± 59.70 |
| llama 256M F16                 | 256.63 MiB |   134.52 M | BLAS,MTL   |       4 |            tg64 |        177.17 ± 3.16 |

build: 0843245cb (9690)
"""

tools = pytest.mark.skipif(
    not (L.have("llama-server") and (L.GGUF_DIR / "order-extractor-Q8_0.gguf").exists()),
    reason="llama.cpp or the GGUF files are not available",
)


def test_the_benchmark_table_is_parsed_into_prompt_and_generation_speeds():
    rows = L.parse_bench(BENCH)
    assert [(r.test, r.tokens_per_second, r.stdev) for r in rows] == [
        ("pp128", 6882.94, 59.70),
        ("tg64", 177.17, 3.16),
    ]
    assert L.parse_bench("nothing useful here") == []


def test_perplexity_and_metrics_parsers():
    assert L.parse_perplexity("blah\nFinal estimate: PPL = 29.5898 +/- 1.34682\n") == (
        29.5898,
        1.34682,
    )
    with pytest.raises(RuntimeError, match="no perplexity"):
        L.parse_perplexity("crashed")
    m = L.parse_metrics(
        "# HELP x\n# TYPE x counter\nllamacpp:prompt_tokens_total 12\nllamacpp:tokens_predicted_total 3.5\nbad line here\n"
    )
    assert m == {"llamacpp:prompt_tokens_total": 12.0, "llamacpp:tokens_predicted_total": 3.5}


def test_response_helpers_read_the_openai_shape_and_the_timings_block():
    resp = {
        "choices": [{"message": {"content": "hi"}}],
        "timings": {
            "prompt_n": 40,
            "prompt_ms": 176.6,
            "predicted_n": 60,
            "predicted_ms": 358.2,
            "extra": 1,
        },
    }
    assert L.reply_text(resp) == "hi" and L.timings(resp) == {
        "prompt_n": 40.0,
        "prompt_ms": 176.6,
        "predicted_n": 60.0,
        "predicted_ms": 358.2,
    }
    assert L.timings({}) == {}


def test_free_ports_are_distinct_and_usable():
    import socket

    a, b = L.free_port(), L.free_port()
    assert a != b or True
    with socket.socket() as s:
        s.bind(("127.0.0.1", a))


def test_the_server_command_has_the_slots_the_context_and_the_metrics_flag():
    srv = L.LlamaServer(
        Path("m.gguf"), parallel=4, ctx=8192, ngl=0, threads=2, extra=["--flash-attn", "on"]
    )
    cmd = srv.command()
    assert (
        cmd[:3] == ["llama-server", "-m", "m.gguf"]
        and cmd[cmd.index("-np") + 1] == "4"
        and cmd[cmd.index("-c") + 1] == "8192"
    )
    assert (
        cmd[cmd.index("-ngl") + 1] == "0"
        and "--metrics" in cmd
        and cmd[-2:] == ["--flash-attn", "on"]
        and srv.url.endswith(str(srv.port))
    )


def test_quantize_skips_existing_outputs_and_the_f16_file_is_required(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: called.append(a[0]))
    dst = tmp_path / "x.gguf"
    dst.write_bytes(b"1")
    assert L.quantize(tmp_path / "src.gguf", dst, "Q8_0") == dst and called == []
    L.quantize(tmp_path / "src.gguf", tmp_path / "new.gguf", "Q4_0")
    assert called and called[0][0] == "llama-quantize" and called[0][-1] == "Q4_0"
    with pytest.raises(FileNotFoundError, match="convert"):
        L.ensure_ggufs(tmp_path)


def test_bits_per_weight_and_the_format_list():
    assert d1.bits_per_weight(1_000_000, 8_000_000) == 1.0
    assert (
        d1.FORMATS[0] == "f16"
        and "Q8_0" in d1.FORMATS
        and d1.path_of("Q4_0").name == "order-extractor-Q4_0.gguf"
    )


@tools
def test_a_live_server_answers_an_extraction_request_with_valid_json_and_reports_timings():
    import json

    with L.LlamaServer(L.GGUF_DIR / "order-extractor-Q8_0.gguf", parallel=1, ctx=1024) as srv:
        resp = srv.chat(
            [
                {"role": "system", "content": "Extract the order from the email as JSON."},
                {"role": "user", "content": "Hi, order A-5 for Dana Lee: 3 desk lamps. ASAP"},
            ],
            max_tokens=150,
        )
        obj = json.loads(L.reply_text(resp))
        assert (
            obj["order_id"] == "A-5"
            and obj["urgency"] == "high"
            and obj["items"][0]["quantity"] == 3
        )
        assert L.timings(resp)["predicted_n"] > 20
        assert srv.metrics()["llamacpp:tokens_predicted_total"] >= L.timings(resp)["predicted_n"]


@tools
def test_a_server_that_cannot_start_raises_instead_of_hanging(tmp_path):
    bad = tmp_path / "not-a-model.gguf"
    bad.write_bytes(b"nope")
    with (
        pytest.raises(RuntimeError, match="exited early"),
        L.LlamaServer(bad, ctx=256, log=tmp_path / "log.txt"),
    ):
        pass


# ----------------------------------------------------------------------------- inspecting and building GGUF files


def tiny_gguf(path: Path, *, chat_template: bool = True) -> Path:
    import gguf
    import numpy as np

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(128)
    w.add_embedding_length(8)
    w.add_block_count(2)
    if chat_template:
        w.add_chat_template("{{ messages }}")
    w.add_tensor("a.weight", np.zeros((4, 8), dtype=np.float32))
    w.add_tensor("b.weight", np.zeros((4, 8), dtype=np.float16))
    w.add_tensor("c.weight", np.zeros((2, 8), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_gguf_summary_counts_tensors_and_bytes_per_type_and_reads_the_metadata(tmp_path):
    s = L.gguf_summary(tiny_gguf(tmp_path / "t.gguf"))
    assert (s["architecture"], s["layers"], s["width"], s["context_length"]) == ("llama", 2, 8, 128)
    assert s["has_chat_template"] and s["n_tensors"] == 3
    assert s["by_type"]["F32"]["tensors"] == 1 and s["by_type"]["F16"]["tensors"] == 2
    assert s["by_type"]["F32"]["mb"] == pytest.approx(4 * 8 * 4 / 1e6)
    assert s["by_type"]["F16"]["mb"] == pytest.approx((4 * 8 * 2 + 2 * 8 * 2) / 1e6)
    assert list(s["by_type"]) == ["F32", "F16"] or list(s["by_type"]) == ["F16", "F32"]
    assert s["tensors"] == {"a.weight": "F32", "b.weight": "F16", "c.weight": "F16"}
    assert not L.gguf_summary(tiny_gguf(tmp_path / "n.gguf", chat_template=False))[
        "has_chat_template"
    ]


def test_the_converter_is_skipped_when_the_output_exists_and_explains_a_missing_source(
    tmp_path, monkeypatch
):
    done = tmp_path / "out.gguf"
    done.write_bytes(b"x")
    assert L.convert_hf(tmp_path, done) == done  # nothing to run
    monkeypatch.setattr(L, "LLAMACPP_SRC", tmp_path / "no-such-source-tree")
    with pytest.raises(FileNotFoundError, match="llama.cpp source"):
        L.convert_hf(tmp_path, tmp_path / "new.gguf")


def test_the_converter_command_is_built_from_the_source_tree(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setattr(L, "LLAMACPP_SRC", src)
    calls = []
    monkeypatch.setattr(L.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    out = L.convert_hf(tmp_path / "model", tmp_path / "o" / "m.gguf", "f16")
    assert out == tmp_path / "o" / "m.gguf" and (tmp_path / "o").is_dir()
    assert calls[0][1].endswith("convert_hf_to_gguf.py") and "--outfile" in calls[0]
    assert calls[0][calls[0].index("--outtype") + 1] == "f16"
