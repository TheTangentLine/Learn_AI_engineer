"""Tests for Week 10 Day 6: exporting weights under Hugging Face names and loading them back, the Modelfile, the hosted-API file checks, sizes, the audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "weeks/week09_transformers-from-scratch/solutions"))

import blocks as B  # noqa: E402
import export as X  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402


def tiny(tie=True, vocab=60, seed=0):
    torch.manual_seed(seed)
    m = B.Decoder(
        B.Config(
            vocab_size=vocab,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=64,
            tie_embeddings=tie,
            rope_theta=100000.0,
        )
    ).eval()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    return m


def hf_cfg(m: B.Decoder, tie=True):
    from transformers import LlamaConfig

    c = m.cfg
    return LlamaConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.d_model,
        intermediate_size=c.d_ff,
        num_hidden_layers=c.n_layers,
        num_attention_heads=c.n_heads,
        num_key_value_heads=c.n_kv_heads,
        max_position_embeddings=c.max_seq_len,
        rms_norm_eps=c.rms_eps,
        rope_theta=c.rope_theta,
        tie_word_embeddings=tie,
    )


# ----------------------------------------------------------------------------- weights under Hugging Face names


def test_the_exported_names_are_hugging_faces_and_a_tied_output_matrix_is_not_written():
    sd = X.to_hf_state_dict(tiny())
    assert (
        "model.embed_tokens.weight" in sd
        and "model.layers.1.mlp.down_proj.weight" in sd
        and "model.norm.weight" in sd
    )
    assert "lm_head.weight" not in sd and not any(k.startswith("layers.") for k in sd)
    assert "lm_head.weight" in X.to_hf_state_dict(tiny(tie=False))


@pytest.mark.parametrize("tie", [True, False])
def test_export_then_load_reproduces_the_weights_exactly(tie):
    src, dst = tiny(tie=tie), tiny(tie=tie, seed=9)
    B.load_hf_state_dict(dst, X.to_hf_state_dict(src))
    ids = torch.randint(0, 60, (2, 11))
    with torch.no_grad():
        assert torch.equal(src(ids), dst(ids))


def test_a_merged_lora_model_exports_and_hugging_face_loads_it_with_the_same_logits(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(I.BASE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(str(exc))
    m = tiny(vocab=len(tok))
    L.add_lora(m, r=4, alpha=8)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.1)
    L.merge_lora(m)
    files = X.save_hf_model(
        m,
        tok,
        tmp_path / "out",
        hf_cfg(m),
        X.model_card(
            "HuggingFaceTB/SmolLM2-135M-Instruct", {"exact match (hand-written)": "0%"}, "synthetic"
        ),
    )
    assert {"config.json", "model.safetensors", "tokenizer.json", "README.md"} <= set(files)
    hf = AutoModelForCausalLM.from_pretrained(tmp_path / "out", dtype=torch.float32).eval()
    ids = torch.randint(0, 1000, (2, 12))
    with torch.no_grad():
        assert (hf(ids).logits - m(ids)).abs().max() < 1e-4
    assert X.audit_directory(tmp_path / "out") == []


# ----------------------------------------------------------------------------- the card, the Modelfile, the hosted-API checks


def test_the_model_card_has_front_matter_results_and_limitations():
    card = X.model_card("org/base", {"exact match": "55%"}, "data note")
    assert (
        card.startswith("---\nbase_model: org/base")
        and "| exact match | 55% |" in card
        and "## Limitations" in card
        and "data note" in card
    )


def test_the_modelfile_names_the_weights_the_chatml_template_and_the_stop_tokens():
    mf = X.ollama_modelfile("./x.gguf", "Be brief.", 0.0)
    assert (
        mf.startswith("FROM ./x.gguf\n")
        and "<|im_start|>assistant" in mf
        and 'SYSTEM """Be brief."""' in mf
    )
    assert mf.count("PARAMETER stop") == 2 and "PARAMETER temperature 0.0" in mf


def test_the_hosted_api_file_checks_catch_the_usual_mistakes():
    good = X.to_openai_jsonl(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        * 10
    )
    assert X.validate_openai_jsonl(good) == []
    bad = [
        "not json",
        json.dumps({"messages": []}),
        json.dumps({"messages": [{"role": "robot", "content": "x"}]}),
        json.dumps({"messages": [{"role": "user", "content": "q"}]}),
    ]
    problems = X.validate_openai_jsonl(bad + good, min_examples=10)
    assert any("not valid JSON" in p for p in problems) and any(
        "missing messages" in p for p in problems
    )
    assert any("unknown role" in p for p in problems) and any(
        "no assistant message" in p for p in problems
    )
    assert any("at least 10" in p for p in X.validate_openai_jsonl(good[:3]))
    assert any(
        "context limit" in p
        for p in X.validate_openai_jsonl(
            X.to_openai_jsonl([{"messages": [{"role": "assistant", "content": "x" * 400000}]}] * 10)
        )
    )


def test_training_cost_is_tokens_times_epochs_times_the_price_you_supply():
    assert X.estimate_training_cost(158_745, 3, 8.0) == pytest.approx(3.809, abs=1e-3)
    assert X.estimate_training_cost(0, 3, 8.0) == 0


# ----------------------------------------------------------------------------- sizes and the audit


def test_size_table_counts_blocks_at_the_formats_bits_and_keeps_the_embedding_at_fp16():
    m = tiny()
    rows = {r["format"]: r["bytes"] for r in X.size_table(m)}
    n_embed = m.embed_tokens.weight.numel()
    n_norm = sum(p.numel() for n, p in m.named_parameters() if "norm" in n)
    n_blocks = sum(
        p.numel() for n, p in m.named_parameters() if "embed_tokens" not in n and "norm" not in n
    )
    assert rows["fp32"] == 4 * m.num_parameters() and rows["fp16"] == 2 * m.num_parameters()
    assert (
        rows["q8_0"] == pytest.approx(n_blocks * 8.5 / 8 + 2 * (n_embed + n_norm))
        and rows["q4_0"] < rows["q8_0"] < rows["fp16"] < rows["fp32"]
    )


def test_the_audit_lists_missing_files_a_bare_card_and_credentials(tmp_path):
    assert any("missing config.json" in p for p in X.audit_directory(tmp_path))
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "README.md").write_text("no front matter")
    problems = X.audit_directory(tmp_path)
    assert any("front matter" in p for p in problems) and any("no results" in p for p in problems)
    (tmp_path / "README.md").write_text(
        X.model_card("org/base", {"m": "1"}, "d") + "\nhf_" + "a" * 30
    )
    assert any("credential" in p for p in X.audit_directory(tmp_path))
    (tmp_path / "README.md").write_text(X.model_card("org/base", {"m": "1"}, "d"))
    assert X.audit_directory(tmp_path) == []
