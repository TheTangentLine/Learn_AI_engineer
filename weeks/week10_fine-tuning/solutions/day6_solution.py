"""Week 10 Day 6 - Solution: merge, export, and what quantisation does to the fine-tuned model.

1. MERGE     LoRA folded into the weights; the merged model's outputs equal the adapter model's
2. EXPORT    a Hugging Face directory (config, safetensors, tokenizer, model card): loaded back by the library and compared logit for logit and token for token
3. AUDIT     the checks to run before publishing; the Modelfile an Ollama user would need (not run); the hosted-API training file and its cost (not run)
4. QUANTISE  the merged model with its weights rounded to 8 and 4 bits: size, extraction accuracy and speed

  uv run python weeks/week10_fine-tuning/solutions/day6_solution.py [--ckpt PATH] [--out DIR] [--n-synth 50]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import chatfmt as C  # noqa: E402
import evalrun as E  # noqa: E402
import export as X  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402
import orders as O  # noqa: E402
import quant as Q  # noqa: E402

CKPT, DATA = ROOT / "outputs/w10_ckpt", ROOT / "outputs/w10_data"


def main(argv: list[str]) -> None:
    arg = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d  # noqa: E731
    ckpt, out, n_synth = (
        Path(arg("--ckpt", str(CKPT / "sft_epoch3.pt"))),
        Path(arg("--out", str(ROOT / "outputs/w10_model"))),
        arg("--n-synth", 50),
    )
    synth = E.records_to_pairs(C.read_jsonl(DATA / "test.jsonl"))[:n_synth]
    items = {"human": O.HUMAN_EMAILS, "synthetic": synth}

    base, tok = I.load_base()
    L.add_lora(
        base,
        r=int(torch.load(ckpt, weights_only=False)["r"]),
        alpha=torch.load(ckpt, weights_only=False)["alpha"],
        targets=tuple(torch.load(ckpt, weights_only=False)["targets"]),
    )
    L.load_lora_state_dict(base, torch.load(ckpt, weights_only=False)["state"])
    ids = torch.tensor(
        [
            tok("Order A-1 please: 3 desk lamps, thanks. Sam Lee", add_special_tokens=False)[
                "input_ids"
            ]
        ]
    )
    base.eval()
    with torch.no_grad():
        adapter_logits = base(ids)
    n_adapter = L.count_parameters(base)["trainable"]
    L.merge_lora(base)
    model = base.eval()
    with torch.no_grad():
        gap = (model(ids) - adapter_logits).abs().max().item()
    print(
        f"1. MERGE: {n_adapter:,} adapter parameters folded into the weights; largest logit difference between the adapter model and the merged model {gap:.1e}\n"
    )

    card = X.model_card(
        I.BASE,
        {
            "valid orders, hand-written emails": "see evaluation",
            "model size": f"{model.num_parameters() / 1e6:.0f}M parameters",
        },
        "About 1,100 synthetic emails, label-checked and deduplicated (Week 10 Day 2). The 38 hand-written evaluation emails were never used for training.",
    )
    files = X.save_hf_model(model, tok, out, I.BASE, card)
    print(f"2. EXPORT to {out.relative_to(ROOT)}: {files}")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        hf_gap = (hf(ids).logits - model(ids)).abs().max().item()
    prompt = C.render(I.tuned_messages(O.HUMAN_EMAILS[3][0]), add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    hf_tokens = hf.generate(
        **enc,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.convert_tokens_to_ids(C.IM_END),
    )[0, enc["input_ids"].shape[1] :].tolist()
    mine, *_ = I.complete(model, tok, I.tuned_messages(O.HUMAN_EMAILS[3][0]))
    theirs = tok.decode([t for t in hf_tokens if t != tok.convert_tokens_to_ids(C.IM_END)])
    print(
        f"   loaded back by transformers: logits differ by {hf_gap:.1e}; the same email gives identical output in both: {mine == theirs}"
    )
    print(f"   -> {mine[:140]}")
    problems = X.audit_directory(out)
    print(f"   pre-publication audit: {problems or 'no problems found'}")
    print(
        "   publishing (NOT RUN, needs a token):  huggingface_hub.HfApi().upload_folder(folder_path=..., repo_id='<you>/order-extractor', repo_type='model')\n"
    )

    print(
        "3. OLLAMA (NOT RUN: llama.cpp's GGUF converter and Ollama are not installed here). The Modelfile:"
    )
    print("   " + X.ollama_modelfile().replace("\n", "\n   "))
    recs = C.read_jsonl(DATA / "train.jsonl")
    lines = X.to_openai_jsonl(recs)
    tokens = sum(len(C.encode_example(r["messages"], tok)) for r in recs)
    print(
        f"4. HOSTED FINE-TUNING (NOT RUN): {len(lines)} training lines, format check: {X.validate_openai_jsonl(lines) or 'ok'}; {tokens:,} tokens x 3 epochs = {tokens * 3:,} billed tokens ({X.estimate_training_cost(tokens, 3, 8.0):.2f} at an ASSUMED $8 per million)\n"
    )

    print(
        "5. QUANTISATION of the merged model (every Linear weight in the blocks rounded; the embedding table kept at fp16)"
    )
    sizes = {r["format"]: r["bytes"] for r in X.size_table(model)}
    results = {}
    for fmt in ("fp32", "q8_0", "q4_0", "nf4", "int4"):
        m2, _ = I.load_base()
        L.add_lora(
            m2,
            r=int(torch.load(ckpt, weights_only=False)["r"]),
            alpha=torch.load(ckpt, weights_only=False)["alpha"],
            targets=tuple(torch.load(ckpt, weights_only=False)["targets"]),
        )
        L.load_lora_state_dict(m2, torch.load(ckpt, weights_only=False)["state"])
        L.merge_lora(m2)
        m2.eval()
        if fmt != "fp32":
            Q.quantize_model_(m2, fmt)
        t0 = time.time()
        res = E.evaluate_sets(m2, tok, items)
        results[fmt] = (res, time.time() - t0)
        print(f"   evaluated {fmt}", file=sys.stderr, flush=True)
    print(
        f"   {'format':<8}{'size (MB)':>10}{'bits/weight':>12}{'hand-written exact':>22}{'field acc':>10}{'synthetic exact':>17}{'field acc':>10}"
    )
    for fmt, (res, _) in results.items():
        h, y = E.summarize_results(res["human"]), E.summarize_results(res["synthetic"])
        name = "fp16" if fmt == "fp32" else fmt
        print(
            f"   {fmt:<8}{sizes.get(fmt if fmt != 'int4' else 'nf4', sizes['fp32']) / 1e6:>10.0f}{Q.BITS_PER_WEIGHT[fmt]:>12.1f}{E.fmt_ci(h['exact_ci']):>22}{h['field_accuracy']:>10.0%}{E.fmt_ci(y['exact_ci']):>17}{y['field_accuracy']:>10.0%}"
        )
        _ = name
    print(
        "   (the quantised models here are fp32 tensors holding rounded values: size is the format's storage size, speed is NOT improved by this simulation)"
    )


if __name__ == "__main__":
    main(sys.argv)
