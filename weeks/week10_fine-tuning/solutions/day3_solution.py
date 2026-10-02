"""Week 10 Day 3 - Solution: LoRA fine-tuning of SmolLM2-135M-Instruct on the order-extraction data.

1. PARITY    the from-scratch LoRA against PEFT's: same adapters, same logits (and the same after merging)
2. COUNTS    trainable parameters for different ranks and target layers
3. TRAIN     3 epochs of supervised fine-tuning on ~1,100 synthetic emails (CPU, a few minutes per epoch), loss on the answer tokens only
4. CHECK     greedy generation on a few dev emails and on the hand-written ones

  uv run python weeks/week10_fine-tuning/solutions/day3_solution.py [--epochs 3] [--r 16] [--alpha 32] [--targets all|attn] [--tag NAME] [--limit N]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import chatfmt as C  # noqa: E402
import day2_solution as d2  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402
import orders as O  # noqa: E402
import train_sft as T  # noqa: E402

CKPT = ROOT / "outputs/w10_ckpt"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def peft_parity(r: int = 8, alpha: float = 16) -> dict[str, float]:
    """Random non-zero adapters in PEFT's LoRA and in ours; the largest logit difference before and after merging."""
    import blocks as B
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(I.BASE, dtype=torch.float32).eval()
    pm = get_peft_model(
        hf,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=list(L.DEFAULT_TARGETS),
            lora_dropout=0.0,
            bias="none",
        ),
    )
    torch.manual_seed(0)
    for n, p in pm.named_parameters():
        if "lora_" in n:
            torch.nn.init.normal_(p, std=0.05)
    pm.eval()
    model = B.Decoder(B.config_from_hf(pm.base_model.model.config)).eval()
    base = {
        k.replace("base_model.model.", "").replace(".base_layer", ""): v
        for k, v in pm.state_dict().items()
        if "lora_" not in k
    }
    B.load_hf_state_dict(model, base)
    L.add_lora(model, r=r, alpha=alpha)
    adapters = {
        k.replace("base_model.model.model.", "").replace(".default.weight", ""): v
        for k, v in pm.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    L.load_lora_state_dict(model, adapters)
    ids = torch.randint(0, 49152, (2, 30))
    with torch.no_grad():
        before = (pm(ids).logits - model(ids)).abs().max().item()
        merged_peft = pm.merge_and_unload()
        L.merge_lora(model)
        after = (merged_peft(ids).logits - model(ids)).abs().max().item()
    return {"adapters": before, "merged": after}


def counts_table() -> list[tuple[str, int, float]]:
    import blocks as B

    rows = []
    for label, targets in (("attention only", ATTN), ("all linear layers", L.DEFAULT_TARGETS)):
        for r in (4, 8, 16, 32, 64):
            model = B.Decoder(
                B.Config(
                    vocab_size=49152, d_model=576, n_layers=30, n_heads=9, n_kv_heads=3, d_ff=1536
                )
            )
            L.add_lora(model, r=r, targets=targets)
            c = L.count_parameters(model)
            rows.append((f"{label}, r={r}", c["trainable"], c["trainable"] / c["total"]))
    return rows


def load_examples(tok, dirpath: Path, limit: int | None = None):
    train = C.read_jsonl(dirpath / "train.jsonl")[:limit]
    dev = C.read_jsonl(dirpath / "dev.jsonl")
    enc = lambda recs: [C.encode_example(r["messages"], tok) for r in recs]  # noqa: E731
    return enc(train), enc(dev), train, dev


def main(argv: list[str]) -> None:
    arg = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d  # noqa: E731
    epochs, r, alpha, tag, limit = (
        arg("--epochs", 3),
        arg("--r", 16),
        arg("--alpha", 32.0),
        arg("--tag", "sft"),
        arg("--limit", 0) or None,
    )
    targets = ATTN if arg("--targets", "all") == "attn" else L.DEFAULT_TARGETS
    if "--no-parity" not in argv:
        gaps = peft_parity()
        print(
            f"1. PARITY with PEFT (r=8, random adapters): largest logit difference {gaps['adapters']:.1e}; after merging both {gaps['merged']:.1e} (the base model alone differs from the library by about 9e-5: float32 noise)\n"
        )
        print("2. TRAINABLE PARAMETERS (SmolLM2-135M: 30 layers, width 576, 134.5M frozen)")
        for name, n, frac in counts_table():
            print(f"   {name:<28}{n:>12,}  ({frac:.2%} of the model)")
    data_dir = ROOT / "outputs/w10_data"
    if not (data_dir / "train.jsonl").exists():
        d2.main(["--out", str(data_dir)])
    model, tok = I.load_base()
    train, dev, train_recs, _ = load_examples(tok, data_dir, limit)
    names = L.add_lora(model, r=r, alpha=alpha, targets=targets, dropout=0.05)
    cnt = L.count_parameters(model)
    cfg = T.SFTConfig(epochs=epochs)
    print(
        f"\n3. TRAINING: r={r}, alpha={alpha}, {len(names)} adapted layers, {cnt['trainable']:,} trainable of {cnt['total']:,} ({cnt['trainable'] / cnt['total']:.2%}); {len(train)} train / {len(dev)} dev examples, batch {cfg.batch}, lr {cfg.lr}, {epochs} epochs"
    )
    CKPT.mkdir(parents=True, exist_ok=True)

    def save(epoch: int, run: T.SFTRun) -> None:
        torch.save(
            {
                "state": L.lora_state_dict(model),
                "r": r,
                "alpha": alpha,
                "targets": list(targets),
                "epoch": epoch,
            },
            CKPT / f"{tag}_epoch{epoch}.pt",
        )

    run = T.train_sft(
        model,
        train,
        dev,
        cfg,
        tok.pad_token_id,
        log=lambda s: print("   " + s, flush=True),
        on_epoch=save,
    )
    print(
        f"\n   {run.tokens_seen:,} tokens in {run.seconds:.0f}s = {run.tokens_seen / run.seconds:.0f} tokens/s on the CPU; adapters saved in {CKPT}"
    )
    print(
        "   dev loss by epoch: "
        + "  ".join(f"{i}: {d['loss']:.4f}" for i, d in enumerate(run.epoch_dev))
    )

    print("\n4. CHECK: greedy generation after training")
    model.eval()
    sample = [(r_["messages"][1]["content"], r_) for r_ in C.read_jsonl(data_dir / "dev.jsonl")[:3]]
    for email, rec in sample:
        reply, *_ = I.complete(model, tok, I.tuned_messages(email))
        print(
            f"   {email[:70]!r}\n     -> {reply[:130]}\n     gold {rec['messages'][-1]['content'][:130]}"
        )
    res = I.run_task(model, tok, O.HUMAN_EMAILS[:12], I.tuned_messages)
    s = O.summarize([x.score for x in res])
    print(
        f"   first 12 hand-written emails: valid JSON {s['valid_json']:.0%}, valid order {s['valid_order']:.0%}, exact {s['exact']:.0%}, field accuracy {s['field_accuracy']:.0%}"
    )


if __name__ == "__main__":
    main(sys.argv)
