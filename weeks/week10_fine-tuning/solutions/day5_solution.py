"""Week 10 Day 5 - Solution: a DPO pass on top of the SFT model, with a measured change.

1. PAIRS     (a) ON-POLICY: sample answers from the SFT model, keep the emails where it gets one wrong: chosen = the gold JSON, rejected = its own mistake
             (b) INJECTED: for other emails, rejected = the gold JSON with one realistic error (wrong id, dropped item, ...), the Day 2 defects
2. DPO       a fresh LoRA on the merged SFT model, the SFT model as the frozen reference (log-probabilities precomputed once)
3. MEASURE   SFT against DPO on the hand-written and held-out synthetic emails (paired), what changed in the answers, how far the model drifted, forgetting

  uv run python weeks/week10_fine-tuning/solutions/day5_solution.py [--sft PATH] [--n-onpolicy 250] [--n-injected 500] [--beta 0.1]
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "weeks/week09_transformers-from-scratch/solutions/weekly/fastgen"))

import chatfmt as C  # noqa: E402
import dpo as D  # noqa: E402
import evalrun as E  # noqa: E402
import gen_data as G  # noqa: E402
import infer as I  # noqa: E402
import lora as L  # noqa: E402
import orders as O  # noqa: E402
from generate import generate  # noqa: E402

CKPT, DATA = ROOT / "outputs/w10_ckpt", ROOT / "outputs/w10_data"


def sample_answers(
    model,
    tok,
    email: str,
    k: int,
    seed: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 160,
) -> list[str]:
    """k sampled answers of the SFT model to one email (a different seed for each)."""
    prompt_ids = tok(
        C.render(I.tuned_messages(email), add_generation_prompt=True), add_special_tokens=False
    )["input_ids"]
    stop = {tok.convert_tokens_to_ids(C.IM_END)}
    out = []
    for j in range(k):
        g = generate(
            model,
            prompt_ids,
            max_new_tokens,
            mode="cache",
            temperature=temperature,
            top_p=top_p,
            stop_ids=stop,
            seed=seed * 1000 + j,
        )
        out.append(tok.decode([t for t in g.tokens if t not in stop]))
    return out


def onpolicy_pairs(
    model, tok, items: list[tuple[str, dict]], k: int = 2, seed: int = 0, log=None
) -> tuple[list[dict], dict]:
    """(preference records, statistics). For each email, the worst sampled answer (by number of correct fields, invalid ones worst) becomes the
    rejected one if it is not an exact match; the chosen one is always the gold JSON."""
    recs, wrong, total = [], 0, 0
    for n, (email, gold) in enumerate(items):
        answers = sample_answers(model, tok, email, k, seed + n)
        scores = [O.score(a, gold) for a in answers]
        total += len(answers)
        wrong += sum(not s.exact for s in scores)
        worst = min(range(len(answers)), key=lambda i: (scores[i].n_correct, scores[i].valid_order))
        if not scores[worst].exact and answers[worst] != O.order_json(gold):
            recs.append(
                C.to_preference(I.tuned_messages(email), O.order_json(gold), answers[worst])
                | {"source": "on-policy"}
            )
        if log and (n + 1) % 25 == 0:
            log(
                f"   sampled {n + 1}/{len(items)} emails: {wrong}/{total} samples wrong so far, {len(recs)} pairs"
            )
    return recs, {
        "emails": len(items),
        "samples": total,
        "wrong_samples": wrong,
        "pairs": len(recs),
    }


def injected_pairs(items: list[tuple[str, dict]], seed: int = 0) -> list[dict]:
    """rejected = the correct JSON of the same email with one injected label error: a plausible mistake, but not one this model made."""
    rng = random.Random(seed)
    recs = []
    for email, gold in items:
        if not gold["is_order"]:
            continue
        bad, kind = G.inject_defect(gold, rng)
        if O.order_json(bad) != O.order_json(gold):
            recs.append(
                C.to_preference(I.tuned_messages(email), O.order_json(gold), O.order_json(bad))
                | {"source": f"injected:{kind}"}
            )
    return recs


def encode_records(recs: list[dict], tok) -> list[D.Pair]:
    return [D.encode_pair(r["prompt"], r["chosen"], r["rejected"], tok) for r in recs]


def drift(
    model, ref_logps: list[tuple[float, float]], pairs: list[D.Pair], pad_id: int
) -> dict[str, float]:
    """How far the policy moved from the reference on the training pairs: the mean change in log-probability of the chosen and the rejected answers."""
    lp = D.reference_logprobs(model, pairs, pad_id)
    dc = sum(a[0] - b[0] for a, b in zip(lp, ref_logps, strict=True)) / len(lp)
    dr = sum(a[1] - b[1] for a, b in zip(lp, ref_logps, strict=True)) / len(lp)
    return {"chosen": dc, "rejected": dr, "margin": dc - dr}


def main(argv: list[str]) -> None:
    arg = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d  # noqa: E731
    sft_path, n_on, n_inj, beta = (
        Path(arg("--sft", str(CKPT / "sft_epoch3.pt"))),
        arg("--n-onpolicy", 250),
        arg("--n-injected", 500),
        arg("--beta", 0.1),
    )
    lr, nll, tag = arg("--lr", 5e-5), arg("--nll", 0.0), arg("--tag", "")
    train_pairs = E.records_to_pairs(C.read_jsonl(DATA / "train.jsonl"))
    rng = random.Random(0)
    rng.shuffle(train_pairs)
    synth = E.records_to_pairs(C.read_jsonl(DATA / "test.jsonl"))[:100]
    model, tok = E.load_tuned(
        sft_path
    )  # the SFT model: base + adapters, merged: this is the reference
    print(f"SFT model: {sft_path.name}")

    t0 = time.time()
    print("\n1. PREFERENCE PAIRS")
    if "--reuse-pairs" in argv and (DATA / "pairs.jsonl").exists():
        recs = C.read_jsonl(DATA / "pairs.jsonl")
        print(
            f"   reusing {len(recs)} saved pairs ({sum(r['source'] == 'on-policy' for r in recs)} on-policy)"
        )
    else:
        on, st = onpolicy_pairs(model, tok, train_pairs[:n_on], log=lambda s: print(s, flush=True))
        inj = injected_pairs(train_pairs[n_on : n_on + n_inj])
        print(
            f"   on-policy: sampled 2 answers for each of {st['emails']} emails; {st['wrong_samples']} of {st['samples']} samples ({st['wrong_samples'] / st['samples']:.1%}) were not exact matches -> {st['pairs']} pairs"
        )
        kinds = sorted(
            {s: sum(1 for r in inj if r["source"] == s) for s in {r["source"] for r in inj}}.items()
        )
        print(f"   injected: {len(inj)} pairs; kinds: " + ", ".join(f"{k} {v}" for k, v in kinds))
        recs = on + inj
        C.write_jsonl(DATA / "pairs.jsonl", recs)
    print(f"   {len(recs)} pairs in all ({time.time() - t0:.0f}s)")
    pairs = encode_records(recs, tok)

    print("\n2. DPO (a fresh LoRA on the merged SFT model; reference = the SFT model)")
    sft_eval = {
        "human": I.run_task(model, tok, O.HUMAN_EMAILS, I.tuned_messages),
        "synthetic": I.run_task(model, tok, synth, I.tuned_messages),
    }
    sft_probes, sft_loss = E.run_probes(model, tok), None
    ref = D.reference_logprobs(model, pairs, tok.pad_token_id)
    L.add_lora(model, r=16, alpha=32, dropout=0.0)
    cfg = D.DPOConfig(epochs=1, batch=4, lr=lr, beta=beta, nll_weight=nll)
    print(
        f"   {len(pairs)} pairs, batch {cfg.batch}, lr {cfg.lr}, beta {cfg.beta}, NLL weight {cfg.nll_weight}, {L.count_parameters(model)['trainable']:,} trainable parameters"
    )
    run = D.train_dpo(
        model, pairs, ref, cfg, tok.pad_token_id, log=lambda s: print("   " + s, flush=True)
    )
    d = drift(model.eval(), ref, pairs, tok.pad_token_id)
    print(
        f"   {run.seconds:.0f}s; mean change in log-probability on the training pairs: chosen {d['chosen']:+.2f}, rejected {d['rejected']:+.2f} (margin {d['margin']:+.2f})"
    )
    torch.save(
        {
            "state": L.lora_state_dict(model),
            "r": 16,
            "alpha": 32,
            "targets": list(L.DEFAULT_TARGETS),
            "beta": beta,
        },
        CKPT / f"dpo{tag or ''}_beta{beta}.pt",
    )
    L.merge_lora(model)

    print("\n3. SFT AGAINST SFT + DPO (same emails; greedy)")
    dpo_eval = {
        "human": I.run_task(model, tok, O.HUMAN_EMAILS, I.tuned_messages),
        "synthetic": I.run_task(model, tok, synth, I.tuned_messages),
    }
    print(f"   {'system':<10}{'set':<11}{'valid order':>12}{'exact':>20}{'field acc':>11}")
    for name, ev in (("SFT", sft_eval), ("SFT+DPO", dpo_eval)):
        for setname in ("human", "synthetic"):
            r = E.summarize_results(ev[setname])
            print(
                f"   {name:<10}{setname:<11}{r['valid_order']:>12.0%}{E.fmt_ci(r['exact_ci']):>20}{r['field_accuracy']:>11.0%}"
            )
    for setname in ("human", "synthetic"):
        for metric in ("exact", "fields"):
            c = E.compare(dpo_eval[setname], sft_eval[setname], metric)
            print(
                f"   DPO - SFT on {setname:<9} {metric:<7} {c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  p = {c['p']:.3f}  (better on {c['wins']}, worse on {c['losses']}, tied on {c['ties']})"
            )
    print(
        "   error types, hand-written: SFT "
        + str(E.error_types(sft_eval["human"]))
        + "  |  DPO "
        + str(E.error_types(dpo_eval["human"]))
    )

    p = E.run_probes(model, tok)
    print(
        f"\n4. FORGETTING: probe accuracy SFT {sft_probes['accuracy']:.0%} -> SFT+DPO {p['accuracy']:.0%}; order JSON for unrelated prompts {sft_probes['order_json_rate']:.0%} -> {p['order_json_rate']:.0%}"
    )
    _ = sft_loss


if __name__ == "__main__":
    main(sys.argv)
