"""Week 2 Day 6 - Solution: optimise a prompt against a metric instead of by hunch.

Part 1  A tiny prompt optimiser written from scratch (instruction candidates x few-shot demo sets,
        scored on a DEV split, winner confirmed on a held-out TEST split).
Part 2  The same job with DSPy (Signature -> Module -> BootstrapFewShot -> Evaluate).

  uv run python .../day6_solution.py --offline          # both parts against scripted fakes
  uv run python .../day6_solution.py                    # Part 1 on your provider
  uv run python .../day6_solution.py --dspy             # Part 2 on your provider (needs `pip install dspy`)

The offline "model" is a keyword matcher that LEARNS from the demos inside the prompt, so which demos
you pick genuinely changes its accuracy. It checks the optimiser's logic - not any real model.
"""

# NOTE: no `from __future__ import annotations` here on purpose: it turns the type hints of a DSPy
# Signature into strings, and DSPy then fails with "Field types must be types".
import itertools
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common import llm  # noqa: E402
from common.fake import fake_llm  # noqa: E402

LABELS = ["billing", "technical", "account", "other"]

DATA: list[tuple[str, str]] = [
    ("I was charged twice this month", "billing"),
    ("Please refund my last order", "billing"),
    ("Where can I find my invoice for March?", "billing"),
    ("The receipt shows the wrong amount", "billing"),
    ("My card was billed after I cancelled", "billing"),
    ("The payment failed but money left my bank", "billing"),
    ("Why is there a surcharge on my statement?", "billing"),
    ("Statement shows a surcharge I never agreed to", "billing"),
    ("The app crashes when I tap export", "technical"),
    ("I get error 500 on the dashboard", "technical"),
    ("The page is very slow since the update", "technical"),
    ("A bug makes the chart disappear", "technical"),
    ("The screen freezes on startup", "technical"),
    ("Startup freezes on my phone every time", "technical"),
    ("Notifications stopped arriving on Android", "technical"),
    ("No notifications arrive on my Android tablet", "technical"),
    ("I forgot my password", "account"),
    ("Cannot login, it says account locked", "account"),
    ("How do I change my email address?", "account"),
    ("My profile picture will not update", "account"),
    ("I want to enable two-factor authentication", "account"),
    ("Two-factor authentication codes never arrive", "account"),
    ("Please delete my account and data", "account"),
    ("How can I delete my account permanently?", "account"),
    ("Do you have an office in Berlin?", "other"),
    ("I run a podcast and want to discuss a partnership", "other"),
    ("Journalist here, who handles press requests?", "other"),
    ("Just wanted to say I love the new design", "other"),
    ("Are you hiring engineers right now?", "other"),
    ("What are the hiring requirements for designers?", "other"),
    ("Is there a student discount program?", "other"),
    ("Do you offer a discount for nonprofits?", "other"),
]


def split(data, seed=7, n_train=12, n_dev=10):
    rows = list(data)
    random.Random(seed).shuffle(rows)
    return rows[:n_train], rows[n_train : n_train + n_dev], rows[n_train + n_dev :]


# ----------------------------------------------------------------- the prompt "program"

INSTRUCTIONS = {
    "terse": "Classify the support ticket as billing, technical, account or other.",
    "defs": (
        "Classify the support ticket into one category.\n<definitions>\n"
        "billing: charges, refunds, invoices, payments\ntechnical: crashes, errors, bugs, slowness\n"
        "account: login, password, profile, email, deleting an account\n"
        "other: everything else (company questions, press, feedback)\n</definitions>"
    ),
    "defs+rules": (
        "You triage support tickets. Choose exactly one category.\n<definitions>\n"
        "billing: charges, refunds, invoices, payments\ntechnical: crashes, errors, bugs, slowness\n"
        "account: login, password, profile, email, deleting an account\n"
        "other: everything else (company questions, press, feedback)\n</definitions>\n"
        "If the ticket mixes topics, choose the one the customer needs fixed first."
    ),
}


@dataclass(frozen=True)
class Config:
    instruction_key: str
    demos: tuple[tuple[str, str], ...] = ()

    def render(self, ticket: str) -> str:
        shots = "".join(
            f"<example><ticket>{t}</ticket><label>{lab}</label></example>\n"
            for t, lab in self.demos
        )
        examples = f"\n<examples>\n{shots}</examples>\n" if shots else "\n"
        return (
            f"{INSTRUCTIONS[self.instruction_key]}{examples}\n<ticket>{ticket}</ticket>\n\n"
            "Reply with only the label: billing, technical, account or other."
        )


def predict(cfg: Config, ticket: str) -> str:
    out = llm.complete(cfg.render(ticket), max_tokens=20).text.lower()
    found = [(out.find(label), label) for label in LABELS if label in out]
    return min(found)[1] if found else "?"


def accuracy(cfg: Config, rows) -> float:
    return sum(predict(cfg, t) == gold for t, gold in rows) / len(rows)


# ----------------------------------------------------------------- the optimiser


def optimise(train, dev, n_demo_sets: int = 4, k: int = 4, seed: int = 0):
    """Grid over (instruction x demo-set). Demo sets are random k-subsets of TRAIN.

    The zero-shot config for each instruction is always a candidate, so the winner's dev score
    can never be worse than the best hand-written (zero-shot) prompt.
    """
    rng = random.Random(seed)
    demo_sets = [()] + [tuple(rng.sample(train, k)) for _ in range(n_demo_sets)]
    results = []
    for key, demos in itertools.product(INSTRUCTIONS, demo_sets):
        cfg = Config(key, demos)
        results.append((accuracy(cfg, dev), cfg))
    results.sort(key=lambda r: -r[0])
    return results


def part1() -> None:
    train, dev, test = split(DATA)
    baseline = Config("terse")  # what you'd write first
    print(f"split: train={len(train)} dev={len(dev)} test={len(test)}  (dev picks, test judges)")
    results = optimise(train, dev)
    best_dev, best = results[0]
    print(f"\n{'dev acc':>8}  instruction   demos")
    for acc, cfg in results[:5]:
        print(f"{acc:>8.0%}  {cfg.instruction_key:<12}  {len(cfg.demos)}")
    base_dev, base_test = accuracy(baseline, dev), accuracy(baseline, test)
    best_test = accuracy(best, test)
    print(f"\nhand-written baseline : dev {base_dev:.0%}  test {base_test:.0%}")
    print(
        f"optimised winner      : dev {best_dev:.0%}  test {best_test:.0%}  "
        f"({best.instruction_key}, {len(best.demos)} demos)"
    )
    print(
        "Caveat: with 10-ish dev examples, the winner is partly luck. Dev scores are optimistic; "
        "only the test split is an honest estimate (and it too is tiny)."
    )
    assert best_dev >= base_dev


# ----------------------------------------------------------------- part 2: DSPy


def dspy_demo(lm) -> dict:
    """Same task in DSPy. `lm` is a dspy.LM, e.g. dspy.LM('anthropic/claude-sonnet-5')."""
    from typing import Literal

    import dspy

    class Triage(dspy.Signature):
        """Classify a customer support ticket."""

        ticket: str = dspy.InputField()
        category: Literal["billing", "technical", "account", "other"] = dspy.OutputField()

    train, dev, test = split(DATA)
    mk = lambda rows: [dspy.Example(ticket=t, category=c).with_inputs("ticket") for t, c in rows]  # noqa: E731
    metric = lambda ex, pred, trace=None: ex.category == pred.category  # noqa: E731

    with dspy.context(lm=lm):
        base = dspy.Predict(Triage)  # signature only: DSPy writes the prompt
        evaluate = dspy.Evaluate(
            devset=mk(test), metric=metric, display_progress=False, num_threads=1
        )
        before = float(evaluate(base).score)
        optimiser = dspy.BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=3, max_labeled_demos=4
        )
        compiled = optimiser.compile(dspy.ChainOfThought(Triage), trainset=mk(train))
        after = float(evaluate(compiled).score)
    return {"before": before, "after": after, "demos": len(compiled.predictors()[0].demos)}


# ----------------------------------------------------------------- offline fakes


def offline_model(prompt: str) -> str:
    """Keyword matcher that learns from the <example>s in the prompt (simulating few-shot)."""
    base = {
        "billing": {"charged", "refund"},
        "technical": {"crashes", "error"},
        "account": {"password", "login"},
        "other": {"office"},
    }
    ext = {
        "billing": {"invoice", "receipt", "billed", "payment"},
        "technical": {"slow", "bug"},
        "account": {"locked", "email", "profile"},
        "other": {"partnership", "press", "love"},
    }
    stop = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "have",
        "your",
        "what",
        "when",
        "ticket",
        "account",
        "make",
        "does",
        "will",
        "never",
        "after",
        "since",
        "every",
        "time",
    }

    def tokenise(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3 and w not in stop}

    ticket = tokenise(re.findall(r"<ticket>(.*?)</ticket>", prompt, re.S)[-1])
    for label, words in base.items():
        if ticket & words:
            return label
    if "<definitions>" in prompt:  # an instruction with definitions lets the model use richer cues
        for label, words in ext.items():
            if ticket & words:
                return label
    for demo_ticket, label in re.findall(
        r"<example><ticket>(.*?)</ticket><label>(.*?)</label>", prompt
    ):
        if ticket & tokenise(demo_ticket):
            return label
    return "other"


def run_dspy_offline() -> None:
    """Run the DSPy code end-to-end against a scripted OpenAI-compatible HTTP server."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import dspy

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            msgs = body["messages"]
            text = msgs[-1]["content"]
            label = offline_model(
                f"<ticket>{text.split('[[ ## ticket ## ]]')[-1].split('Respond')[0]}</ticket>"
                + "<definitions>"
            )
            out = (
                "[[ ## reasoning ## ]]\nmatched keywords\n\n"
                if "reasoning" in msgs[0]["content"]
                else ""
            )
            out += f"[[ ## category ## ]]\n{label}\n\n[[ ## completed ## ]]"
            resp = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": out},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    lm = dspy.LM(
        "openai/fake-model",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="x",
        cache=False,
    )
    res = dspy_demo(lm)
    server.shutdown()
    print(
        f"DSPy (scripted server): before={res['before']:.0f}%  after={res['after']:.0f}%  "
        f"bootstrapped demos={res['demos']}"
    )
    assert res["demos"] > 0 and res["after"] >= 0


def main() -> None:
    if "--offline" in sys.argv:
        print("*** OFFLINE: keyword-matching fake that learns from in-prompt demos ***\n")
        with fake_llm([(r"(?s).*", offline_model)]):
            part1()
        print("\n--- Part 2: DSPy code against a scripted OpenAI-compatible server ---")
        run_dspy_offline()
        print("\nself-test passed")
    elif "--dspy" in sys.argv:
        import dspy

        provider, model = llm.resolve()
        res = dspy_demo(dspy.LM(f"{provider}/{model}"))
        print(res)
    else:
        print(f"provider: {llm.resolve()}")
        part1()


if __name__ == "__main__":
    main()
