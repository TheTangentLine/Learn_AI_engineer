"""A discrete-event model of batching policies for LLM serving: static batching against continuous (iteration-level) batching.

Decoding is done one token at a time for every sequence in the batch ("a step"). A step costs a fixed part (reading all the weights once: ``base``) plus a part per
sequence in the batch (``per_seq``: the attention over its cache and its share of the matrix products). Because ``base`` dominates for small models and short contexts,
adding a sequence to a step is almost free: batching raises throughput.

  STATIC      form a batch from the waiting requests, run until the LONGEST one finishes, then return everything. A finished sequence keeps its slot (it is padding).
  CONTINUOUS  after EVERY step, finished sequences leave and waiting requests join (their prompt is processed in that step). No slot is ever idle while someone waits.

The model is deliberately simple (no chunked prefill, no memory limit, a linear step cost); ``fit_step_model`` fits the two constants from real measurements so the
simulator can be compared with a real server (Day 2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    id: int
    arrival: float
    prompt: int
    output: int  # tokens to generate


@dataclass
class Done:
    request: Request
    first_token: float
    finish: float

    @property
    def latency(self) -> float:
        return self.finish - self.request.arrival

    @property
    def ttft(self) -> float:
        return self.first_token - self.request.arrival


@dataclass(frozen=True)
class StepModel:
    base: float  # seconds per step, whatever the batch size
    per_seq: float  # extra seconds per sequence in the step
    prefill_per_token: float  # seconds per prompt token, charged when a request is admitted

    def step(self, batch: int) -> float:
        return self.base + self.per_seq * batch


@dataclass
class Report:
    policy: str
    done: list[Done] = field(default_factory=list)
    makespan: float = 0.0
    steps: int = 0
    slot_steps_used: int = 0
    slot_steps_total: int = 0

    @property
    def throughput(self) -> float:
        return sum(d.request.output for d in self.done) / self.makespan if self.makespan else 0.0

    @property
    def utilisation(self) -> float:
        """Share of batch slots that produced a real token (the rest was padding or idle)."""
        return self.slot_steps_used / self.slot_steps_total if self.slot_steps_total else 0.0

    def latency(self, p: float) -> float:
        from loadgen import percentile

        return percentile([d.latency for d in self.done], p)

    def ttft(self, p: float) -> float:
        from loadgen import percentile

        return percentile([d.ttft for d in self.done], p)


def simulate_static(requests: list[Request], model: StepModel, max_batch: int) -> Report:
    rep = Report("static")
    pending = sorted(requests, key=lambda r: (r.arrival, r.id))
    t, i = 0.0, 0
    while i < len(pending):
        t = max(t, pending[i].arrival)  # idle until the next request arrives
        batch = []
        while i < len(pending) and pending[i].arrival <= t and len(batch) < max_batch:
            batch.append(pending[i])
            i += 1
        t += sum(r.prompt for r in batch) * model.prefill_per_token  # prefill the whole batch first
        longest = max(r.output for r in batch)
        first = {}
        for s in range(1, longest + 1):
            t += model.step(
                len(batch)
            )  # the batch keeps its full size: finished sequences are padding
            rep.steps += 1
            rep.slot_steps_total += len(batch)
            for r in batch:
                if s <= r.output:
                    rep.slot_steps_used += 1
                if s == 1:
                    first[r.id] = t
        # results are returned together when the longest sequence finishes
        rep.done += [Done(r, first[r.id], t) for r in batch]
        rep.makespan = t
    return rep


def simulate_continuous(requests: list[Request], model: StepModel, max_batch: int) -> Report:
    rep = Report("continuous")
    waiting = sorted(requests, key=lambda r: (r.arrival, r.id))
    active: dict[int, list] = {}  # id -> [request, tokens left, first_token_time]
    t, i = 0.0, 0
    while i < len(waiting) or active:
        if not active and waiting[i].arrival > t:
            t = waiting[i].arrival
        admitted = 0
        while i < len(waiting) and waiting[i].arrival <= t and len(active) < max_batch:
            r = waiting[i]
            active[r.id] = [r, r.output, None]
            t += r.prompt * model.prefill_per_token  # its prompt is processed before the next step
            i += 1
            admitted += 1
        t += model.step(len(active))
        rep.steps += 1
        rep.slot_steps_total += len(active)
        rep.slot_steps_used += len(active)
        finished = []
        for rid, st in active.items():
            st[1] -= 1
            if st[2] is None:
                st[2] = t
            if st[1] == 0:
                finished.append(rid)
        for rid in finished:
            r, _, ft = active.pop(rid)
            rep.done.append(Done(r, ft, t))
        rep.makespan = t
    return rep


def poisson_workload(
    n: int,
    rate: float,
    *,
    prompt: tuple[int, int] = (40, 120),
    output: tuple[int, int] = (20, 120),
    seed: int = 0,
    long_tail: float = 0.0,
) -> list[Request]:
    """``n`` requests with exponential inter-arrival times (mean 1/rate) and uniformly random lengths; ``long_tail`` is the share of requests whose output is 4x longer
    (a few long answers are what hurts static batching most)."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for k in range(n):
        t += rng.expovariate(rate)
        o = rng.randint(*output)
        if rng.random() < long_tail:
            o *= 4
        out.append(Request(k, t, rng.randint(*prompt), o))
    return out


def fit_step_model(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares line through (batch size, seconds per step) measurements: returns (base, per_seq). With one point the whole step is ``base``."""
    if len({b for b, _ in points}) < 2:
        return points[0][1], 0.0
    n = len(points)
    sx, sy = sum(b for b, _ in points), sum(s for _, s in points)
    sxx, sxy = sum(b * b for b, _ in points), sum(b * s for b, s in points)
    per = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - per * sx) / n, per
