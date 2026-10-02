"""In-process metrics in the Prometheus text exposition format: counters with labels and fixed-bucket histograms. No dependency, no global state (one ``Metrics`` per app)."""

from __future__ import annotations

import bisect
import threading

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def _labels(labels: dict[str, str]) -> str:
    return (
        "{"
        + ",".join(
            f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
            for k, v in sorted(labels.items())
        )
        + "}"
        if labels
        else ""
    )


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = {}
        self._hist: dict[tuple[str, tuple], list[float]] = {}  # [bucket counts..., +Inf count, sum]
        self._help: dict[str, tuple[str, str]] = {}

    def describe(self, name: str, kind: str, help_text: str) -> None:
        self._help[name] = (kind, help_text)

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            key = (name, tuple(sorted(labels.items())))
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def observe(
        self, name: str, value: float, buckets: tuple[float, ...] = LATENCY_BUCKETS, **labels: str
    ) -> None:
        with self._lock:
            key = (name, tuple(sorted(labels.items())))
            row = self._hist.setdefault(key, [0.0] * (len(buckets) + 2))
            row[bisect.bisect_left(buckets, value)] += (
                1  # the first bucket whose upper bound is >= value
            )
            row[-1] += value

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] = value

    def value(self, name: str, **labels: str) -> float:
        return self._counters.get((name, tuple(sorted(labels.items()))), 0.0)

    def render(self, buckets: tuple[float, ...] = LATENCY_BUCKETS) -> str:
        lines: list[str] = []
        with self._lock:
            seen: set[str] = set()
            for (name, lab), v in sorted(self._counters.items()):
                if name not in seen:
                    kind, help_text = self._help.get(name, ("counter", name))
                    lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
                    seen.add(name)
                lines.append(f"{name}{_labels(dict(lab))} {v:g}")
            for (name, lab), row in sorted(self._hist.items()):
                if name not in seen:
                    lines += [
                        f"# HELP {name} {self._help.get(name, ('histogram', name))[1]}",
                        f"# TYPE {name} histogram",
                    ]
                    seen.add(name)
                cumulative = 0.0
                for ub, count in zip(buckets, row[: len(buckets)], strict=True):
                    cumulative += count
                    lines.append(
                        f"{name}_bucket{_labels({**dict(lab), 'le': f'{ub:g}'})} {cumulative:g}"
                    )
                cumulative += row[len(buckets)]
                lines.append(f"{name}_bucket{_labels({**dict(lab), 'le': '+Inf'})} {cumulative:g}")
                lines.append(f"{name}_sum{_labels(dict(lab))} {row[-1]:g}")
                lines.append(f"{name}_count{_labels(dict(lab))} {cumulative:g}")
        return "\n".join(lines) + "\n"
