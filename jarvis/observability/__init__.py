"""Observability — structured metrics, latency histograms, and Prometheus export."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from jarvis.config import cfg


@dataclass
class Counter:
    name: str
    value: int = 0

    def inc(self, amount: int = 1) -> None:
        self.value += amount


@dataclass
class Histogram:
    name: str
    observations: list[float] = field(default_factory=list)
    _max_size: int = 1000

    def observe(self, value: float) -> None:
        self.observations.append(value)
        if len(self.observations) > self._max_size:
            self.observations = self.observations[-self._max_size:]

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def total(self) -> float:
        return sum(self.observations)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def p95(self) -> float:
        if not self.observations:
            return 0.0
        s = sorted(self.observations)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def p99(self) -> float:
        if not self.observations:
            return 0.0
        s = sorted(self.observations)
        return s[min(int(len(s) * 0.99), len(s) - 1)]


class MetricsRegistry:
    """Central registry for all JARVIS runtime metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._start_time = time.time()

    # ── Accessors ────────────────────────────────────────────────────────

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name=name)
        return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name=name)
        return self._histograms[name]

    def inc(self, name: str, amount: int = 1) -> None:
        self.counter(name).inc(amount)

    def observe(self, name: str, value: float) -> None:
        self.histogram(name).observe(value)

    # ── Context managers ─────────────────────────────────────────────────

    @contextmanager
    def time(self, name: str):
        start = time.time()
        try:
            yield
        finally:
            self.observe(name, time.time() - start)

    @asynccontextmanager
    async def atime(self, name: str):
        start = time.time()
        try:
            yield
        finally:
            self.observe(name, time.time() - start)

    # ── Export formats ───────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "counters": {k: v.value for k, v in self._counters.items()},
            "histograms": {
                k: {
                    "count": v.count,
                    "avg_ms": round(v.avg * 1000, 2),
                    "p95_ms": round(v.p95 * 1000, 2),
                    "p99_ms": round(v.p99 * 1000, 2),
                    "sum_ms": round(v.total * 1000, 2),
                }
                for k, v in self._histograms.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def prometheus_text(self) -> str:
        lines = [
            "# HELP jarvis_uptime_seconds Seconds since JARVIS started",
            f"jarvis_uptime_seconds {round(time.time() - self._start_time, 2)}",
        ]
        for name, c in self._counters.items():
            safe = name.replace(".", "_").replace("-", "_")
            lines += [f"# HELP jarvis_{safe}_total Counter", f"jarvis_{safe}_total {c.value}"]
        for name, h in self._histograms.items():
            safe = name.replace(".", "_").replace("-", "_")
            lines += [
                f"# HELP jarvis_{safe} Latency histogram",
                f"jarvis_{safe}_count {h.count}",
                f"jarvis_{safe}_sum {round(h.total, 6)}",
            ]
        return "\n".join(lines) + "\n"

    def log_snapshot(self) -> None:
        log_dir = cfg.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.snapshot()) + "\n")


# Global metrics instance
metrics = MetricsRegistry()
