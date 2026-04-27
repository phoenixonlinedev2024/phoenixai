"""Self-improvement — automated benchmarking, gap synthesis, capability evolution."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.core import Jarvis


@dataclass
class BenchmarkCase:
    id: str
    prompt: str
    expected_keywords: list[str]
    tool_required: str | None = None
    timeout: float = 30.0


@dataclass
class BenchmarkResult:
    case_id: str
    passed: bool
    score: float
    latency: float
    response: str
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": round(self.score, 3),
            "latency_s": round(self.latency, 3),
            "response_preview": self.response[:200],
            "error": self.error,
            "timestamp": self.timestamp,
        }


BUILTIN_BENCHMARKS = [
    BenchmarkCase("basic_math", "What is 17 × 23? Reply with just the number.", ["391"]),
    BenchmarkCase("reasoning", "Alice is taller than Bob, Bob taller than Carol. Who is shortest? One word.", ["carol", "Carol"]),
    BenchmarkCase("code", "Write a Python one-liner to reverse a string.", ["[::-1]"]),
    BenchmarkCase("factual", "What is the capital of France? One word.", ["paris", "Paris"]),
    BenchmarkCase("memory_recall", "My favourite colour is indigo. What is my favourite colour?", ["indigo", "Indigo"]),
    BenchmarkCase("tool_list", "List your available tools briefly.", ["tool", "Tool", "web", "file"]),
]


class BenchmarkRunner:
    """Runs benchmark cases against JARVIS and persists results for trend analysis."""

    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self._history_path = cfg.DATA_DIR / "benchmarks.jsonl"

    async def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        start = time.time()
        try:
            response = await asyncio.wait_for(self.jarvis.chat(case.prompt), timeout=case.timeout)
            latency = time.time() - start
            low = response.lower()
            hits = sum(1 for kw in case.expected_keywords if kw.lower() in low)
            score = hits / len(case.expected_keywords) if case.expected_keywords else 1.0
            return BenchmarkResult(case.id, passed=score >= 0.5, score=score, latency=latency, response=response)
        except asyncio.TimeoutError:
            return BenchmarkResult(case.id, False, 0.0, case.timeout, "", error="timeout")
        except Exception as exc:
            return BenchmarkResult(case.id, False, 0.0, time.time() - start, "", error=str(exc))

    async def run_suite(self, cases: list[BenchmarkCase] | None = None) -> dict:
        cases = cases or BUILTIN_BENCHMARKS
        results = [await self.run_case(c) for c in cases]
        passed = sum(1 for r in results if r.passed)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 3),
            "avg_score": round(sum(r.score for r in results) / len(results), 3),
            "avg_latency_s": round(sum(r.latency for r in results) / len(results), 3),
            "results": [r.to_dict() for r in results],
        }
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        return summary

    def load_history(self, limit: int = 20) -> list[dict]:
        if not self._history_path.exists():
            return []
        lines = self._history_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-limit:] if line]

    def trend(self) -> dict:
        history = self.load_history()
        if len(history) < 2:
            return {"trend": "insufficient data", "runs": len(history)}
        rates = [h["pass_rate"] for h in history]
        delta = rates[-1] - rates[0]
        return {
            "runs": len(history),
            "latest_pass_rate": rates[-1],
            "first_pass_rate": rates[0],
            "delta": round(delta, 3),
            "improving": delta > 0,
        }


class CapabilityEvolver:
    """Auto-synthesises tools to fill open capability gaps."""

    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis

    async def auto_fill_gaps(self, max_gaps: int = 3) -> list[str]:
        gaps = self.jarvis.memory.get_open_gaps()[:max_gaps]
        filled: list[str] = []
        for gap in gaps:
            description = gap["description"]
            try:
                from jarvis.tools.creator import synthesise_tool
                tool = await synthesise_tool(description, self.jarvis.client, self.jarvis.registry)
                if tool:
                    self.jarvis.memory.resolve_gap(gap["id"])
                    filled.append(tool.name)
                    print(f"[JARVIS Self-Improve] Synthesised tool: {tool.name}")
            except Exception as exc:
                print(f"[JARVIS Self-Improve] Gap synthesis failed for '{description}': {exc}")
        return filled

    def capability_report(self) -> dict:
        gaps = self.jarvis.memory.get_open_gaps()
        tools = self.jarvis.registry.all()
        dynamic = [t for t in tools if t.dynamic]
        return {
            "total_tools": len(tools),
            "dynamic_tools": len(dynamic),
            "open_gaps": len(gaps),
            "gap_descriptions": [g["description"] for g in gaps[:10]],
        }


class SelfImproveEngine:
    """Orchestrates benchmarking + gap filling on a background schedule."""

    def __init__(self, jarvis: "Jarvis") -> None:
        self.jarvis = jarvis
        self.benchmarks = BenchmarkRunner(jarvis)
        self.evolver = CapabilityEvolver(jarvis)

    async def run_cycle(self) -> dict:
        print("[JARVIS Self-Improve] Running improvement cycle...")
        bench = await self.benchmarks.run_suite()
        filled = await self.evolver.auto_fill_gaps()
        report = self.evolver.capability_report()
        return {
            "benchmark": bench,
            "gaps_filled": filled,
            "capability": report,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def start_background(self, interval_hours: float | None = None) -> None:
        interval = interval_hours or cfg.SELF_IMPROVE_INTERVAL_HOURS
        while True:
            try:
                await self.run_cycle()
            except Exception as exc:
                print(f"[JARVIS Self-Improve] Cycle error: {exc}")
            await asyncio.sleep(interval * 3600)
