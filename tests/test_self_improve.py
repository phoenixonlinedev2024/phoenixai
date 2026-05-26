"""Tests for jarvis.self_improve — benchmarks, gap synthesis, trend tracking."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.self_improve import BenchmarkCase, BenchmarkResult, BenchmarkRunner, CapabilityEvolver, SelfImproveEngine


# ── BenchmarkCase / Result ───────────────────────────────────────────────────

def test_result_to_dict():
    r = BenchmarkResult(
        case_id="math",
        passed=True,
        score=1.0,
        latency=0.5,
        response="391",
    )
    d = r.to_dict()
    assert d["case_id"] == "math"
    assert d["passed"] is True
    assert d["score"] == 1.0
    assert "latency_s" in d
    assert "timestamp" in d


def test_result_response_preview_truncated():
    long_response = "x" * 500
    r = BenchmarkResult("id", True, 1.0, 0.1, long_response)
    d = r.to_dict()
    assert len(d["response_preview"]) <= 200


# ── BenchmarkRunner ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_jarvis(tmp_path, monkeypatch):
    """Minimal jarvis-like object that returns a canned response."""
    class FakeMemory:
        def get_open_gaps(self):
            return []

    class FakeRegistry:
        def all(self):
            return []

    class FakeJarvis:
        memory = FakeMemory()
        registry = FakeRegistry()

        async def chat(self, prompt):
            # Return a response that passes 'basic_math' (contains 391)
            if "17" in prompt and "23" in prompt:
                return "391"
            if "capital" in prompt.lower():
                return "Paris"
            if "Alice" in prompt:
                return "Carol"
            if "reverse" in prompt.lower():
                return "s[::-1]"
            if "favourite colour" in prompt.lower():
                return "indigo"
            return "tool Tool web file"

    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    return FakeJarvis()


@pytest.mark.asyncio
async def test_run_case_pass(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    runner = BenchmarkRunner(mock_jarvis)
    case = BenchmarkCase("math", "What is 17 × 23?", ["391"])
    result = await runner.run_case(case)
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_run_case_fail(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    runner = BenchmarkRunner(mock_jarvis)
    case = BenchmarkCase("impossible", "What is the airspeed velocity?", ["swallow", "african"])
    result = await runner.run_case(case)
    assert result.passed is False


@pytest.mark.asyncio
async def test_run_suite_saves_history(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    runner = BenchmarkRunner(mock_jarvis)
    summary = await runner.run_suite()

    history_path = tmp_path / "benchmarks.jsonl"
    assert history_path.exists()
    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert "pass_rate" in saved
    assert "results" in saved


@pytest.mark.asyncio
async def test_run_suite_summary_fields(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    runner = BenchmarkRunner(mock_jarvis)
    summary = await runner.run_suite()
    for field in ("total", "passed", "failed", "pass_rate", "avg_score", "avg_latency_s", "results"):
        assert field in summary, f"Missing field: {field}"


def test_trend_no_data(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    runner = BenchmarkRunner(mock_jarvis)
    t = runner.trend()
    assert "trend" in t or "runs" in t


@pytest.mark.asyncio
async def test_trend_with_data(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    runner = BenchmarkRunner(mock_jarvis)
    await runner.run_suite()
    await runner.run_suite()
    t = runner.trend()
    assert "runs" in t
    assert t["runs"] == 2


# ── CapabilityEvolver ────────────────────────────────────────────────────────

def test_capability_report_structure(mock_jarvis):
    evolver = CapabilityEvolver(mock_jarvis)
    report = evolver.capability_report()
    assert "total_tools" in report
    assert "dynamic_tools" in report
    assert "open_gaps" in report
    assert isinstance(report["gap_descriptions"], list)


# ── Additional BenchmarkRunner tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_case_no_keyword_match_fails(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    case = BenchmarkCase("nomatch", "What is the airspeed velocity?", ["swallow", "african"])
    result = await runner.run_case(case)
    assert result.score == 0.0
    assert result.passed is False


def test_benchmark_runner_history_path_is_jsonl(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    assert runner._history_path.suffix == ".jsonl"


def test_capability_report_gaps_capped_at_10(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    class FakeMemory:
        def get_open_gaps(self):
            return [{"id": i, "description": f"gap {i}"} for i in range(15)]

    class FakeRegistry:
        def all(self):
            return []

    class FakeJarvis:
        memory = FakeMemory()
        registry = FakeRegistry()

        async def chat(self, prompt):
            return "resp"

    evolver = CapabilityEvolver(FakeJarvis())
    report = evolver.capability_report()
    assert len(report["gap_descriptions"]) <= 10
    assert report["open_gaps"] == 15


@pytest.mark.asyncio
async def test_run_suite_avg_latency_present(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    summary = await runner.run_suite()
    assert "avg_latency_s" in summary
    assert summary["avg_latency_s"] >= 0.0


def test_benchmark_case_has_correct_id(mock_jarvis):
    case = BenchmarkCase("my_id", "prompt", ["keyword"])
    assert case.id == "my_id"
    assert case.prompt == "prompt"
    assert "keyword" in case.expected_keywords


# ── BenchmarkResult error field ───────────────────────────────────────────────

def test_result_to_dict_includes_error_field():
    r = BenchmarkResult("case1", False, 0.0, 1.2, "", error="timeout")
    d = r.to_dict()
    assert d["error"] == "timeout"


def test_result_to_dict_error_none_by_default():
    r = BenchmarkResult("case2", True, 1.0, 0.3, "ok")
    d = r.to_dict()
    assert d["error"] is None


# ── run_case timeout / exception paths ───────────────────────────────────────

@pytest.mark.asyncio
async def test_run_case_timeout_sets_error(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    class TimeoutJarvis:
        async def chat(self, prompt):
            raise asyncio.TimeoutError()

    runner = BenchmarkRunner(TimeoutJarvis())
    case = BenchmarkCase("t", "prompt", ["keyword"], timeout=0.001)
    result = await runner.run_case(case)
    assert result.passed is False
    assert result.error == "timeout"
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_run_case_exception_sets_error(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    class ErrorJarvis:
        async def chat(self, prompt):
            raise RuntimeError("api exploded")

    runner = BenchmarkRunner(ErrorJarvis())
    case = BenchmarkCase("err", "prompt", ["keyword"])
    result = await runner.run_case(case)
    assert result.passed is False
    assert "api exploded" in result.error
    assert result.score == 0.0


# ── load_history ──────────────────────────────────────────────────────────────

def test_load_history_empty_when_no_file(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    assert runner.load_history() == []


@pytest.mark.asyncio
async def test_load_history_returns_entries(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    await runner.run_suite()
    await runner.run_suite()
    history = runner.load_history()
    assert len(history) == 2
    assert "pass_rate" in history[0]


@pytest.mark.asyncio
async def test_load_history_respects_limit(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    for _ in range(5):
        await runner.run_suite()
    history = runner.load_history(limit=3)
    assert len(history) == 3


# ── trend fields ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trend_improving_field(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    await runner.run_suite()
    await runner.run_suite()
    t = runner.trend()
    assert "improving" in t
    assert "delta" in t
    assert "latest_pass_rate" in t
    assert "first_pass_rate" in t


# ── SelfImproveEngine ────────────────────────────────────────────────────────

def test_self_improve_engine_stores_components(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    engine = SelfImproveEngine(mock_jarvis)
    assert isinstance(engine.benchmarks, BenchmarkRunner)
    assert isinstance(engine.evolver, CapabilityEvolver)
    assert engine.jarvis is mock_jarvis


@pytest.mark.asyncio
async def test_self_improve_engine_run_cycle_structure(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    engine = SelfImproveEngine(mock_jarvis)
    result = await engine.run_cycle()
    assert "benchmark" in result
    assert "gaps_filled" in result
    assert "capability" in result
    assert "timestamp" in result


# ── CapabilityEvolver.auto_fill_gaps ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_fill_gaps_empty_when_no_gaps(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    class FakeMemory:
        def get_open_gaps(self):
            return []

    class FakeRegistry:
        def all(self):
            return []

    class FakeJarvis:
        memory = FakeMemory()
        registry = FakeRegistry()
        client = MagicMock()

        async def chat(self, prompt):
            return "resp"

    evolver = CapabilityEvolver(FakeJarvis())
    filled = await evolver.auto_fill_gaps()
    assert filled == []


# ── BUILTIN_BENCHMARKS constant ───────────────────────────────────────────────

def test_builtin_benchmarks_has_six_cases():
    from jarvis.self_improve import BUILTIN_BENCHMARKS
    assert len(BUILTIN_BENCHMARKS) == 6


def test_builtin_benchmarks_each_has_id_and_keywords():
    from jarvis.self_improve import BUILTIN_BENCHMARKS
    for case in BUILTIN_BENCHMARKS:
        assert isinstance(case.id, str) and len(case.id) > 0
        assert isinstance(case.expected_keywords, list)
        assert len(case.expected_keywords) >= 1


# ── BenchmarkCase default fields ──────────────────────────────────────────────

def test_benchmark_case_default_timeout():
    case = BenchmarkCase("t", "prompt", ["kw"])
    assert case.timeout == 30.0


def test_benchmark_case_default_tool_required_is_none():
    case = BenchmarkCase("t", "prompt", ["kw"])
    assert case.tool_required is None


# ── BenchmarkResult timestamp default ────────────────────────────────────────

def test_benchmark_result_timestamp_default_is_set():
    r = BenchmarkResult("c", True, 1.0, 0.1, "ok")
    assert r.timestamp is not None
    assert "T" in r.timestamp  # ISO 8601 format contains 'T'


# ── run_suite with custom cases ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_suite_with_custom_cases(mock_jarvis, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(mock_jarvis)
    custom = [BenchmarkCase("custom1", "What is 17 × 23?", ["391"])]
    summary = await runner.run_suite(cases=custom)
    assert summary["total"] == 1
    assert summary["passed"] == 1
