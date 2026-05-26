"""Tests for jarvis.self_improve — benchmarks, gap synthesis, trend tracking."""

import json
import pytest
from pathlib import Path
from jarvis.self_improve import BenchmarkCase, BenchmarkResult, BenchmarkRunner, CapabilityEvolver


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
