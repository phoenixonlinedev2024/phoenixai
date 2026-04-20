"""Additional tests for jarvis.self_improve — evolver + engine + trend details."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.self_improve import (  # noqa: E402
    BUILTIN_BENCHMARKS,
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkRunner,
    CapabilityEvolver,
    SelfImproveEngine,
)


def _make_jarvis_mock(gaps=None, tools=None, chat_reply="response"):
    jarvis = MagicMock()
    jarvis.memory.get_open_gaps = MagicMock(return_value=gaps or [])
    jarvis.memory.resolve_gap = MagicMock()
    jarvis.registry.all = MagicMock(return_value=tools or [])
    jarvis.client = MagicMock()
    jarvis.chat = AsyncMock(return_value=chat_reply)
    return jarvis


# ── BUILTIN_BENCHMARKS catalogue ──────────────────────────────────────────────

def test_builtin_benchmarks_non_empty():
    assert len(BUILTIN_BENCHMARKS) >= 6


def test_builtin_benchmarks_all_have_ids():
    ids = [c.id for c in BUILTIN_BENCHMARKS]
    assert len(ids) == len(set(ids))  # unique IDs


def test_builtin_benchmarks_have_keywords():
    for case in BUILTIN_BENCHMARKS:
        assert case.expected_keywords, f"{case.id} has empty keywords"


# ── BenchmarkCase defaults ────────────────────────────────────────────────────

def test_benchmark_case_default_timeout():
    c = BenchmarkCase("x", "p", ["k"])
    assert c.timeout == 30.0
    assert c.tool_required is None


# ── BenchmarkResult error case ────────────────────────────────────────────────

def test_benchmark_result_with_error():
    r = BenchmarkResult("x", passed=False, score=0.0, latency=0.1, response="", error="timeout")
    d = r.to_dict()
    assert d["error"] == "timeout"
    assert d["passed"] is False


# ── BenchmarkRunner edge cases ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_case_timeout(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock()

    async def slow(*a, **k):
        import asyncio
        await asyncio.sleep(5)
        return "too late"

    jarvis.chat = AsyncMock(side_effect=slow)
    runner = BenchmarkRunner(jarvis)
    case = BenchmarkCase("slow", "wait", ["x"], timeout=0.05)
    result = await runner.run_case(case)
    assert result.passed is False
    assert result.error == "timeout"


@pytest.mark.asyncio
async def test_run_case_exception(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock()
    jarvis.chat = AsyncMock(side_effect=RuntimeError("api broke"))
    runner = BenchmarkRunner(jarvis)
    case = BenchmarkCase("broken", "q", ["x"])
    result = await runner.run_case(case)
    assert result.passed is False
    assert result.error == "api broke"


@pytest.mark.asyncio
async def test_run_suite_custom_cases(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock(chat_reply="foo")
    runner = BenchmarkRunner(jarvis)
    custom = [BenchmarkCase("c1", "hello", ["foo"])]
    summary = await runner.run_suite(cases=custom)
    assert summary["total"] == 1
    assert summary["passed"] == 1


def test_load_history_empty(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    runner = BenchmarkRunner(_make_jarvis_mock())
    assert runner.load_history() == []


def test_load_history_returns_last_n(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    import json as _json
    history = tmp_path / "benchmarks.jsonl"
    for i in range(5):
        history.write_text(
            (history.read_text() if history.exists() else "")
            + _json.dumps({"run": i, "pass_rate": 0.1 * i}) + "\n"
        )

    runner = BenchmarkRunner(_make_jarvis_mock())
    result = runner.load_history(limit=3)
    assert len(result) == 3
    assert result[-1]["run"] == 4


def test_trend_improving(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    import json as _json
    history = tmp_path / "benchmarks.jsonl"
    history.write_text(
        _json.dumps({"pass_rate": 0.3}) + "\n" +
        _json.dumps({"pass_rate": 0.7}) + "\n"
    )
    runner = BenchmarkRunner(_make_jarvis_mock())
    t = runner.trend()
    assert t["improving"] is True
    assert t["delta"] == 0.4
    assert t["latest_pass_rate"] == 0.7


def test_trend_regressing(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    import json as _json
    history = tmp_path / "benchmarks.jsonl"
    history.write_text(
        _json.dumps({"pass_rate": 0.9}) + "\n" +
        _json.dumps({"pass_rate": 0.5}) + "\n"
    )
    runner = BenchmarkRunner(_make_jarvis_mock())
    t = runner.trend()
    assert t["improving"] is False


# ── CapabilityEvolver ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_fill_gaps_synthesises_tools():
    gaps = [
        {"id": 1, "description": "parse YAML files"},
        {"id": 2, "description": "convert PDF to markdown"},
    ]
    jarvis = _make_jarvis_mock(gaps=gaps)

    fake_tool = MagicMock(name="fake_tool")
    fake_tool.name = "yaml_parser"

    evolver = CapabilityEvolver(jarvis)
    with patch("jarvis.tools.creator.synthesise_tool", AsyncMock(return_value=fake_tool)):
        filled = await evolver.auto_fill_gaps(max_gaps=2)

    assert "yaml_parser" in filled
    assert jarvis.memory.resolve_gap.call_count == 2


@pytest.mark.asyncio
async def test_auto_fill_gaps_handles_none_returned():
    gaps = [{"id": 7, "description": "impossible task"}]
    jarvis = _make_jarvis_mock(gaps=gaps)
    evolver = CapabilityEvolver(jarvis)
    with patch("jarvis.tools.creator.synthesise_tool", AsyncMock(return_value=None)):
        filled = await evolver.auto_fill_gaps()
    assert filled == []
    jarvis.memory.resolve_gap.assert_not_called()


@pytest.mark.asyncio
async def test_auto_fill_gaps_handles_exception(capsys):
    gaps = [{"id": 3, "description": "broken capability"}]
    jarvis = _make_jarvis_mock(gaps=gaps)
    evolver = CapabilityEvolver(jarvis)
    with patch("jarvis.tools.creator.synthesise_tool",
               AsyncMock(side_effect=RuntimeError("synth failed"))):
        filled = await evolver.auto_fill_gaps()
    assert filled == []
    assert "synth failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_auto_fill_gaps_respects_max_gaps():
    gaps = [{"id": i, "description": f"gap {i}"} for i in range(10)]
    jarvis = _make_jarvis_mock(gaps=gaps)
    tool = MagicMock()
    tool.name = "t"
    evolver = CapabilityEvolver(jarvis)
    synth = AsyncMock(return_value=tool)
    with patch("jarvis.tools.creator.synthesise_tool", synth):
        filled = await evolver.auto_fill_gaps(max_gaps=3)
    assert len(filled) == 3
    assert synth.call_count == 3


def test_capability_report_counts_dynamic_tools():
    t1 = MagicMock(dynamic=True)
    t2 = MagicMock(dynamic=False)
    t3 = MagicMock(dynamic=True)
    jarvis = _make_jarvis_mock(tools=[t1, t2, t3],
                                gaps=[{"description": "g1"}, {"description": "g2"}])
    evolver = CapabilityEvolver(jarvis)
    report = evolver.capability_report()
    assert report["total_tools"] == 3
    assert report["dynamic_tools"] == 2
    assert report["open_gaps"] == 2
    assert len(report["gap_descriptions"]) == 2


# ── SelfImproveEngine ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_self_improve_engine_run_cycle(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock(chat_reply="answer", gaps=[], tools=[])
    engine = SelfImproveEngine(jarvis)

    result = await engine.run_cycle()

    assert "benchmark" in result
    assert "gaps_filled" in result
    assert "capability" in result
    assert "timestamp" in result


def test_self_improve_engine_components():
    jarvis = _make_jarvis_mock()
    engine = SelfImproveEngine(jarvis)
    assert isinstance(engine.benchmarks, BenchmarkRunner)
    assert isinstance(engine.evolver, CapabilityEvolver)
