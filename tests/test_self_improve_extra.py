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


@pytest.mark.asyncio
async def test_start_background_runs_one_cycle_then_sleeps(monkeypatch):
    """start_background loops forever; we cancel after first cycle."""
    import asyncio
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "SELF_IMPROVE_INTERVAL_HOURS", 0.0)

    jarvis = _make_jarvis_mock(chat_reply="ok", gaps=[], tools=[])
    engine = SelfImproveEngine(jarvis)

    cycle_calls = []

    async def fake_run_cycle():
        cycle_calls.append(1)
        return {"benchmark": {}, "gaps_filled": [], "capability": {}, "timestamp": ""}

    monkeypatch.setattr(engine, "run_cycle", fake_run_cycle)

    # Cancel after a tiny sleep to let one iteration run
    async def cancel_soon(task):
        await asyncio.sleep(0.01)
        task.cancel()

    task = asyncio.create_task(engine.start_background(interval_hours=0.0))
    await asyncio.gather(cancel_soon(task), task, return_exceptions=True)

    assert len(cycle_calls) >= 1


@pytest.mark.asyncio
async def test_start_background_handles_cycle_exception(monkeypatch, capsys):
    """Cycle exceptions should be caught and logged, not propagate."""
    import asyncio
    from jarvis.config import cfg

    jarvis = _make_jarvis_mock()
    engine = SelfImproveEngine(jarvis)

    call_count = [0]

    async def flaky_cycle():
        call_count[0] += 1
        raise RuntimeError("cycle failed")

    monkeypatch.setattr(engine, "run_cycle", flaky_cycle)

    async def cancel_soon(task):
        await asyncio.sleep(0.02)
        task.cancel()

    task = asyncio.create_task(engine.start_background(interval_hours=0.0))
    await asyncio.gather(cancel_soon(task), task, return_exceptions=True)

    assert call_count[0] >= 1
    assert "Cycle error" in capsys.readouterr().out


# ── BenchmarkResult.to_dict() fields ─────────────────────────────────────────

def test_benchmark_result_to_dict_all_fields():
    result = BenchmarkResult(
        case_id="test_case",
        passed=True,
        score=0.875,
        latency=0.123,
        response="The answer is 42",
        error=None,
    )
    d = result.to_dict()
    assert d["case_id"] == "test_case"
    assert d["passed"] is True
    assert d["score"] == 0.875
    assert d["latency_s"] == 0.123
    assert "The answer is 42" in d["response_preview"]
    assert d["error"] is None
    assert "T" in d["timestamp"]


def test_benchmark_result_to_dict_truncates_preview():
    long_response = "x" * 500
    result = BenchmarkResult("case", True, 1.0, 0.1, long_response)
    d = result.to_dict()
    assert len(d["response_preview"]) == 200


def test_benchmark_result_with_error():
    result = BenchmarkResult("fail_case", False, 0.0, 30.0, "", error="timeout")
    d = result.to_dict()
    assert d["passed"] is False
    assert d["error"] == "timeout"


# ── BenchmarkRunner.run_case() score calculation ──────────────────────────────

@pytest.mark.asyncio
async def test_run_case_partial_keyword_match(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock(chat_reply="The result is 391 approximately")
    runner = BenchmarkRunner(jarvis)
    case = BenchmarkCase(
        id="partial",
        prompt="What is 17x23?",
        expected_keywords=["391", "not_present"],
    )
    result = await runner.run_case(case)
    assert result.score == pytest.approx(0.5)
    assert result.passed is True  # score >= 0.5 threshold passes


@pytest.mark.asyncio
async def test_run_case_all_keywords_found(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock(chat_reply="carol is the shortest person")
    runner = BenchmarkRunner(jarvis)
    case = BenchmarkCase(
        id="full",
        prompt="Who is shortest?",
        expected_keywords=["carol", "shortest"],
    )
    result = await runner.run_case(case)
    assert result.score == pytest.approx(1.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_run_case_timeout(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    import asyncio

    jarvis = _make_jarvis_mock()
    jarvis.chat = AsyncMock(side_effect=asyncio.TimeoutError())
    runner = BenchmarkRunner(jarvis)
    case = BenchmarkCase("timeout_case", "slow question", ["answer"], timeout=0.01)
    result = await runner.run_case(case)
    assert result.passed is False
    assert result.error == "timeout"


@pytest.mark.asyncio
async def test_run_case_exception(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock()
    jarvis.chat = AsyncMock(side_effect=RuntimeError("api down"))
    runner = BenchmarkRunner(jarvis)
    case = BenchmarkCase("error_case", "question", ["answer"])
    result = await runner.run_case(case)
    assert result.passed is False
    assert "api down" in result.error


# ── BenchmarkRunner.trend() ───────────────────────────────────────────────────

def test_trend_insufficient_data(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    jarvis = _make_jarvis_mock()
    runner = BenchmarkRunner(jarvis)
    result = runner.trend()
    assert result["trend"] == "insufficient data"
    assert result["runs"] == 0


def test_trend_improving(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    import json

    jarvis = _make_jarvis_mock()
    runner = BenchmarkRunner(jarvis)
    runner._history_path.parent.mkdir(parents=True, exist_ok=True)
    runner._history_path.write_text(
        json.dumps({"pass_rate": 0.5, "results": []}) + "\n" +
        json.dumps({"pass_rate": 0.8, "results": []}) + "\n"
    )
    t = runner.trend()
    assert t["improving"] is True
    assert t["delta"] == pytest.approx(0.3)


def test_trend_declining(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    import json

    jarvis = _make_jarvis_mock()
    runner = BenchmarkRunner(jarvis)
    runner._history_path.parent.mkdir(parents=True, exist_ok=True)
    runner._history_path.write_text(
        json.dumps({"pass_rate": 0.9, "results": []}) + "\n" +
        json.dumps({"pass_rate": 0.6, "results": []}) + "\n"
    )
    t = runner.trend()
    assert t["improving"] is False
    assert t["delta"] == pytest.approx(-0.3)


# ── CapabilityEvolver.capability_report() ────────────────────────────────────

def test_capability_report_counts_dynamic_tools():
    static_tool = MagicMock()
    static_tool.dynamic = False
    dynamic_tool = MagicMock()
    dynamic_tool.dynamic = True

    jarvis = _make_jarvis_mock(
        tools=[static_tool, dynamic_tool, dynamic_tool],
        gaps=[{"id": "g1", "description": "need X"}, {"id": "g2", "description": "need Y"}],
    )
    evolver = CapabilityEvolver(jarvis)
    report = evolver.capability_report()
    assert report["total_tools"] == 3
    assert report["dynamic_tools"] == 2
    assert report["open_gaps"] == 2
    assert "need X" in report["gap_descriptions"]


def test_capability_report_no_gaps_no_dynamic():
    static_tool = MagicMock()
    static_tool.dynamic = False

    jarvis = _make_jarvis_mock(tools=[static_tool, static_tool], gaps=[])
    evolver = CapabilityEvolver(jarvis)
    report = evolver.capability_report()
    assert report["total_tools"] == 2
    assert report["dynamic_tools"] == 0
    assert report["open_gaps"] == 0
    assert report["gap_descriptions"] == []


# ── BUILTIN_BENCHMARKS sanity ─────────────────────────────────────────────────

def test_builtin_benchmarks_count():
    assert len(BUILTIN_BENCHMARKS) >= 5


def test_builtin_benchmarks_all_have_prompts():
    for case in BUILTIN_BENCHMARKS:
        assert len(case.prompt) > 5, f"Case '{case.id}' has too short prompt"
        assert len(case.expected_keywords) >= 1, f"Case '{case.id}' has no keywords"


# ── BenchmarkCase custom values ──────────────────────────────────────────────

def test_benchmark_case_custom_timeout():
    case = BenchmarkCase("c", "prompt", ["kw"], timeout=60.0)
    assert case.timeout == 60.0


def test_benchmark_case_with_tool_required():
    case = BenchmarkCase("c", "prompt", ["kw"], tool_required="web_search")
    assert case.tool_required == "web_search"


# ── BenchmarkRunner.run_suite with custom cases ───────────────────────────────

@pytest.mark.asyncio
async def test_run_suite_with_single_custom_case(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.self_improve.cfg.DATA_DIR", tmp_path)
    jarvis = _make_jarvis_mock()
    jarvis.chat = AsyncMock(return_value="391")
    runner = BenchmarkRunner(jarvis)
    custom = [BenchmarkCase("math", "17*23", ["391"])]
    result = await runner.run_suite(cases=custom)
    assert result["total"] == 1
    assert result["passed"] == 1
    assert result["pass_rate"] == 1.0


@pytest.mark.asyncio
async def test_run_suite_failed_case_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.self_improve.cfg.DATA_DIR", tmp_path)
    jarvis = _make_jarvis_mock()
    jarvis.chat = AsyncMock(return_value="wrong answer")
    runner = BenchmarkRunner(jarvis)
    custom = [BenchmarkCase("math", "17*23", ["391"])]
    result = await runner.run_suite(cases=custom)
    assert result["total"] == 1
    assert result["passed"] == 0
    assert result["failed"] == 1


# ── CapabilityEvolver.auto_fill_gaps with empty gaps ─────────────────────────

@pytest.mark.asyncio
async def test_auto_fill_gaps_empty_returns_empty_list():
    jarvis = _make_jarvis_mock(gaps=[])
    evolver = CapabilityEvolver(jarvis)
    filled = await evolver.auto_fill_gaps()
    assert filled == []


# ── SelfImproveEngine components ──────────────────────────────────────────────

def test_self_improve_engine_has_benchmarks_and_evolver():
    jarvis = _make_jarvis_mock()
    engine = SelfImproveEngine(jarvis)
    assert isinstance(engine.benchmarks, BenchmarkRunner)
    assert isinstance(engine.evolver, CapabilityEvolver)


# ── BenchmarkResult score rounding ────────────────────────────────────────────

def test_benchmark_result_to_dict_score_rounded_to_3_decimals():
    r = BenchmarkResult("c", True, 0.123456, 0.5, "resp")
    d = r.to_dict()
    assert d["score"] == 0.123


def test_benchmark_result_to_dict_latency_rounded():
    r = BenchmarkResult("c", True, 1.0, 1.23456, "resp")
    d = r.to_dict()
    assert d["latency_s"] == 1.235


# ── CapabilityEvolver.auto_fill_gaps with gaps ────────────────────────────────

@pytest.mark.asyncio
async def test_auto_fill_gaps_fills_one_gap(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.memory.get_open_gaps.return_value = [{"id": 1, "description": "need csv parser"}]
    fake_tool = MagicMock()
    fake_tool.name = "csv_parser"
    jarvis.client = AsyncMock()
    jarvis.registry = MagicMock()

    with patch("jarvis.tools.creator.synthesise_tool", AsyncMock(return_value=fake_tool)):
        evolver = CapabilityEvolver(jarvis)
        filled = await evolver.auto_fill_gaps()

    assert "csv_parser" in filled
    jarvis.memory.resolve_gap.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_auto_fill_gaps_synthesis_exception_continues(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.memory.get_open_gaps.return_value = [{"id": 2, "description": "failing gap"}]
    jarvis.client = AsyncMock()
    jarvis.registry = MagicMock()

    with patch("jarvis.tools.creator.synthesise_tool", AsyncMock(side_effect=RuntimeError("boom"))):
        evolver = CapabilityEvolver(jarvis)
        filled = await evolver.auto_fill_gaps()

    assert filled == []


# ── CapabilityEvolver.capability_report ───────────────────────────────────────

def test_capability_report_has_all_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.memory.get_open_gaps.return_value = []
    jarvis.registry.all.return_value = []
    evolver = CapabilityEvolver(jarvis)
    report = evolver.capability_report()
    for key in ("total_tools", "dynamic_tools", "open_gaps", "gap_descriptions"):
        assert key in report


def test_capability_report_gap_descriptions_truncated_to_10(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.memory.get_open_gaps.return_value = [{"id": i, "description": f"gap{i}"} for i in range(15)]
    jarvis.registry.all.return_value = []
    evolver = CapabilityEvolver(jarvis)
    report = evolver.capability_report()
    assert len(report["gap_descriptions"]) <= 10


# ── SelfImproveEngine.run_cycle response keys ─────────────────────────────────

@pytest.mark.asyncio
async def test_run_cycle_has_all_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.chat = AsyncMock(return_value="ok answer with hello world")
    jarvis.memory.get_open_gaps.return_value = []
    jarvis.registry.all.return_value = []
    engine = SelfImproveEngine(jarvis)
    result = await engine.run_cycle()
    for key in ("benchmark", "gaps_filled", "capability", "timestamp"):
        assert key in result


# ── BenchmarkRunner.run_suite summary keys ────────────────────────────────────

@pytest.mark.asyncio
async def test_run_suite_summary_has_failed_key(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.chat = AsyncMock(return_value="hello world")
    case = BenchmarkCase(id="t1", prompt="say hi", expected_keywords=["hello"])
    runner = BenchmarkRunner(jarvis)
    result = await runner.run_suite([case])
    assert "failed" in result
    assert result["failed"] == result["total"] - result["passed"]


@pytest.mark.asyncio
async def test_run_suite_appends_to_history(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.config.cfg.DATA_DIR", tmp_path)
    jarvis = MagicMock()
    jarvis.chat = AsyncMock(return_value="pong")
    case = BenchmarkCase(id="ping", prompt="ping", expected_keywords=["pong"])
    runner = BenchmarkRunner(jarvis)
    await runner.run_suite([case])
    await runner.run_suite([case])
    history = runner.load_history()
    assert len(history) == 2
