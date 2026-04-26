"""Tests for jarvis.observability — metrics registry."""

import time
import pytest
from jarvis.observability import MetricsRegistry


@pytest.fixture
def reg():
    return MetricsRegistry()


def test_counter_starts_at_zero(reg):
    c = reg.counter("requests")
    assert c.value == 0


def test_counter_inc(reg):
    reg.inc("hits")
    reg.inc("hits")
    reg.inc("hits", 3)
    assert reg.counter("hits").value == 5


def test_histogram_observe(reg):
    reg.observe("latency", 0.1)
    reg.observe("latency", 0.2)
    h = reg.histogram("latency")
    assert h.count == 2
    assert abs(h.avg - 0.15) < 1e-9


def test_histogram_p95(reg):
    for i in range(100):
        reg.observe("resp", float(i))
    h = reg.histogram("resp")
    assert h.p95 >= 90.0


def test_time_context_manager(reg):
    with reg.time("op"):
        time.sleep(0.01)
    h = reg.histogram("op")
    assert h.count == 1
    assert h.avg > 0.005


@pytest.mark.asyncio
async def test_atime_context_manager(reg):
    import asyncio
    async with reg.atime("async_op"):
        await asyncio.sleep(0.01)
    assert reg.histogram("async_op").count == 1


def test_snapshot_structure(reg):
    reg.inc("x")
    reg.observe("y", 1.0)
    snap = reg.snapshot()
    assert "uptime_seconds" in snap
    assert "counters" in snap
    assert "histograms" in snap
    assert "timestamp" in snap
    assert snap["counters"]["x"] == 1


def test_prometheus_text_format(reg):
    reg.inc("req_count")
    reg.observe("req_latency", 0.05)
    text = reg.prometheus_text()
    assert "jarvis_req_count_total 1" in text
    assert "jarvis_req_latency_count 1" in text


def test_log_snapshot_creates_file(reg, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs")
    reg.inc("logged")
    reg.log_snapshot()
    log_file = tmp_path / "logs" / "metrics.jsonl"
    assert log_file.exists()
    import json
    data = json.loads(log_file.read_text().strip())
    assert data["counters"]["logged"] == 1


# ── Additional edge-case coverage ────────────────────────────────────────────

def test_histogram_p99(reg):
    for i in range(100):
        reg.observe("resp2", float(i))
    assert reg.histogram("resp2").p99 >= 95.0


def test_histogram_p95_empty():
    from jarvis.observability import Histogram
    h = Histogram(name="empty")
    assert h.p95 == 0.0


def test_histogram_p99_empty():
    from jarvis.observability import Histogram
    h = Histogram(name="empty")
    assert h.p99 == 0.0


def test_histogram_avg_empty():
    from jarvis.observability import Histogram
    h = Histogram(name="empty")
    assert h.avg == 0.0


def test_histogram_total_and_count():
    from jarvis.observability import Histogram
    h = Histogram(name="t")
    h.observe(1.0)
    h.observe(2.0)
    h.observe(3.0)
    assert h.total == 6.0
    assert h.count == 3


def test_histogram_trims_to_max_size():
    from jarvis.observability import Histogram
    h = Histogram(name="big", _max_size=10)
    for i in range(20):
        h.observe(float(i))
    assert h.count == 10
    # Should keep the newest 10
    assert h.observations[0] == 10.0


def test_counter_auto_creates_on_first_access(reg):
    # Access without calling inc first
    c1 = reg.counter("new_one")
    c2 = reg.counter("new_one")
    assert c1 is c2
    assert c1.value == 0


def test_histogram_auto_creates_on_first_access(reg):
    h1 = reg.histogram("lat")
    h2 = reg.histogram("lat")
    assert h1 is h2


def test_prometheus_sanitises_dots_and_dashes(reg):
    reg.inc("tool.calls-made")
    text = reg.prometheus_text()
    assert "jarvis_tool_calls_made_total" in text
    assert "tool.calls-made" not in text


def test_log_snapshot_appends_lines(reg, tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs2")
    reg.log_snapshot()
    reg.log_snapshot()
    import json
    log_file = tmp_path / "logs2" / "metrics.jsonl"
    lines = [l for l in log_file.read_text().strip().splitlines() if l]
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_snapshot_uptime_increases(reg):
    import time
    snap1 = reg.snapshot()
    time.sleep(0.05)
    snap2 = reg.snapshot()
    assert snap2["uptime_seconds"] > snap1["uptime_seconds"]


def test_counter_inc_by_large_amount(reg):
    reg.inc("bulk", 1000)
    assert reg.counter("bulk").value == 1000


# ── Histogram min/max and single-observation statistics ────────────────────────

def test_histogram_single_observation(reg):
    reg.observe("single", 42.0)
    h = reg.histogram("single")
    assert h.count == 1
    assert h.avg == pytest.approx(42.0)
    assert h.p95 == pytest.approx(42.0)
    assert h.p99 == pytest.approx(42.0)
    assert h.total == pytest.approx(42.0)


def test_histogram_uniform_distribution(reg):
    for i in range(1, 11):
        reg.observe("uniform", float(i))
    h = reg.histogram("uniform")
    assert h.count == 10
    assert h.avg == pytest.approx(5.5)
    assert h.total == pytest.approx(55.0)


def test_counter_increment_by_zero(reg):
    reg.inc("zero_inc", 0)
    assert reg.counter("zero_inc").value == 0


def test_counter_large_increment(reg):
    reg.inc("big", 1_000_000)
    assert reg.counter("big").value == 1_000_000


def test_prometheus_text_includes_all_registered_metrics(reg):
    reg.inc("req_count")
    reg.observe("resp_time", 0.1)
    text = reg.prometheus_text()
    assert "req_count" in text
    assert "resp_time" in text


def test_snapshot_histogram_fields_present(reg):
    reg.observe("lat", 0.5)
    snap = reg.snapshot()
    h = snap["histograms"]["lat"]
    assert "count" in h
    assert "avg_ms" in h
    assert "p95_ms" in h
    assert "p99_ms" in h
