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
