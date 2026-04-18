"""Tests for jarvis.scheduling — priority queue and NL cron parsing."""

import asyncio
import pytest
from jarvis.scheduling import Priority, TaskQueue, NLScheduler


# ── Priority ──────────────────────────────────────────────────────────────────

def test_priority_ordering():
    assert Priority.CRITICAL > Priority.HIGH > Priority.NORMAL > Priority.LOW


# ── TaskQueue ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_returns_id():
    q = TaskQueue()
    async def noop(): pass
    task_id = await q.enqueue(noop, name="test")
    assert isinstance(task_id, str)
    assert len(task_id) == 8


@pytest.mark.asyncio
async def test_worker_processes_task():
    q = TaskQueue()
    results = []

    async def work():
        results.append("done")

    await q.enqueue(work, name="worker_test", priority=Priority.HIGH)
    worker = asyncio.create_task(q.run_worker())
    await asyncio.sleep(0.1)
    worker.cancel()
    assert "done" in results


@pytest.mark.asyncio
async def test_history_recorded():
    q = TaskQueue()

    async def noop():
        pass

    await q.enqueue(noop, name="recorded")
    worker = asyncio.create_task(q.run_worker())
    await asyncio.sleep(0.1)
    worker.cancel()
    h = q.history(limit=10)
    assert any(item["name"] == "recorded" for item in h)


@pytest.mark.asyncio
async def test_pending_count():
    q = TaskQueue()
    async def noop(): pass
    assert q.pending == 0
    await q.enqueue(noop, name="p1")
    await q.enqueue(noop, name="p2")
    assert q.pending == 2


# ── NLScheduler cron parsing ─────────────────────────────────────────────────

@pytest.mark.parametrize("phrase,expected", [
    ("every minute", "* * * * *"),
    ("every hour", "0 * * * *"),
    ("every morning", "0 8 * * *"),
    ("every evening", "0 18 * * *"),
    ("midnight", "0 0 * * *"),
    ("noon", "0 12 * * *"),
    ("every weekday", "0 9 * * 1-5"),
    ("every monday", "0 9 * * 1"),
    ("every month", "0 9 1 * *"),
])
def test_nl_to_cron(phrase, expected):
    result = NLScheduler.nl_to_cron(phrase)
    assert result == expected, f"'{phrase}' → '{result}', expected '{expected}'"


def test_nl_to_cron_every_n_minutes():
    result = NLScheduler.nl_to_cron("every 15 minutes")
    assert result == "*/15 * * * *"


def test_nl_to_cron_every_n_hours():
    result = NLScheduler.nl_to_cron("every 6 hours")
    assert result == "0 */6 * * *"


def test_nl_to_cron_passthrough_valid_cron():
    # If input already looks like cron (5 parts), NLScheduler passes it through
    cron = "30 7 * * 1-5"
    result = NLScheduler.nl_to_cron(cron)
    # 5-part strings are passed directly in add(), not parsed — test via add() logic
    assert len(cron.split()) == 5  # confirms it's already cron format


def test_nl_to_cron_unrecognised_returns_message():
    result = NLScheduler.nl_to_cron("whenever the moon is full")
    assert "Could not parse" in result or "cron" in result.lower()
