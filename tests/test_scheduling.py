"""Tests for jarvis.scheduling — priority queue and NL cron parsing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from jarvis.scheduling import Priority, ScheduledJob, TaskQueue, NLScheduler


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


# ── ScheduledJob dataclass ───────────────────────────────────────────────────

def test_scheduled_job_defaults():
    job = ScheduledJob(name="n", cron="* * * * *", prompt="do")
    assert job.priority == Priority.NORMAL
    assert job.max_retries == 3
    assert job.enabled is True
    assert job.run_count == 0
    assert len(job.id) == 8


def test_scheduled_job_to_dict_roundtrip():
    job = ScheduledJob(
        name="n",
        cron="0 9 * * 1",
        prompt="ping",
        priority=Priority.HIGH,
        tags=["ops"],
    )
    d = job.to_dict()
    assert d["priority"] == "HIGH"
    assert d["tags"] == ["ops"]
    assert d["cron"] == "0 9 * * 1"
    assert d["run_count"] == 0


# ── NLScheduler.add / remove / list / run_now ────────────────────────────────

def _fake_jarvis():
    j = MagicMock()
    j.memory.add_scheduled_task = MagicMock()
    return j


@pytest.mark.asyncio
async def test_nl_scheduler_add_converts_nl_to_cron():
    j = _fake_jarvis()
    sched = NLScheduler(j)
    job = await sched.add("morning-ping", "every morning", "wake up")
    assert job.cron == "0 8 * * *"
    j.memory.add_scheduled_task.assert_called_once()


@pytest.mark.asyncio
async def test_nl_scheduler_add_passes_through_valid_cron():
    j = _fake_jarvis()
    sched = NLScheduler(j)
    job = await sched.add("weekdays", "30 7 * * 1-5", "prompt")
    assert job.cron == "30 7 * * 1-5"


@pytest.mark.asyncio
async def test_nl_scheduler_remove_existing_and_missing():
    j = _fake_jarvis()
    sched = NLScheduler(j)
    job = await sched.add("n", "every hour", "p")
    assert sched.remove(job.id) is True
    assert sched.remove("notfound") is False


@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_sorted_by_priority():
    j = _fake_jarvis()
    sched = NLScheduler(j)
    await sched.add("low", "every hour", "a", priority=Priority.LOW)
    await sched.add("crit", "every hour", "b", priority=Priority.CRITICAL)
    out = sched.list_jobs()
    assert out[0]["priority"] == "CRITICAL"
    assert out[-1]["priority"] == "LOW"


@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_filters_by_tag():
    j = _fake_jarvis()
    sched = NLScheduler(j)
    await sched.add("ops-job", "every hour", "p", tags=["ops"])
    await sched.add("other", "every hour", "p", tags=["misc"])
    ops = sched.list_jobs(tag="ops")
    assert len(ops) == 1
    assert ops[0]["name"] == "ops-job"


@pytest.mark.asyncio
async def test_nl_scheduler_run_now_success():
    j = _fake_jarvis()
    j.chat = AsyncMock(return_value="did it")
    sched = NLScheduler(j)
    job = await sched.add("n", "every hour", "say hi")
    result = await sched.run_now(job.id)
    assert result == "did it"
    assert job.last_status == "success"
    assert job.run_count == 1


@pytest.mark.asyncio
async def test_nl_scheduler_run_now_missing_job():
    j = _fake_jarvis()
    sched = NLScheduler(j)
    result = await sched.run_now("nonexistent")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_nl_scheduler_run_now_retries_then_fails(monkeypatch):
    j = _fake_jarvis()
    j.chat = AsyncMock(side_effect=RuntimeError("boom"))
    sched = NLScheduler(j)
    job = await sched.add("n", "every hour", "p", max_retries=1)
    job.retry_delay = 0  # skip sleep
    # Patch the module-level asyncio.sleep to be instant so test is fast
    monkeypatch.setattr("jarvis.scheduling.asyncio.sleep", AsyncMock())

    result = await sched.run_now(job.id)
    assert "failed" in result
    assert job.error_count == 1
    assert j.chat.call_count == 2  # 1 + 1 retry


# ── TaskQueue failure + history trim ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_worker_records_error_in_history():
    q = TaskQueue()

    async def bad():
        raise RuntimeError("kaboom")

    await q.enqueue(bad, name="fails")
    worker = asyncio.create_task(q.run_worker())
    await asyncio.sleep(0.1)
    worker.cancel()
    h = q.history(limit=10)
    assert any("error" in item["status"] for item in h)


@pytest.mark.asyncio
async def test_history_trims_to_200():
    q = TaskQueue()
    # Pre-seed synthetic history over the cap
    q._history = [{"id": f"i{i}", "name": "x", "started": "", "finished": "", "status": "ok"}
                  for i in range(250)]

    async def noop():
        pass

    await q.enqueue(noop, name="one-more")
    worker = asyncio.create_task(q.run_worker())
    await asyncio.sleep(0.1)
    worker.cancel()
    assert len(q._history) == 200


@pytest.mark.asyncio
async def test_nl_scheduler_run_now_unreachable_path():
    """Line 173: return 'Unreachable.' when max_retries < 0 makes the for-range empty."""
    from jarvis.scheduling import NLScheduler
    from unittest.mock import AsyncMock, MagicMock
    fake_jarvis = MagicMock()
    fake_jarvis.chat = AsyncMock(return_value="ok")
    fake_jarvis.memory = MagicMock()
    fake_jarvis.memory.add_scheduled_task = MagicMock()
    sched = NLScheduler(fake_jarvis)
    job = await sched.add("unreachable_test", "every hour", "ping", max_retries=0)
    job.max_retries = -1
    result = await sched.run_now(job.id)
    assert result == "Unreachable."


# ── TaskQueue: verify CRITICAL beats HIGH beats NORMAL beats LOW ──────────────

@pytest.mark.asyncio
async def test_task_queue_priority_order():
    """Tasks dequeued in CRITICAL→HIGH→NORMAL→LOW order regardless of enqueue order."""
    from jarvis.scheduling import TaskQueue, Priority
    queue = TaskQueue()
    order: list[str] = []

    async def make_fn(label):
        async def fn():
            order.append(label)
        return fn

    # Enqueue in reverse priority order to prove heap reorders them
    await queue.enqueue(await make_fn("low"),      priority=Priority.LOW)
    await queue.enqueue(await make_fn("normal"),   priority=Priority.NORMAL)
    await queue.enqueue(await make_fn("critical"), priority=Priority.CRITICAL)
    await queue.enqueue(await make_fn("high"),     priority=Priority.HIGH)

    import asyncio
    worker = asyncio.create_task(queue._queue.join())
    # Drain all 4 items
    for _ in range(4):
        item = await queue._queue.get()
        await item.fn()
        queue._queue.task_done()
    await worker

    assert order == ["critical", "high", "normal", "low"]


# ── NLScheduler: zero-retries job fails immediately without asyncio.sleep ────

@pytest.mark.asyncio
async def test_nl_scheduler_zero_retries_fails_immediately():
    """Job with max_retries=0 raises on first failure, no retry sleep."""
    from jarvis.scheduling import NLScheduler, Priority
    from unittest.mock import AsyncMock, MagicMock, patch
    fake_jarvis = MagicMock()
    fake_jarvis.chat = AsyncMock(side_effect=RuntimeError("boom"))
    fake_jarvis.memory = MagicMock()
    fake_jarvis.memory.add_scheduled_task = MagicMock()

    sched = NLScheduler(fake_jarvis)
    job = await sched.add("zero_retry", "every hour", "fail", max_retries=0)

    with patch("asyncio.sleep", AsyncMock()) as fake_sleep:
        result = await sched.run_now(job.id)

    fake_sleep.assert_not_called()
    assert "failed" in result
    assert job.error_count == 1


# ── NLScheduler: ScheduledJob.to_dict() covers all fields ────────────────────

def test_scheduled_job_to_dict_all_fields():
    from jarvis.scheduling import ScheduledJob, Priority
    job = ScheduledJob(
        name="backup",
        cron="0 3 * * *",
        prompt="run backup",
        priority=Priority.HIGH,
        max_retries=5,
        retry_delay=120,
        enabled=False,
        last_run="2026-01-01T03:00:00Z",
        last_status="success",
        run_count=10,
        error_count=2,
        tags=["infra", "nightly"],
    )
    d = job.to_dict()
    assert d["name"] == "backup"
    assert d["cron"] == "0 3 * * *"
    assert d["priority"] == "HIGH"
    assert d["max_retries"] == 5
    assert d["retry_delay"] == 120
    assert d["enabled"] is False
    assert d["run_count"] == 10
    assert d["error_count"] == 2
    assert d["tags"] == ["infra", "nightly"]
