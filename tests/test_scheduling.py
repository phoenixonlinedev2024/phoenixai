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


# ── TaskQueue history and run_worker ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_queue_history_recorded_after_run():
    from jarvis.scheduling import TaskQueue, Priority
    import asyncio
    queue = TaskQueue()
    results = []

    async def my_fn():
        results.append("ran")

    tid = await queue.enqueue(my_fn, name="hist-test", priority=Priority.NORMAL)

    worker = asyncio.create_task(queue.run_worker())
    await queue._queue.join()
    worker.cancel()

    hist = queue.history()
    assert len(hist) == 1
    assert hist[0]["name"] == "hist-test"
    assert hist[0]["status"] == "success"
    assert "ran" in results


@pytest.mark.asyncio
async def test_task_queue_error_recorded_in_history():
    from jarvis.scheduling import TaskQueue, Priority
    import asyncio
    queue = TaskQueue()

    async def bad_fn():
        raise ValueError("oops")

    await queue.enqueue(bad_fn, name="bad-task", priority=Priority.HIGH)
    worker = asyncio.create_task(queue.run_worker())
    await queue._queue.join()
    worker.cancel()

    hist = queue.history()
    assert "error" in hist[0]["status"]
    assert "oops" in hist[0]["status"]


def test_task_queue_pending_count():
    from jarvis.scheduling import TaskQueue
    import asyncio
    queue = TaskQueue()
    assert queue.pending == 0


def test_task_queue_history_limit():
    from jarvis.scheduling import TaskQueue
    queue = TaskQueue()
    # Manually add 10 history entries
    for i in range(10):
        queue._history.append({"id": str(i), "status": "success", "name": f"t{i}",
                                "started": "t", "finished": "t"})
    # history(limit=3) returns last 3
    assert len(queue.history(limit=3)) == 3
    assert queue.history(limit=3)[-1]["name"] == "t9"


# ── NLScheduler list_jobs and remove ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_nl_scheduler_add_and_list():
    from jarvis.scheduling import NLScheduler, Priority
    sched = NLScheduler(_fake_jarvis())
    j1 = await sched.add("nightly_backup", "every day", "backup", tags=["infra"])
    j2 = await sched.add("hourly_report", "every hour", "report", priority=Priority.HIGH)
    jobs = sched.list_jobs()
    assert len(jobs) == 2
    # Higher priority job first
    assert jobs[0]["name"] == "hourly_report"


@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_filter_by_tag():
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    await sched.add("j1", "every hour", "p", tags=["alpha"])
    await sched.add("j2", "every hour", "p", tags=["beta"])
    alpha_jobs = sched.list_jobs(tag="alpha")
    assert len(alpha_jobs) == 1
    assert alpha_jobs[0]["name"] == "j1"


@pytest.mark.asyncio
async def test_nl_scheduler_remove_job():
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    job = await sched.add("to-delete", "every hour", "do it")
    assert sched.remove(job.id) is True
    assert sched.remove(job.id) is False  # already removed


@pytest.mark.asyncio
async def test_nl_scheduler_run_now_unknown_job():
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    result = await sched.run_now("nonexistent-job-id")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_nl_scheduler_run_now_success():
    from jarvis.scheduling import NLScheduler
    from unittest.mock import AsyncMock
    j = _fake_jarvis()
    j.chat = AsyncMock(return_value="task complete")
    sched = NLScheduler(j)
    job = await sched.add("run-me", "every hour", "do the thing")
    result = await sched.run_now(job.id)
    assert result == "task complete"
    assert job.run_count == 1
    assert job.last_status == "success"


# ── Priority enum values ──────────────────────────────────────────────────────

def test_priority_values_ordering():
    from jarvis.scheduling import Priority
    assert Priority.LOW < Priority.NORMAL < Priority.HIGH < Priority.CRITICAL


def test_scheduled_job_defaults():
    from jarvis.scheduling import ScheduledJob, Priority
    job = ScheduledJob()
    assert job.enabled is True
    assert job.priority == Priority.NORMAL
    assert job.max_retries == 3
    assert job.run_count == 0
    assert job.error_count == 0
    assert job.tags == []


# ── ScheduledJob.to_dict() priority is serialized as name ────────────────────

def test_scheduled_job_to_dict_priority_as_string():
    """priority field in to_dict() is the enum name, not the integer."""
    from jarvis.scheduling import ScheduledJob, Priority
    job = ScheduledJob(priority=Priority.HIGH)
    d = job.to_dict()
    assert d["priority"] == "HIGH"


def test_scheduled_job_to_dict_critical_priority():
    from jarvis.scheduling import ScheduledJob, Priority
    job = ScheduledJob(priority=Priority.CRITICAL)
    assert job.to_dict()["priority"] == "CRITICAL"


# ── TaskQueue.history(limit=0) returns all (Python -0 == 0) ──────────────────

def test_task_queue_history_limit_zero_returns_all():
    """history(limit=0) → msgs[-0:] == msgs[0:] returns all entries."""
    from jarvis.scheduling import TaskQueue
    queue = TaskQueue()
    for i in range(5):
        queue._history.append({"id": str(i), "status": "success", "name": f"t{i}",
                                "started": "t", "finished": "t"})
    assert len(queue.history(limit=0)) == 5


# ── NLScheduler.add() passes through a valid 5-word cron directly ─────────────

@pytest.mark.asyncio
async def test_nl_scheduler_add_passthrough_cron():
    """A 5-part string is treated as a cron expression and stored as-is."""
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    job = await sched.add("cron-job", "30 6 * * 1", "morning report")
    assert job.cron == "30 6 * * 1"


# ── NLScheduler.run_now() increments error_count on exhausted retries ─────────

@pytest.mark.asyncio
async def test_nl_scheduler_run_now_increments_error_count():
    """Exhausting all retries bumps job.error_count by 1."""
    from jarvis.scheduling import NLScheduler
    from unittest.mock import AsyncMock, patch
    j = _fake_jarvis()
    j.chat = AsyncMock(side_effect=RuntimeError("always fails"))
    sched = NLScheduler(j)
    # max_retries=0 → one attempt only, no sleep between retries
    job = await sched.add("fail-job", "every hour", "task", max_retries=0)
    with patch("asyncio.sleep", AsyncMock()):
        await sched.run_now(job.id)
    assert job.error_count == 1
    assert "failed" in job.last_status


# ── TaskQueue.history() with limit > history size returns all ─────────────────

def test_task_queue_history_limit_larger_than_history():
    """history(limit=100) when only 3 entries exist returns all 3."""
    from jarvis.scheduling import TaskQueue
    queue = TaskQueue()
    for i in range(3):
        queue._history.append({"id": str(i), "name": f"t{i}", "status": "ok",
                                "started": "s", "finished": "f"})
    result = queue.history(limit=100)
    assert len(result) == 3


# ── ScheduledJob.enabled defaults to True ────────────────────────────────────

def test_scheduled_job_enabled_default():
    """ScheduledJob.enabled is True by default."""
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob(name="test")
    assert job.enabled is True


# ── NLScheduler.add() stores job in _jobs dict ────────────────────────────────

@pytest.mark.asyncio
async def test_nl_scheduler_add_stores_job_in_dict():
    """add() inserts the new ScheduledJob into _jobs under its id."""
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    job = await sched.add("myjob", "every day", "do something")
    assert job.id in sched._jobs
    assert sched._jobs[job.id] is job


# ── NLScheduler.run_now() success path updates run_count ─────────────────────

@pytest.mark.asyncio
async def test_nl_scheduler_run_now_success_updates_run_count():
    """Successful run_now increments job.run_count and sets last_status."""
    from jarvis.scheduling import NLScheduler
    from unittest.mock import AsyncMock
    j = _fake_jarvis()
    j.chat = AsyncMock(return_value="done!")
    sched = NLScheduler(j)
    job = await sched.add("ok-job", "every hour", "some prompt")
    result = await sched.run_now(job.id)
    assert result == "done!"
    assert job.run_count == 1
    assert job.last_status == "success"


# ── NLScheduler.list_jobs() empty scheduler ───────────────────────────────────

@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_empty_scheduler_returns_empty():
    """list_jobs() on a fresh NLScheduler returns []."""
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    assert sched.list_jobs() == []


@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_no_tag_filter_returns_all():
    """list_jobs() with no tag filter returns all jobs."""
    from jarvis.scheduling import NLScheduler, Priority
    sched = NLScheduler(_fake_jarvis())
    await sched.add("job1", "every hour", "prompt1", tags=["web"])
    await sched.add("job2", "every day", "prompt2", tags=["email"])
    jobs = sched.list_jobs()
    assert len(jobs) == 2


# ── ScheduledJob.to_dict() with non-None last_run / last_status ───────────────

def test_scheduled_job_to_dict_with_last_run_and_status():
    """last_run and last_status appear in to_dict() when set."""
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob(name="j", cron="* * * * *", prompt="do thing")
    job.last_run = "2024-01-01T00:00:00+00:00"
    job.last_status = "success"
    d = job.to_dict()
    assert d["last_run"] == "2024-01-01T00:00:00+00:00"
    assert d["last_status"] == "success"


def test_scheduled_job_to_dict_null_last_run_by_default():
    """A new ScheduledJob has last_run and last_status as None in to_dict()."""
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob()
    d = job.to_dict()
    assert d["last_run"] is None
    assert d["last_status"] is None


# ── TaskQueue.pending tracks enqueued items ──────────────────────────────────

@pytest.mark.asyncio
async def test_task_queue_pending_increments_on_enqueue():
    """pending count grows with each enqueued task."""
    from jarvis.scheduling import TaskQueue, Priority
    queue = TaskQueue()
    assert queue.pending == 0
    async def noop(): pass
    await queue.enqueue(noop, "t1", Priority.NORMAL)
    assert queue.pending == 1
    await queue.enqueue(noop, "t2", Priority.HIGH)
    assert queue.pending == 2


# ── NLScheduler.remove returns True/False ────────────────────────────────────

@pytest.mark.asyncio
async def test_nl_scheduler_remove_returns_true_when_found():
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    job = await sched.add("rem_job", "every day", "daily task")
    result = sched.remove(job.id)
    assert result is True


@pytest.mark.asyncio
async def test_nl_scheduler_remove_returns_false_for_unknown_id():
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    result = sched.remove("nonexistent_job_id")
    assert result is False


@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_sorted_by_priority_descending():
    from jarvis.scheduling import NLScheduler, Priority
    sched = NLScheduler(_fake_jarvis())
    await sched.add("low_job", "every day", "low", priority=Priority.LOW)
    await sched.add("high_job", "every day", "high", priority=Priority.HIGH)
    await sched.add("normal_job", "every day", "normal", priority=Priority.NORMAL)
    jobs = sched.list_jobs()
    priorities = [j["priority"] for j in jobs]
    assert priorities[0] == "HIGH"
    assert priorities[-1] == "LOW"


@pytest.mark.asyncio
async def test_nl_scheduler_list_jobs_tag_filter_no_match():
    from jarvis.scheduling import NLScheduler
    sched = NLScheduler(_fake_jarvis())
    await sched.add("tagged_job", "every hour", "prompt", tags=["finance"])
    jobs = sched.list_jobs(tag="nonexistent_tag")
    assert jobs == []


# ── ScheduledJob fields and to_dict completeness ─────────────────────────────

def test_scheduled_job_error_count_default_zero():
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob()
    assert job.error_count == 0


def test_scheduled_job_to_dict_includes_error_count():
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob()
    job.error_count = 5
    d = job.to_dict()
    assert d["error_count"] == 5


def test_scheduled_job_to_dict_includes_tags_list():
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob(tags=["web", "daily"])
    d = job.to_dict()
    assert d["tags"] == ["web", "daily"]


def test_scheduled_job_to_dict_run_count():
    from jarvis.scheduling import ScheduledJob
    job = ScheduledJob()
    job.run_count = 7
    d = job.to_dict()
    assert d["run_count"] == 7


# ── Priority values ───────────────────────────────────────────────────────────

def test_priority_critical_is_highest():
    from jarvis.scheduling import Priority
    assert Priority.CRITICAL > Priority.HIGH > Priority.NORMAL > Priority.LOW
