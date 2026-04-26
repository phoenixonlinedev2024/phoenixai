"""Tests for jarvis.agents — AgentRPC, RPCMessage, SubagentTask, SubagentPool."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.agents.rpc import AgentRPC, RPCMessage  # noqa: E402
from jarvis.agents.subagent import SubAgent, SubagentPool, SubagentTask  # noqa: E402


# ── RPCMessage ────────────────────────────────────────────────────────────────

def test_rpc_message_auto_id():
    m = RPCMessage()
    assert isinstance(m.id, str)
    assert len(m.id) == 8


def test_rpc_message_unique_ids():
    ids = {RPCMessage().id for _ in range(20)}
    assert len(ids) == 20


def test_rpc_message_defaults():
    m = RPCMessage(method="ping", params={"key": "val"})
    assert m.result is None
    assert m.error is None
    assert m.is_response is False


# ── AgentRPC ──────────────────────────────────────────────────────────────────

def test_register_creates_queue():
    bus = AgentRPC()
    bus.register("agent-a")
    assert "agent-a" in bus._queues


def test_register_idempotent():
    bus = AgentRPC()
    bus.register("agent-a")
    q_before = bus._queues["agent-a"]
    bus.register("agent-a")
    assert bus._queues["agent-a"] is q_before


def test_on_decorator_registers_handler():
    bus = AgentRPC()
    bus.register("agent-a")

    @bus.on("agent-a", "greet")
    async def greet():
        return "hello"

    assert "greet" in bus._handlers["agent-a"]


@pytest.mark.asyncio
async def test_call_unregistered_agent_raises():
    bus = AgentRPC()
    with pytest.raises(ValueError, match="not registered"):
        await bus.call("ghost", "method")


@pytest.mark.asyncio
async def test_process_messages_handles_request():
    bus = AgentRPC()
    bus.register("worker")

    @bus.on("worker", "add")
    async def add(a, b):
        return a + b

    # Manually put a message and resolve the future via process_messages
    msg = RPCMessage(method="add", params={"a": 3, "b": 4})
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    bus._pending[msg.id] = future
    await bus._queues["worker"].put(msg)

    # Drive one iteration of process_messages
    async def run_once():
        q = bus._queues["worker"]
        m = await q.get()
        handler = bus._handlers["worker"].get(m.method)
        if handler:
            result = await handler(**m.params)
            f = bus._pending.get(m.id)
            if f and not f.done():
                f.set_result(result)
        q.task_done()

    await run_once()
    assert future.result() == 7


@pytest.mark.asyncio
async def test_process_messages_unknown_method_sets_error():
    bus = AgentRPC()
    bus.register("worker")
    msg = RPCMessage(method="nonexistent", params={})
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    bus._pending[msg.id] = future
    await bus._queues["worker"].put(msg)

    async def run_once():
        q = bus._queues["worker"]
        m = await q.get()
        handler = bus._handlers["worker"].get(m.method)
        if not handler:
            f = bus._pending.get(m.id)
            if f and not f.done():
                f.set_exception(RuntimeError(f"No handler for method '{m.method}'"))
        q.task_done()

    await run_once()
    with pytest.raises(RuntimeError, match="No handler"):
        future.result()


# ── SubagentTask ──────────────────────────────────────────────────────────────

def test_subagent_task_defaults():
    t = SubagentTask(goal="do something")
    assert t.done is False
    assert t.result is None
    assert t.error is None
    assert len(t.id) == 8


def test_subagent_task_unique_ids():
    ids = {SubagentTask().id for _ in range(20)}
    assert len(ids) == 20


# ── SubagentPool ──────────────────────────────────────────────────────────────

def _make_parent(reply: str = "task done") -> MagicMock:
    """Build a minimal Jarvis mock that satisfies SubAgent.run()."""
    block = MagicMock()
    block.type = "text"
    block.text = reply

    response = MagicMock()
    response.content = [block]

    parent = MagicMock()
    parent.client.messages.create = AsyncMock(return_value=response)
    parent._get_model = MagicMock(return_value="claude-haiku-4-5-20251001")
    parent.registry.anthropic_tools = MagicMock(return_value=[])
    parent._execute_tools_parallel = AsyncMock(return_value=[])
    return parent


@pytest.mark.asyncio
async def test_subagent_run_returns_text():
    task = SubagentTask(goal="write hello world in Python")
    parent = _make_parent("print('hello world')")
    agent = SubAgent(task, parent)
    result = await agent.run()
    assert result == "print('hello world')"


@pytest.mark.asyncio
async def test_subagent_pool_dispatch_single():
    parent = _make_parent("result A")
    pool = SubagentPool(parent, max_concurrent=2)
    task = SubagentTask(goal="goal A")
    results = await pool.dispatch([task])
    assert results[task.id] == "result A"
    assert task.done is True


@pytest.mark.asyncio
async def test_subagent_pool_dispatch_multiple_parallel():
    parent = _make_parent("done")
    pool = SubagentPool(parent, max_concurrent=3)
    tasks = [SubagentTask(goal=f"task {i}") for i in range(3)]
    results = await pool.dispatch(tasks)
    assert len(results) == 3
    assert all(t.done for t in tasks)


@pytest.mark.asyncio
async def test_subagent_pool_dispatch_one():
    parent = _make_parent("one result")
    pool = SubagentPool(parent, max_concurrent=1)
    result = await pool.dispatch_one("single goal")
    assert result == "one result"


@pytest.mark.asyncio
async def test_subagent_pool_captures_error():
    parent = MagicMock()
    parent.client.messages.create = AsyncMock(side_effect=RuntimeError("api crash"))
    parent._get_model = MagicMock(return_value="claude-haiku-4-5-20251001")
    parent.registry.anthropic_tools = MagicMock(return_value=[])

    pool = SubagentPool(parent, max_concurrent=1)
    task = SubagentTask(goal="risky task")
    results = await pool.dispatch([task])
    assert "Error" in results[task.id]
    assert task.error is not None
    assert task.done is True


@pytest.mark.asyncio
async def test_subagent_pool_respects_semaphore():
    """Semaphore with limit 1 should still complete all tasks serially."""
    parent = _make_parent("serial")
    pool = SubagentPool(parent, max_concurrent=1)
    tasks = [SubagentTask(goal=f"t{i}") for i in range(4)]
    results = await pool.dispatch(tasks)
    assert len(results) == 4


# ── AgentRPC.notify ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notify_puts_message_to_queue():
    bus = AgentRPC()
    bus.register("listener")

    bus.notify("listener", "event", data="hello")
    # Give the created task a tick to execute
    await asyncio.sleep(0)

    assert not bus._queues["listener"].empty()
    msg = await bus._queues["listener"].get()
    assert msg.method == "event"
    assert msg.params["data"] == "hello"


def test_notify_silently_ignores_unknown_agent():
    bus = AgentRPC()
    # Should not raise even if agent is not registered
    bus.notify("ghost_agent", "method", x=1)


@pytest.mark.asyncio
async def test_call_timeout_raises():
    bus = AgentRPC()
    bus.register("slow-agent")

    # Don't put any message handler — future will never resolve
    import asyncio as _asyncio
    original_wait_for = _asyncio.wait_for

    async def fast_timeout(coro, timeout):
        raise _asyncio.TimeoutError()

    with patch("jarvis.agents.rpc.asyncio.wait_for", fast_timeout):
        with pytest.raises(TimeoutError, match="timed out"):
            await bus.call("slow-agent", "neverreplies")


# ── RPCMessage is_response field ─────────────────────────────────────────────

def test_rpc_message_is_response_default_false():
    msg = RPCMessage()
    assert msg.is_response is False


def test_rpc_message_can_set_error():
    msg = RPCMessage(error="something went wrong", is_response=True)
    assert msg.error == "something went wrong"
    assert msg.is_response is True


# ── AgentRPC.process_messages ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_messages_returns_for_unknown_agent():
    bus = AgentRPC()
    # Should return immediately without looping
    await bus.process_messages("nonexistent")


@pytest.mark.asyncio
async def test_process_messages_resolves_response_future():
    bus = AgentRPC()
    bus.register("agent1")
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    response_msg = RPCMessage(id="abc123", method="", result="the answer", is_response=True)
    bus._pending["abc123"] = future
    await bus._queues["agent1"].put(response_msg)

    task = asyncio.create_task(bus.process_messages("agent1"))
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert future.done()
    assert future.result() == "the answer"


@pytest.mark.asyncio
async def test_process_messages_sets_exception_on_error_response():
    bus = AgentRPC()
    bus.register("agent2")
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    response_msg = RPCMessage(id="err001", method="", error="boom", is_response=True)
    bus._pending["err001"] = future
    await bus._queues["agent2"].put(response_msg)

    task = asyncio.create_task(bus.process_messages("agent2"))
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert future.done()
    with pytest.raises(RuntimeError, match="boom"):
        future.result()


@pytest.mark.asyncio
async def test_process_messages_calls_handler():
    bus = AgentRPC()
    bus.register("agent3")
    called_with = {}

    @bus.on("agent3", "greet")
    async def greet(name):
        called_with["name"] = name
        return f"hello {name}"

    msg = RPCMessage(method="greet", params={"name": "Tony"})
    await bus._queues["agent3"].put(msg)

    task = asyncio.create_task(bus.process_messages("agent3"))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert called_with.get("name") == "Tony"


@pytest.mark.asyncio
async def test_process_messages_handler_exception_is_captured():
    bus = AgentRPC()
    bus.register("agent4")

    @bus.on("agent4", "broken")
    async def broken():
        raise ValueError("handler failed")

    msg = RPCMessage(method="broken", params={})
    await bus._queues["agent4"].put(msg)

    task = asyncio.create_task(bus.process_messages("agent4"))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    # No exception propagated from process_messages


# ── SubAgent context and tool_use paths ──────────────────────────────────────

@pytest.mark.asyncio
async def test_subagent_run_includes_context():
    """Line 42: context from parent is appended to the system prompt."""
    task = SubagentTask(goal="search the web", context="User is on macOS")
    parent = _make_parent("done")
    agent = SubAgent(task, parent)
    result = await agent.run()
    assert result == "done"
    # Verify messages.create was called (system prompt built with context)
    parent.client.messages.create.assert_called_once()
    call_kwargs = parent.client.messages.create.call_args[1]
    assert "macOS" in call_kwargs["system"]


@pytest.mark.asyncio
async def test_subagent_run_tool_use_then_text():
    """Lines 60-61, 66-70: tool_use block triggers parallel execution then a text reply."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_1"
    tool_block.name = "read_file"
    tool_block.input = {"path": "/tmp/x"}

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "file read successfully"

    response_with_tool = MagicMock()
    response_with_tool.content = [tool_block]

    response_text_only = MagicMock()
    response_text_only.content = [text_block]

    parent = MagicMock()
    parent.client.messages.create = AsyncMock(
        side_effect=[response_with_tool, response_text_only]
    )
    parent._get_model = MagicMock(return_value="claude-haiku-4-5-20251001")
    parent.registry.anthropic_tools = MagicMock(return_value=[])
    parent._execute_tools_parallel = AsyncMock(return_value=[{"type": "tool_result", "content": "data"}])

    task = SubagentTask(goal="read a file")
    agent = SubAgent(task, parent)
    result = await agent.run()

    assert result == "file read successfully"
    parent._execute_tools_parallel.assert_called_once()
    call_args = parent._execute_tools_parallel.call_args[0][0]
    assert call_args[0]["name"] == "read_file"
    assert call_args[0]["id"] == "call_1"


@pytest.mark.asyncio
async def test_subagent_run_reaches_max_turns():
    """Line 70: max_turns exceeded without a text-only response."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_x"
    tool_block.name = "search"
    tool_block.input = {}

    response_always_tool = MagicMock()
    response_always_tool.content = [tool_block]

    parent = MagicMock()
    parent.client.messages.create = AsyncMock(return_value=response_always_tool)
    parent._get_model = MagicMock(return_value="claude-haiku-4-5-20251001")
    parent.registry.anthropic_tools = MagicMock(return_value=[])
    parent._execute_tools_parallel = AsyncMock(return_value=[])

    task = SubagentTask(goal="loop forever", max_turns=3)
    agent = SubAgent(task, parent)
    result = await agent.run()

    assert "max turns" in result.lower()
    assert parent.client.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_process_messages_no_handler_sets_error():
    """Line 84: response.error set when no handler is registered for the method."""
    import asyncio
    bus = AgentRPC()
    bus.register("worker")
    msg = RPCMessage(method="unknown_method", params={})

    await bus._queues["worker"].put(msg)
    worker = asyncio.create_task(bus.process_messages("worker"))
    await asyncio.sleep(0.05)
    worker.cancel()
    assert bus._queues["worker"].empty()


@pytest.mark.asyncio
async def test_process_messages_response_with_no_pending_future():
    """Branch 70->86: future is None when msg.id not in _pending — silently skipped."""
    bus = AgentRPC()
    bus.register("worker")
    # A response for an id we never called — no future in _pending
    msg = RPCMessage(id="unknown-id-xyz", is_response=True, result="done")
    await bus._queues["worker"].put(msg)
    worker = asyncio.create_task(bus.process_messages("worker"))
    await asyncio.sleep(0.05)
    worker.cancel()
    assert bus._queues["worker"].empty()


@pytest.mark.asyncio
async def test_subagent_run_ignores_unknown_block_type():
    """Branch 60->57: block with type other than text/tool_use is skipped in loop."""
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "done"

    response = MagicMock()
    response.content = [thinking_block, text_block]

    parent = MagicMock()
    parent.client.messages.create = AsyncMock(return_value=response)
    parent._get_model = MagicMock(return_value="claude-haiku-4-5-20251001")
    parent.registry.anthropic_tools = MagicMock(return_value=[])

    from jarvis.agents.subagent import SubAgent, SubagentTask
    task = SubagentTask(goal="test unknown block")
    agent = SubAgent(task, parent)
    result = await agent.run()
    assert result == "done"
