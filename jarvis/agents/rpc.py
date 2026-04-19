"""Inter-agent RPC — asyncio queue-based message passing between agents."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RPCMessage:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    method: str = ""
    params: dict = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    is_response: bool = False


class AgentRPC:
    """Zero-dependency RPC over asyncio queues.
    Agents communicate by name through a shared bus.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._handlers: dict[str, dict[str, Callable]] = {}
        self._pending: dict[str, asyncio.Future] = {}

    def register(self, agent_id: str) -> None:
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
            self._handlers[agent_id] = {}

    def on(self, agent_id: str, method: str) -> Callable:
        """Decorator: register a method handler for an agent."""
        def decorator(fn: Callable) -> Callable:
            self._handlers.setdefault(agent_id, {})[method] = fn
            return fn
        return decorator

    async def call(self, target_id: str, method: str, **params) -> Any:
        """Call a remote method on another agent and wait for result."""
        msg = RPCMessage(method=method, params=params)
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg.id] = future

        q = self._queues.get(target_id)
        if not q:
            raise ValueError(f"Agent '{target_id}' not registered")
        await q.put(msg)

        try:
            return await asyncio.wait_for(future, timeout=60)
        except asyncio.TimeoutError:
            raise TimeoutError(f"RPC call to '{target_id}.{method}' timed out")
        finally:
            self._pending.pop(msg.id, None)

    async def process_messages(self, agent_id: str) -> None:
        """Process all pending messages for an agent (run in background)."""
        q = self._queues.get(agent_id)
        if not q:
            return
        while True:
            msg = await q.get()
            if msg.is_response:
                future = self._pending.get(msg.id)
                if future and not future.done():
                    if msg.error:
                        future.set_exception(RuntimeError(msg.error))
                    else:
                        future.set_result(msg.result)
            else:
                handler = self._handlers.get(agent_id, {}).get(msg.method)
                response = RPCMessage(id=msg.id, is_response=True)
                if handler:
                    try:
                        response.result = await handler(**msg.params)
                    except Exception as exc:
                        response.error = str(exc)
                else:
                    response.error = f"No handler for method '{msg.method}'"
                # Reply to caller's queue (not implemented for simplicity — direct future resolution)
            q.task_done()

    def notify(self, agent_id: str, method: str, **params) -> None:
        """Fire-and-forget notification to an agent."""
        msg = RPCMessage(method=method, params=params)
        q = self._queues.get(agent_id)
        if q:
            asyncio.create_task(q.put(msg))


# Global RPC bus
rpc_bus = AgentRPC()
