"""Tests for jarvis.acp — asyncio pub/sub message bus."""

import asyncio
import pytest
from jarvis.acp import MessageBus, ACPMessage


@pytest.fixture
def bus():
    return MessageBus()


# ── Publish / subscribe ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subscriber_receives_message(bus):
    received = []

    async def handler(msg: ACPMessage):
        received.append(msg)

    bus.subscribe("test.topic", handler)
    await bus.publish("test.topic", payload="hello")
    assert len(received) == 1
    assert received[0].payload == "hello"
    assert received[0].topic == "test.topic"


@pytest.mark.asyncio
async def test_wildcard_subscriber(bus):
    received = []

    async def catch_all(msg: ACPMessage):
        received.append(msg)

    bus.subscribe("*", catch_all)
    await bus.publish("anything", payload=42)
    await bus.publish("something.else", payload="yo")
    assert len(received) == 2


@pytest.mark.asyncio
async def test_multiple_subscribers_same_topic(bus):
    counts = [0, 0]

    async def h1(msg):
        counts[0] += 1

    async def h2(msg):
        counts[1] += 1

    bus.subscribe("multi", h1)
    bus.subscribe("multi", h2)
    await bus.publish("multi")
    assert counts == [1, 1]


@pytest.mark.asyncio
async def test_unsubscribe(bus):
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("ev", handler)
    await bus.publish("ev")
    bus.unsubscribe("ev", handler)
    await bus.publish("ev")
    assert len(received) == 1


@pytest.mark.asyncio
async def test_different_topics_isolated(bus):
    a_msgs, b_msgs = [], []

    async def ha(msg):
        a_msgs.append(msg)

    async def hb(msg):
        b_msgs.append(msg)

    bus.subscribe("topic.a", ha)
    bus.subscribe("topic.b", hb)
    await bus.publish("topic.a", payload="for_a")
    assert len(a_msgs) == 1
    assert len(b_msgs) == 0


# ── History ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_records_messages(bus):
    await bus.publish("hist", payload="one")
    await bus.publish("hist", payload="two")
    h = bus.history(topic="hist")
    assert len(h) == 2
    payloads = [m["payload"] for m in h]
    assert "one" in payloads and "two" in payloads


@pytest.mark.asyncio
async def test_history_topic_filter(bus):
    await bus.publish("topic.x", payload="x")
    await bus.publish("topic.y", payload="y")
    h = bus.history(topic="topic.x")
    assert all(m["topic"] == "topic.x" for m in h)


@pytest.mark.asyncio
async def test_history_limit(bus):
    for i in range(10):
        await bus.publish("flood", payload=i)
    h = bus.history(limit=3)
    assert len(h) == 3


# ── Stats ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats(bus):
    async def noop(msg): pass
    bus.subscribe("s1", noop)
    bus.subscribe("s2", noop)
    await bus.publish("s1")
    s = bus.stats()
    assert s["topics"] >= 2
    assert s["history_size"] >= 1


# ── Message structure ─────────────────────────────────────────────────────────

def test_message_to_dict():
    msg = ACPMessage(topic="t", payload={"k": "v"}, sender="agent-1")
    d = msg.to_dict()
    assert d["topic"] == "t"
    assert d["payload"] == {"k": "v"}
    assert d["sender"] == "agent-1"
    assert "timestamp" in d
    assert "id" in d


# ── Additional coverage ───────────────────────────────────────────────────────

def test_message_unique_ids():
    ids = {ACPMessage().id for _ in range(20)}
    assert len(ids) == 20


@pytest.mark.asyncio
async def test_history_trims_to_max(bus):
    # Fill past the 500-message limit
    bus._max_history = 10
    for i in range(15):
        await bus.publish("trim", payload=i)
    assert len(bus._history) == 10
    # Most recent messages should be retained
    payloads = [m.payload for m in bus._history]
    assert 14 in payloads
    assert 0 not in payloads


@pytest.mark.asyncio
async def test_handler_exception_does_not_crash_bus(bus):
    async def bad_handler(msg):
        raise RuntimeError("handler exploded")

    received = []

    async def good_handler(msg):
        received.append(msg)

    bus.subscribe("err_topic", bad_handler)
    bus.subscribe("err_topic", good_handler)
    # publish uses return_exceptions=True so should not raise
    await bus.publish("err_topic", payload="test")
    assert len(received) == 1


@pytest.mark.asyncio
async def test_publish_custom_sender(bus):
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("s", handler)
    await bus.publish("s", payload="x", sender="agent-007")
    assert received[0].sender == "agent-007"


def test_topics_lists_registered(bus):
    async def noop(msg): pass
    bus.subscribe("alpha", noop)
    bus.subscribe("beta", noop)
    assert "alpha" in bus.topics()
    assert "beta" in bus.topics()


@pytest.mark.asyncio
async def test_publish_sync_enqueues_in_running_loop(bus):
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("sync_topic", handler)

    async def _drive():
        bus.publish_sync("sync_topic", payload="fire")
        await asyncio.sleep(0.05)  # let the created task execute

    await _drive()
    assert len(received) == 1
    assert received[0].payload == "fire"


def test_publish_sync_no_running_loop_is_silent(bus):
    """Lines 63-64: publish_sync swallows RuntimeError when no event loop is running."""
    from unittest.mock import patch
    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
        bus.publish_sync("no_loop_topic", payload="ignored")
    # No exception raised — test passes if we reach here
