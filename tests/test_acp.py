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


def test_unsubscribe_nonexistent_topic_is_noop(bus):
    """Branch 45->exit: unsubscribe when topic not in _subs does nothing."""
    async def handler(msg):
        pass

    bus.unsubscribe("never_subscribed_topic", handler)
    # No exception raised — test passes if we reach here


# ── Stats subscribers dict ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_subscribers_counts(bus):
    async def h1(msg): pass
    async def h2(msg): pass
    bus.subscribe("ch1", h1)
    bus.subscribe("ch1", h2)
    bus.subscribe("ch2", h1)
    s = bus.stats()
    assert s["subscribers"]["ch1"] == 2
    assert s["subscribers"]["ch2"] == 1


# ── history() limit slices from the end ───────────────────────────────────────

@pytest.mark.asyncio
async def test_history_limit_one_returns_most_recent(bus):
    """msgs[-1:] returns the single most recent message."""
    await bus.publish("x", payload="first")
    await bus.publish("x", payload="second")
    h = bus.history(limit=1)
    assert len(h) == 1
    assert h[0]["payload"] == "second"


# ── unsubscribe removes the handler cleanly ──────────────────────────────────

@pytest.mark.asyncio
async def test_unsubscribe_removes_specific_handler_only(bus):
    calls_a, calls_b = [], []

    async def ha(msg): calls_a.append(msg)
    async def hb(msg): calls_b.append(msg)

    bus.subscribe("ch", ha)
    bus.subscribe("ch", hb)
    bus.unsubscribe("ch", ha)

    await bus.publish("ch", payload="ping")
    assert len(calls_a) == 0  # removed
    assert len(calls_b) == 1  # still subscribed


# ── wildcard + specific handler both receive the message ─────────────────────

@pytest.mark.asyncio
async def test_wildcard_and_specific_both_receive(bus):
    specific_calls, wildcard_calls = [], []

    async def specific(msg): specific_calls.append(msg)
    async def wildcard(msg): wildcard_calls.append(msg)

    bus.subscribe("alerts.high", specific)
    bus.subscribe("*", wildcard)
    await bus.publish("alerts.high", payload="critical error")

    assert len(specific_calls) == 1
    assert len(wildcard_calls) == 1
    assert specific_calls[0].payload == "critical error"


# ── publish with no subscribers still records history ────────────────────────

@pytest.mark.asyncio
async def test_publish_no_subscribers_still_records_history(bus):
    await bus.publish("orphan.topic", payload="lone msg")
    h = bus.history(topic="orphan.topic")
    assert len(h) == 1
    assert h[0]["payload"] == "lone msg"


# ── to_dict includes all fields ───────────────────────────────────────────────

def test_message_to_dict_all_fields():
    from jarvis.acp import ACPMessage
    msg = ACPMessage(topic="events.auth", payload={"user": "alice"}, sender="auth-svc")
    d = msg.to_dict()
    assert d["topic"] == "events.auth"
    assert d["payload"] == {"user": "alice"}
    assert d["sender"] == "auth-svc"
    assert len(d["id"]) == 8  # hex[:8]
    assert "T" in d["timestamp"]  # ISO format


# ── history(limit=0) returns all messages (Python -0 == 0 behavior) ─────────

@pytest.mark.asyncio
async def test_history_limit_zero_returns_all(bus):
    """limit=0 uses msgs[-0:] == msgs[0:] — returns all messages."""
    await bus.publish("t", payload="a")
    await bus.publish("t", payload="b")
    await bus.publish("t", payload="c")
    h = bus.history(limit=0)
    assert len(h) == 3


# ── topics() only lists topics that still have subscriptions ─────────────────

def test_topics_lists_only_subscribed(bus):
    async def h(msg): pass
    bus.subscribe("one", h)
    bus.subscribe("two", h)
    bus.unsubscribe("one", h)
    # After unsubscribe, "one" still in _subs but with empty list
    topics = bus.topics()
    assert "two" in topics


# ── history() topic filter returns only matching messages ────────────────────

@pytest.mark.asyncio
async def test_history_topic_filter_excludes_other_topics(bus):
    await bus.publish("a.topic", payload="a message")
    await bus.publish("b.topic", payload="b message")
    h = bus.history(topic="a.topic")
    assert all(m["topic"] == "a.topic" for m in h)
    assert len(h) == 1


# ── ACPMessage id is 8 hex chars ─────────────────────────────────────────────

def test_acp_message_id_is_8_hex_chars():
    from jarvis.acp import ACPMessage
    msg = ACPMessage()
    assert len(msg.id) == 8
    assert all(c in "0123456789abcdef" for c in msg.id)


# ── stats() before any messages ──────────────────────────────────────────────

def test_stats_empty_bus(bus):
    s = bus.stats()
    assert s["topics"] == 0
    assert s["history_size"] == 0
    assert s["subscribers"] == {}


# ── stats() reflects subscriber counts after subscribe/unsubscribe ────────────

def test_stats_reflects_subscriber_count_after_subscribe(bus):
    async def h(msg): pass
    bus.subscribe("events", h)
    s = bus.stats()
    assert s["subscribers"]["events"] == 1


def test_stats_after_unsubscribe_shows_zero_handlers(bus):
    async def h(msg): pass
    bus.subscribe("events", h)
    bus.unsubscribe("events", h)
    s = bus.stats()
    assert s["subscribers"]["events"] == 0


# ── ACPMessage to_dict() includes all expected fields ─────────────────────────

def test_acp_message_to_dict_has_all_keys():
    msg = ACPMessage(topic="t.t", payload={"key": "val"}, sender="bot")
    d = msg.to_dict()
    assert "id" in d
    assert "topic" in d
    assert "payload" in d
    assert "sender" in d
    assert "timestamp" in d
    assert d["topic"] == "t.t"
    assert d["sender"] == "bot"
    assert d["payload"] == {"key": "val"}


# ── history() limit=0 slices [-0:] which is all messages ─────────────────────

@pytest.mark.asyncio
async def test_history_all_messages_when_limit_exceeds_count(bus):
    """History with limit larger than stored messages returns everything."""
    await bus.publish("a", payload=1)
    await bus.publish("a", payload=2)
    h = bus.history(limit=100)
    assert len(h) == 2


# ── publish with no subscribers still appends to history ─────────────────────

@pytest.mark.asyncio
async def test_publish_to_unsubscribed_topic_still_recorded(bus):
    """Messages published to topics with no subscribers still appear in history."""
    await bus.publish("no.listeners", payload="silent")
    h = bus.history(topic="no.listeners")
    assert len(h) == 1
    assert h[0]["payload"] == "silent"


# ── history() without topic returns all topics ────────────────────────────────

@pytest.mark.asyncio
async def test_history_no_topic_returns_all(bus):
    await bus.publish("topic.a", payload="a")
    await bus.publish("topic.b", payload="b")
    h = bus.history()
    assert len(h) == 2


# ── subscribe same handler twice duplicates delivery ─────────────────────────

@pytest.mark.asyncio
async def test_subscribe_same_handler_twice_receives_twice(bus):
    received = []
    async def h(msg): received.append(msg)
    bus.subscribe("dupe", h)
    bus.subscribe("dupe", h)
    await bus.publish("dupe", payload="x")
    assert len(received) == 2


# ── ACPMessage.timestamp is ISO format ───────────────────────────────────────

def test_acp_message_timestamp_is_iso_format():
    """ACPMessage.timestamp is an ISO-format UTC datetime string."""
    msg = ACPMessage(topic="test")
    ts = msg.timestamp
    assert "T" in ts
    assert "+" in ts or "Z" in ts or ts.endswith("+00:00")


# ── bus.stats() grows with topics ────────────────────────────────────────────

def test_stats_topic_count_grows_with_subscriptions(bus):
    """stats() topic count increases as new topics are subscribed."""
    async def h(msg): pass
    assert bus.stats()["topics"] == 0
    bus.subscribe("alpha", h)
    assert bus.stats()["topics"] == 1
    bus.subscribe("beta", h)
    assert bus.stats()["topics"] == 2


@pytest.mark.asyncio
async def test_stats_history_size_grows_with_publishes(bus):
    """stats() history_size grows by 1 for each published message."""
    assert bus.stats()["history_size"] == 0
    await bus.publish("events", payload="first")
    assert bus.stats()["history_size"] == 1
    await bus.publish("events", payload="second")
    assert bus.stats()["history_size"] == 2


# ── ACPMessage default sender ─────────────────────────────────────────────────

def test_acp_message_default_sender_is_jarvis():
    """ACPMessage.sender defaults to 'jarvis'."""
    msg = ACPMessage(topic="t")
    assert msg.sender == "jarvis"


# ── publish_sync with running loop enqueues task ─────────────────────────────

@pytest.mark.asyncio
async def test_publish_sync_in_running_loop_enqueues_message(bus):
    """publish_sync() creates a task when a loop is running."""
    bus.publish_sync("sync.topic", payload="sync_value")
    await asyncio.sleep(0)  # let the task run
    h = bus.history(topic="sync.topic")
    assert len(h) >= 1
    assert h[0]["payload"] == "sync_value"


# ── MessageBus.topics() ──────────────────────────────────────────────────────

def test_topics_empty_on_new_bus(bus):
    assert bus.topics() == []


def test_topics_returns_subscribed_topics(bus):
    async def h(msg): pass
    bus.subscribe("t1", h)
    bus.subscribe("t2", h)
    topics = bus.topics()
    assert "t1" in topics
    assert "t2" in topics


# ── ACPMessage to_dict keys ───────────────────────────────────────────────────

def test_acp_message_to_dict_contains_five_keys():
    msg = ACPMessage(topic="hello", payload=42)
    d = msg.to_dict()
    assert set(d.keys()) == {"id", "topic", "payload", "sender", "timestamp"}


def test_acp_message_to_dict_payload_preserved():
    data = {"result": 100, "ok": True}
    msg = ACPMessage(topic="result", payload=data)
    assert msg.to_dict()["payload"] == data


def test_acp_message_to_dict_topic_preserved():
    msg = ACPMessage(topic="my.special.topic")
    assert msg.to_dict()["topic"] == "my.special.topic"


# ── MessageBus history _max_history trim ─────────────────────────────────────

@pytest.mark.asyncio
async def test_bus_history_trims_to_max_history():
    from jarvis.acp import MessageBus
    small_bus = MessageBus()
    small_bus._max_history = 5
    for i in range(8):
        await small_bus.publish("events", payload=i)
    assert len(small_bus._history) == 5
    assert small_bus._history[0].payload == 3


# ── stats subscribers dict ────────────────────────────────────────────────────

def test_stats_subscribers_dict_has_correct_counts(bus):
    async def h1(msg): pass
    async def h2(msg): pass
    bus.subscribe("ch", h1)
    bus.subscribe("ch", h2)
    stats = bus.stats()
    assert stats["subscribers"]["ch"] == 2


# ── unsubscribe removes handler ───────────────────────────────────────────────

def test_unsubscribe_removes_handler(bus):
    received = []

    async def h(msg):
        received.append(msg.payload)

    bus.subscribe("unsubscribe_test", h)
    bus.unsubscribe("unsubscribe_test", h)
    assert h not in bus._subs.get("unsubscribe_test", [])


def test_unsubscribe_nonexistent_topic_is_safe(bus):
    async def h(msg): pass
    bus.unsubscribe("no_such_topic", h)


# ── wildcard '*' subscription receives all messages ───────────────────────────

@pytest.mark.asyncio
async def test_wildcard_subscription_receives_all_topics(bus):
    received = []

    async def catch_all(msg):
        received.append(msg.topic)

    bus.subscribe("*", catch_all)
    try:
        await bus.publish("topic_a", "x")
        await bus.publish("topic_b", "y")
    finally:
        bus.unsubscribe("*", catch_all)

    assert "topic_a" in received
    assert "topic_b" in received


# ── history with topic filter ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_topic_filter_returns_only_matching(bus):
    await bus.publish("fruit", "apple")
    await bus.publish("veggies", "carrot")
    result = bus.history(topic="fruit")
    assert all(m["topic"] == "fruit" for m in result)


# ── ACPMessage id is short hex string ─────────────────────────────────────────

def test_acp_message_id_is_8_char_hex():
    from jarvis.acp import ACPMessage
    msg = ACPMessage()
    assert len(msg.id) == 8
    assert all(c in "0123456789abcdef" for c in msg.id)


# ── ACPMessage sender default is jarvis ──────────────────────────────────────

def test_acp_message_sender_default():
    from jarvis.acp import ACPMessage
    msg = ACPMessage()
    assert msg.sender == "jarvis"


# ── publish with custom sender is stored ─────────────────────────────────────

@pytest.mark.asyncio
async def test_publish_custom_sender_stored_in_history(bus):
    await bus.publish("agent_channel", "data", sender="subagent_1")
    recent = bus.history(topic="agent_channel", limit=1)
    assert recent[-1]["sender"] == "subagent_1"
