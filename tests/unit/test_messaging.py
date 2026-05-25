"""Unit tests for the inter-agent messaging subsystem."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from lauren_ai._exceptions import MessageBusTimeoutError, MessageValidationError
from lauren_ai._messaging import (
    AgentMessage,
    AgentMessageBus,
    AgentMessageMiddleware,
    AgentMessageType,
    DeadLetterReason,
    InMemoryAgentMessageTransport,
    JSONAgentMessageSerializer,
    MessageRetryPolicy,
)
from lauren_ai._signals import AgentMessageRequestCompleted, AgentMessageSent, SignalBus


def _message(
    *,
    from_agent: str = "planner",
    to: str | None = None,
    topic: str | None = None,
    message_type: AgentMessageType = AgentMessageType.NOTIFICATION,
    payload: dict | None = None,
    session_id: str | None = "session-1",
    correlation_id: uuid.UUID | None = None,
    reply_to: str | None = None,
    stream_id: str | None = None,
) -> AgentMessage:
    return AgentMessage(
        from_agent=from_agent,
        to=to,
        topic=topic,
        message_type=message_type,
        payload=dict(payload or {}),
        session_id=session_id,
        correlation_id=correlation_id,
        reply_to=reply_to,
        stream_id=stream_id,
    )


class TrackingMiddleware(AgentMessageMiddleware):
    def __init__(self) -> None:
        self.sent: list[uuid.UUID] = []
        self.received: list[tuple[uuid.UUID, str]] = []
        self.errors: list[str] = []

    async def on_send(self, message: AgentMessage) -> AgentMessage:
        self.sent.append(message.id)
        message.headers["x-test-middleware"] = "enabled"
        return message

    async def on_receive(
        self,
        message: AgentMessage,
        *,
        recipient: str,
    ) -> AgentMessage:
        self.received.append((message.id, recipient))
        message.payload["seen_by"] = recipient
        return message

    async def on_error(
        self,
        error: Exception,
        *,
        message: AgentMessage | None = None,
    ) -> None:
        self.errors.append(type(error).__name__)


class TestJSONAgentMessageSerializer:
    def test_round_trip_preserves_message_fields(self) -> None:
        serializer = JSONAgentMessageSerializer()
        message = _message(
            from_agent="researcher",
            to="analyst",
            message_type=AgentMessageType.QUERY,
            payload={"query": "status"},
            reply_to="researcher",
            stream_id="stream-1",
        )

        restored = serializer.deserialize(serializer.serialize(message))

        assert restored.id == message.id
        assert restored.from_agent == "researcher"
        assert restored.to == "analyst"
        assert restored.message_type is AgentMessageType.QUERY
        assert restored.payload == {"query": "status"}
        assert restored.reply_to == "researcher"
        assert restored.stream_id == "stream-1"


class TestRoutingAndFanout:
    @pytest.mark.asyncio
    async def test_send_and_drain_direct_message_with_session_isolation(self) -> None:
        bus = AgentMessageBus()
        await bus.register_agent("worker", session_id="session-1")
        await bus.register_agent("worker", session_id="session-2")

        result = await bus.send(
            _message(
                to="worker",
                message_type=AgentMessageType.TASK_REQUEST,
                payload={"task": "summarize"},
                session_id="session-1",
            )
        )

        assert result.receiver_count == 1
        assert (await bus.drain("worker", session_id="session-1"))[0].payload == {"task": "summarize"}
        assert await bus.drain("worker", session_id="session-2") == []

    @pytest.mark.asyncio
    async def test_broadcast_and_topic_publish_reach_matching_subscribers(self) -> None:
        bus = AgentMessageBus()
        await bus.register_agent("a", session_id="session-1")
        await bus.register_agent("b", session_id="session-1", include_broadcast=False, topics=["findings"])
        await bus.register_agent("c", session_id="session-1", include_broadcast=False)

        broadcast = await bus.broadcast(
            from_agent="planner",
            message_type=AgentMessageType.NOTIFICATION,
            payload={"kind": "broadcast"},
            session_id="session-1",
        )
        publish = await bus.publish(
            "findings",
            from_agent="planner",
            message_type=AgentMessageType.EVENT,
            payload={"kind": "topic"},
            session_id="session-1",
        )

        assert broadcast.receiver_count == 1
        assert publish.receiver_count == 1
        assert [m.payload["kind"] for m in await bus.drain("a", session_id="session-1")] == ["broadcast"]
        assert [m.payload["kind"] for m in await bus.drain("b", session_id="session-1")] == ["topic"]
        assert await bus.drain("c", session_id="session-1") == []

    @pytest.mark.asyncio
    async def test_concurrent_direct_sends_are_all_delivered_within_capacity(self) -> None:
        bus = AgentMessageBus(transport=InMemoryAgentMessageTransport(capacity=32))
        await bus.register_agent("worker", session_id="session-1")

        async with asyncio.TaskGroup() as tg:
            for index in range(10):
                tg.create_task(
                    bus.send(
                        _message(
                            to="worker",
                            message_type=AgentMessageType.EVENT,
                            payload={"index": index},
                        )
                    )
                )

        drained = await bus.drain("worker", session_id="session-1")

        assert sorted(message.payload["index"] for message in drained) == list(range(10))


class TestRequestResponse:
    @pytest.mark.asyncio
    async def test_request_ignores_unrelated_correlation_ids(self) -> None:
        bus = AgentMessageBus()
        request = _message(
            from_agent="planner",
            message_type=AgentMessageType.QUERY,
            payload={"question": "status"},
            reply_to="planner",
        )

        async def responder() -> None:
            await asyncio.sleep(0.01)
            await bus.send(
                _message(
                    from_agent="worker",
                    to="planner",
                    message_type=AgentMessageType.QUERY_RESPONSE,
                    payload={"value": "noise"},
                    correlation_id=uuid.uuid4(),
                )
            )
            await asyncio.sleep(0.01)
            await bus.send(
                _message(
                    from_agent="worker",
                    to="planner",
                    message_type=AgentMessageType.QUERY_RESPONSE,
                    payload={"value": "answer"},
                    correlation_id=request.id,
                )
            )

        responder_task = asyncio.create_task(responder())
        reply = await bus.request("worker", request, timeout=0.25)
        await responder_task

        assert reply.payload["value"] == "answer"

    @pytest.mark.asyncio
    async def test_request_retries_until_a_reply_arrives(self) -> None:
        bus = AgentMessageBus()
        await bus.register_agent("worker", session_id="session-1")
        request = _message(
            from_agent="planner",
            message_type=AgentMessageType.TASK_REQUEST,
            payload={"task": "retry"},
            reply_to="planner",
        )

        async def worker() -> list[int]:
            attempts: list[int] = []
            while len(attempts) < 2:
                drained = await bus.drain("worker", session_id="session-1")
                if drained:
                    attempts.extend(message.attempt for message in drained)
                    if len(attempts) >= 2:
                        await bus.reply(
                            drained[-1],
                            from_agent="worker",
                            message_type=AgentMessageType.TASK_RESULT,
                            payload={"attempt": drained[-1].attempt},
                        )
                        return attempts
                await asyncio.sleep(0.005)
            return attempts

        worker_task = asyncio.create_task(worker())
        reply = await bus.request(
            "worker",
            request,
            timeout=0.12,
            retry_policy=MessageRetryPolicy(
                max_attempts=2,
                initial_backoff_s=0.01,
            ),
        )
        attempts = await worker_task

        assert attempts == [1, 2]
        assert reply.payload == {"attempt": 2}

    @pytest.mark.asyncio
    async def test_request_target_mismatch_raises_validation_error(self) -> None:
        bus = AgentMessageBus()
        request = _message(
            from_agent="planner",
            to="other-worker",
            message_type=AgentMessageType.QUERY,
            payload={"question": "status"},
        )

        with pytest.raises(MessageValidationError, match="target mismatch"):
            await bus.request("worker", request, timeout=0.05)

    @pytest.mark.asyncio
    async def test_request_timeout_emits_completion_signal(self) -> None:
        signals = SignalBus()
        bus = AgentMessageBus(signals=signals)
        completed: list[AgentMessageRequestCompleted] = []

        @signals.on(AgentMessageRequestCompleted)
        async def on_completed(event: AgentMessageRequestCompleted) -> None:
            completed.append(event)

        with pytest.raises(MessageBusTimeoutError):
            await bus.request(
                "worker",
                _message(
                    from_agent="planner",
                    message_type=AgentMessageType.QUERY,
                    payload={"question": "status"},
                ),
                timeout=0.03,
            )

        assert len(completed) == 1
        assert completed[0].timed_out is True
        assert completed[0].target == "worker"


class TestStreamingAndCancellation:
    @pytest.mark.asyncio
    async def test_request_stream_propagates_cancel_on_consumer_close(self) -> None:
        bus = AgentMessageBus()
        worker = await bus.subscribe("worker", session_id="session-1", include_broadcast=False)
        request = _message(
            from_agent="planner",
            message_type=AgentMessageType.QUERY,
            payload={"stream": True},
            reply_to="planner",
            stream_id="stream-1",
        )

        stream = await bus.request_stream("worker", request, timeout=0.25)
        incoming = await worker.get(timeout=0.1)
        await bus.stream_chunk(
            incoming,
            from_agent="worker",
            payload={"chunk": "one"},
            stream_id="stream-1",
        )

        first = await anext(stream)
        await stream.aclose()
        cancel_message = await worker.get(timeout=0.1)
        await worker.close()

        assert first.payload == {"chunk": "one"}
        assert cancel_message.message_type is AgentMessageType.CANCEL
        assert cancel_message.correlation_id == request.id

    @pytest.mark.asyncio
    async def test_request_stream_timeout_raises_message_bus_timeout(self) -> None:
        bus = AgentMessageBus()
        stream = await bus.request_stream(
            "worker",
            _message(
                from_agent="planner",
                message_type=AgentMessageType.QUERY,
                payload={"stream": True},
                reply_to="planner",
            ),
            timeout=0.03,
        )

        with pytest.raises(MessageBusTimeoutError):
            await anext(stream)


class TestMiddlewareAndErrors:
    @pytest.mark.asyncio
    async def test_middleware_wraps_send_and_receive_paths(self) -> None:
        middleware = TrackingMiddleware()
        bus = AgentMessageBus(middleware=[middleware])
        subscription = await bus.subscribe("worker", session_id="session-1", include_broadcast=False)

        await bus.send(
            _message(
                to="worker",
                message_type=AgentMessageType.NOTIFICATION,
                payload={"value": "ok"},
            )
        )
        received = await subscription.get(timeout=0.1)
        await subscription.close()

        assert middleware.sent == [received.id]
        assert middleware.received == [(received.id, "worker")]
        assert received.headers["x-test-middleware"] == "enabled"
        assert received.payload["seen_by"] == "worker"

    @pytest.mark.asyncio
    async def test_validation_errors_hit_error_middleware_and_dead_letter_buffer(self) -> None:
        middleware = TrackingMiddleware()
        bus = AgentMessageBus(middleware=[middleware])

        with pytest.raises(MessageValidationError):
            await bus.send(
                _message(
                    to="worker",
                    payload={"bad": object()},
                )
            )

        assert middleware.errors == ["MessageValidationError"]
        assert bus.dead_letters()[-1].reason is DeadLetterReason.VALIDATION_ERROR


class TestBackpressureAndShutdown:
    @pytest.mark.asyncio
    async def test_bounded_queue_drops_when_full_and_records_dead_letter(self) -> None:
        bus = AgentMessageBus(transport=InMemoryAgentMessageTransport(capacity=1))
        await bus.register_agent("worker", session_id="session-1", capacity=1)

        first = await bus.send(_message(to="worker", payload={"index": 1}))
        second = await bus.send(_message(to="worker", payload={"index": 2}))

        assert first.receiver_count == 1
        assert second.dropped_count == 1
        assert bus.dead_letters()[-1].reason is DeadLetterReason.QUEUE_FULL

    @pytest.mark.asyncio
    async def test_shutdown_rejects_new_messages_and_records_dead_letter(self) -> None:
        bus = AgentMessageBus()
        await bus.shutdown()

        with pytest.raises(RuntimeError, match="shut down"):
            await bus.send(_message(to="worker", payload={"value": "late"}))

        assert bus.dead_letters()[-1].reason is DeadLetterReason.BUS_CLOSED


class TestSignals:
    @pytest.mark.asyncio
    async def test_send_emits_agent_message_sent_signal(self) -> None:
        signals = SignalBus()
        bus = AgentMessageBus(signals=signals)
        await bus.register_agent("worker", session_id="session-1")
        sent: list[AgentMessageSent] = []

        @signals.on(AgentMessageSent)
        async def on_sent(event: AgentMessageSent) -> None:
            sent.append(event)

        message = _message(
            to="worker",
            message_type=AgentMessageType.NOTIFICATION,
            payload={"value": "ok"},
        )
        await bus.send(message)

        assert len(sent) == 1
        assert sent[0].message_id == message.id
        assert sent[0].receiver_count == 1
