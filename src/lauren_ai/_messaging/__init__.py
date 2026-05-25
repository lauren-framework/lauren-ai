"""Async inter-agent messaging primitives for ``lauren-ai``.

The messaging subsystem is intentionally transport-shaped, mirroring the rest
of the repository:

* strongly typed dataclass envelopes
* protocol-based transport abstraction
* in-memory async transport for the current runtime
* optional middleware for cross-cutting concerns
* SignalBus hooks for observability

The public entry point is :class:`AgentMessageBus`.  It exposes ergonomic
helpers for:

* point-to-point delivery
* broadcast and topic fan-out
* request/response with timeout + retry policy
* correlated streaming responses
* graceful shutdown and dead-letter inspection

The default delivery semantics are deliberately explicit:

* direct and broadcast deliveries are **at-most-once** per destination queue
* bounded queues use ``put_nowait`` and drop when full
* request retries are **at-least-once** across attempts, so handlers must be
  tolerant of duplicate request IDs when retries are enabled
"""

from __future__ import annotations

__all__ = [
    "AgentMessage",
    "AgentMessageBus",
    "AgentMessageMiddleware",
    "AgentMessageSerializer",
    "AgentMessageTransport",
    "AgentMessageType",
    "DeadLetterReason",
    "DeadLetterRecord",
    "DeliveryGuarantee",
    "DeliveryResult",
    "InMemoryAgentMessageTransport",
    "JSONAgentMessageSerializer",
    "MessageRetryPolicy",
    "MessageSubscription",
]

import asyncio
import contextlib
import json
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Protocol, runtime_checkable

from lauren_ai._exceptions import MessageBusTimeoutError, MessageValidationError


class AgentMessageType(StrEnum):
    """Semantic type for an :class:`AgentMessage`."""

    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    QUERY = "query"
    QUERY_RESPONSE = "query_response"
    EVENT = "event"
    NOTIFICATION = "notification"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    ERROR = "error"
    CANCEL = "cancel"


class DeliveryGuarantee(StrEnum):
    """Delivery guarantee exposed by the bus/transport."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE_RETRY = "at_least_once_retry"


class DeadLetterReason(StrEnum):
    """Why a message was moved to the dead-letter buffer."""

    QUEUE_FULL = "queue_full"
    BUS_CLOSED = "bus_closed"
    VALIDATION_ERROR = "validation_error"
    SUBSCRIPTION_CLOSED = "subscription_closed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class MessageRetryPolicy:
    """Retry policy used by :meth:`AgentMessageBus.request`.

    :param max_attempts: Total attempts including the initial send.
    :param initial_backoff_s: Delay before the second attempt.
    :param backoff_multiplier: Multiplier applied after each failed attempt.
    :param max_backoff_s: Maximum delay between attempts.
    """

    max_attempts: int = 1
    initial_backoff_s: float = 0.0
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 30.0

    def delays(self) -> list[float]:
        """Return one delay per retry after the first attempt."""
        if self.max_attempts <= 1:
            return []
        delay = max(0.0, self.initial_backoff_s)
        values: list[float] = []
        for _ in range(self.max_attempts - 1):
            values.append(delay)
            delay = min(self.max_backoff_s, delay * self.backoff_multiplier) if delay > 0 else 0.0
        return values


@dataclass
class AgentMessage:
    """A structured inter-agent message.

    :param id: Globally unique message identifier.
    :param from_agent: Sender identity.
    :param message_type: Semantic message type.
    :param payload: JSON-serialisable payload.
    :param created_at: UTC timestamp when the message was created.
    :param to: Direct recipient agent name, or ``None`` for broadcast/topic.
    :param topic: Topic name for publish/subscribe delivery.
    :param correlation_id: Request/response correlation ID.
    :param causation_id: Parent message ID that triggered this message.
    :param reply_to: Logical reply address override.
    :param session_id: Optional session/conversation scope.
    :param task_id: Optional task/run scope.
    :param stream_id: Logical stream identifier for chunked replies.
    :param attempt: Attempt counter for retries.
    :param expects_response: Whether the sender expects a reply.
    :param headers: String headers for lightweight routing metadata.
    :param metadata: Additional structured metadata.
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    from_agent: str = ""
    message_type: AgentMessageType = AgentMessageType.NOTIFICATION
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    to: str | None = None
    topic: str | None = None
    correlation_id: uuid.UUID | None = None
    causation_id: uuid.UUID | None = None
    reply_to: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    stream_id: str | None = None
    attempt: int = 1
    expects_response: bool = False
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message invariants eagerly."""
        if not self.from_agent:
            raise ValueError("AgentMessage.from_agent must be a non-empty string")
        if self.to is not None and self.topic is not None:
            raise ValueError("AgentMessage cannot target both 'to' and 'topic' simultaneously")
        if self.attempt < 1:
            raise ValueError("AgentMessage.attempt must be >= 1")
        if self.created_at.tzinfo is None:
            raise ValueError("AgentMessage.created_at must be timezone-aware")
        if not isinstance(self.payload, dict):
            raise TypeError("AgentMessage.payload must be a dict[str, Any]")
        if not isinstance(self.headers, dict):
            raise TypeError("AgentMessage.headers must be a dict[str, str]")
        if not isinstance(self.metadata, dict):
            raise TypeError("AgentMessage.metadata must be a dict[str, Any]")

    @property
    def is_broadcast(self) -> bool:
        """Return ``True`` when the message is a broadcast."""
        return self.to is None and self.topic is None

    @property
    def is_topic(self) -> bool:
        """Return ``True`` when the message targets a topic."""
        return self.topic is not None

    def for_retry(self, *, attempt: int) -> AgentMessage:
        """Return a copy of the message for a retry attempt."""
        return AgentMessage(
            id=self.id,
            from_agent=self.from_agent,
            message_type=self.message_type,
            payload=dict(self.payload),
            created_at=self.created_at,
            to=self.to,
            topic=self.topic,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            reply_to=self.reply_to,
            session_id=self.session_id,
            task_id=self.task_id,
            stream_id=self.stream_id,
            attempt=attempt,
            expects_response=self.expects_response,
            headers=dict(self.headers),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class DeadLetterRecord:
    """Dead-letter entry recorded by the transport."""

    message: AgentMessage
    reason: DeadLetterReason
    target: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class DeliveryResult:
    """Result of a transport send operation."""

    receiver_count: int
    dropped_count: int = 0
    guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_MOST_ONCE


def _json_default(value: Any) -> Any:
    """Encode non-primitive message fields for JSON serialisation."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _message_to_mapping(message: AgentMessage) -> dict[str, Any]:
    """Serialise an :class:`AgentMessage` to a JSON-friendly dict."""
    return {
        "id": str(message.id),
        "from_agent": message.from_agent,
        "message_type": message.message_type.value,
        "payload": message.payload,
        "created_at": message.created_at.isoformat(),
        "to": message.to,
        "topic": message.topic,
        "correlation_id": str(message.correlation_id) if message.correlation_id is not None else None,
        "causation_id": str(message.causation_id) if message.causation_id is not None else None,
        "reply_to": message.reply_to,
        "session_id": message.session_id,
        "task_id": message.task_id,
        "stream_id": message.stream_id,
        "attempt": message.attempt,
        "expects_response": message.expects_response,
        "headers": message.headers,
        "metadata": message.metadata,
    }


def _mapping_to_message(data: Mapping[str, Any]) -> AgentMessage:
    """Deserialize an :class:`AgentMessage` from a mapping."""
    return AgentMessage(
        id=uuid.UUID(str(data["id"])),
        from_agent=str(data["from_agent"]),
        message_type=AgentMessageType(str(data["message_type"])),
        payload=dict(data.get("payload", {})),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        to=data.get("to"),
        topic=data.get("topic"),
        correlation_id=(uuid.UUID(str(data["correlation_id"])) if data.get("correlation_id") is not None else None),
        causation_id=(uuid.UUID(str(data["causation_id"])) if data.get("causation_id") is not None else None),
        reply_to=data.get("reply_to"),
        session_id=data.get("session_id"),
        task_id=data.get("task_id"),
        stream_id=data.get("stream_id"),
        attempt=int(data.get("attempt", 1)),
        expects_response=bool(data.get("expects_response", False)),
        headers={str(k): str(v) for k, v in dict(data.get("headers", {})).items()},
        metadata=dict(data.get("metadata", {})),
    )


@runtime_checkable
class AgentMessageSerializer(Protocol):
    """Serializer interface for message transports."""

    def serialize(self, message: AgentMessage) -> bytes: ...

    def deserialize(self, payload: bytes) -> AgentMessage: ...


class JSONAgentMessageSerializer:
    """JSON serializer for :class:`AgentMessage`."""

    def serialize(self, message: AgentMessage) -> bytes:
        """Serialize *message* to UTF-8 JSON bytes."""
        return json.dumps(_message_to_mapping(message), default=_json_default).encode("utf-8")

    def deserialize(self, payload: bytes) -> AgentMessage:
        """Deserialize UTF-8 JSON bytes into an :class:`AgentMessage`."""
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise TypeError("Serialized AgentMessage payload must decode to a JSON object")
        return _mapping_to_message(data)


class AgentMessageMiddleware:
    """Base class for message bus middleware.

    Subclass and override any of the hooks you need.  Methods are called in
    registration order.
    """

    async def on_send(self, message: AgentMessage) -> AgentMessage:
        """Inspect or replace an outbound message."""
        return message

    async def on_receive(
        self,
        message: AgentMessage,
        *,
        recipient: str,
    ) -> AgentMessage:
        """Inspect or replace a delivered message."""
        return message

    async def on_error(
        self,
        error: Exception,
        *,
        message: AgentMessage | None = None,
    ) -> None:
        """Observe an internal bus error."""
        return None


_CLOSE_SENTINEL = object()


def _as_message(item: AgentMessage | object) -> AgentMessage:
    """Return *item* as an :class:`AgentMessage` or raise if it is the sentinel."""
    if item is _CLOSE_SENTINEL:
        raise RuntimeError("MessageSubscription is closed")
    if not isinstance(item, AgentMessage):
        raise TypeError(f"Expected AgentMessage, got {type(item).__name__}")
    return item


class MessageSubscription:
    """A bounded async subscription queue."""

    def __init__(
        self,
        *,
        agent_name: str,
        session_id: str | None,
        queue: asyncio.Queue[AgentMessage | object],
        close_callback: Any,
        include_broadcast: bool,
        topics: frozenset[str],
        receive_hook: Callable[[AgentMessage], Awaitable[AgentMessage]] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.session_id = session_id
        self._queue = queue
        self._close_callback = close_callback
        self.include_broadcast = include_broadcast
        self.topics = topics
        self._receive_hook = receive_hook
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the subscription has been closed."""
        return self._closed

    async def get(self, *, timeout: float | None = None) -> AgentMessage:
        """Wait for the next message, optionally bounded by *timeout*."""
        if self._closed and self._queue.empty():
            raise RuntimeError("MessageSubscription is closed")
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if item is _CLOSE_SENTINEL:
            self._closed = True
            raise RuntimeError("MessageSubscription is closed")
        message = _as_message(item)
        if self._receive_hook is not None:
            return await self._receive_hook(message)
        return message

    def get_nowait(self) -> AgentMessage:
        """Return the next queued message without waiting."""
        item = self._queue.get_nowait()
        if item is _CLOSE_SENTINEL:
            self._closed = True
            raise RuntimeError("MessageSubscription is closed")
        return _as_message(item)

    def drain(self) -> list[AgentMessage]:
        """Return all currently buffered messages.

        Receive middleware is only applied by the async ``get()`` path and by
        :meth:`AgentMessageBus.drain`.
        """
        items: list[AgentMessage] = []
        while True:
            with contextlib.suppress(asyncio.QueueEmpty):
                item = self._queue.get_nowait()
                if item is _CLOSE_SENTINEL:
                    self._closed = True
                    break
                items.append(_as_message(item))
                continue
            break
        return items

    async def close(self) -> None:
        """Close the subscription and unregister it from the transport."""
        if self._closed:
            return
        self._closed = True
        await self._close_callback(self)
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(_CLOSE_SENTINEL)

    def set_receive_hook(
        self,
        hook: Callable[[AgentMessage], Awaitable[AgentMessage]] | None,
    ) -> MessageSubscription:
        """Attach a receive hook used by async retrieval methods."""
        self._receive_hook = hook
        return self

    def __aiter__(self) -> MessageSubscription:
        return self

    async def __anext__(self) -> AgentMessage:
        try:
            return await self.get()
        except RuntimeError as exc:
            raise StopAsyncIteration from exc


@runtime_checkable
class AgentMessageTransport(Protocol):
    """Transport abstraction for message delivery."""

    async def register_agent(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        capacity: int | None = None,
        include_broadcast: bool = True,
        topics: Sequence[str] | None = None,
    ) -> None: ...

    async def subscribe(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        capacity: int | None = None,
        include_broadcast: bool = True,
        topics: Sequence[str] | None = None,
    ) -> MessageSubscription: ...

    async def send(self, message: AgentMessage) -> DeliveryResult: ...

    async def drain(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
    ) -> list[AgentMessage]: ...

    async def shutdown(self) -> None: ...

    def dead_letters(self) -> list[DeadLetterRecord]: ...


@runtime_checkable
class _DeadLetterRecorder(Protocol):
    def record_dead_letter(self, record: DeadLetterRecord) -> None: ...


@dataclass
class _Mailbox:
    queue: asyncio.Queue[AgentMessage | object]
    include_broadcast: bool
    topics: set[str]


class InMemoryAgentMessageTransport:
    """In-process asyncio transport for inter-agent messaging."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        dead_letter_capacity: int = 1024,
    ) -> None:
        self._capacity = capacity
        self._lock = asyncio.Lock()
        self._mailboxes: dict[tuple[str, str | None], _Mailbox] = {}
        self._watchers: dict[tuple[str, str | None], set[MessageSubscription]] = {}
        self._dead_letters: deque[DeadLetterRecord] = deque(maxlen=dead_letter_capacity)
        self._closed = False

    async def register_agent(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        capacity: int | None = None,
        include_broadcast: bool = True,
        topics: Sequence[str] | None = None,
    ) -> None:
        """Ensure a durable inbox exists for *agent_name*."""
        if not agent_name:
            raise ValueError("agent_name must be a non-empty string")
        async with self._lock:
            self._assert_open()
            key = (agent_name, session_id)
            mailbox = self._mailboxes.get(key)
            if mailbox is None:
                mailbox = _Mailbox(
                    queue=asyncio.Queue(maxsize=capacity or self._capacity),
                    include_broadcast=include_broadcast,
                    topics=set(topics or ()),
                )
                self._mailboxes[key] = mailbox
            else:
                mailbox.include_broadcast = include_broadcast
                mailbox.topics.update(topics or ())

    async def subscribe(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        capacity: int | None = None,
        include_broadcast: bool = True,
        topics: Sequence[str] | None = None,
    ) -> MessageSubscription:
        """Create an ephemeral subscription for *agent_name*."""
        await self.register_agent(
            agent_name,
            session_id=session_id,
            include_broadcast=include_broadcast,
            topics=topics,
        )
        key = (agent_name, session_id)
        subscription = MessageSubscription(
            agent_name=agent_name,
            session_id=session_id,
            queue=asyncio.Queue(maxsize=capacity or self._capacity),
            close_callback=self._unsubscribe,
            include_broadcast=include_broadcast,
            topics=frozenset(topics or ()),
        )
        async with self._lock:
            self._watchers.setdefault(key, set()).add(subscription)
        return subscription

    async def send(self, message: AgentMessage) -> DeliveryResult:
        """Deliver *message* to matching inboxes/subscriptions."""
        async with self._lock:
            self._assert_open()
            mailbox_targets: list[tuple[str | None, asyncio.Queue[AgentMessage | object]]] = []
            watcher_targets: list[tuple[str | None, asyncio.Queue[AgentMessage | object]]] = []

            if message.to is not None:
                key = (message.to, message.session_id)
                if key not in self._mailboxes:
                    self._mailboxes[key] = _Mailbox(
                        queue=asyncio.Queue(maxsize=self._capacity),
                        include_broadcast=True,
                        topics=set(),
                    )
                mailbox_targets.append((message.to, self._mailboxes[key].queue))
                for subscription in self._watchers.get(key, set()):
                    if not subscription.closed:
                        watcher_targets.append((message.to, subscription._queue))
            else:
                for (recipient, session_id), mailbox in self._mailboxes.items():
                    if not self._matches_session(session_id, message.session_id):
                        continue
                    if (
                        message.topic is None
                        and mailbox.include_broadcast
                        or message.topic is not None
                        and message.topic in mailbox.topics
                    ):
                        mailbox_targets.append((recipient, mailbox.queue))
                for (recipient, session_id), subscriptions in self._watchers.items():
                    if not self._matches_session(session_id, message.session_id):
                        continue
                    for subscription in subscriptions:
                        if subscription.closed:
                            continue
                        if (
                            message.topic is None
                            and subscription.include_broadcast
                            or message.topic is not None
                            and message.topic in subscription.topics
                        ):
                            watcher_targets.append((recipient, subscription._queue))

        receiver_count = 0
        dropped_count = 0
        for target, queue in [*mailbox_targets, *watcher_targets]:
            try:
                queue.put_nowait(message)
                receiver_count += 1
            except asyncio.QueueFull:
                dropped_count += 1
                self._dead_letters.append(
                    DeadLetterRecord(
                        message=message,
                        reason=DeadLetterReason.QUEUE_FULL,
                        target=target,
                    )
                )
        return DeliveryResult(receiver_count=receiver_count, dropped_count=dropped_count)

    async def drain(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
    ) -> list[AgentMessage]:
        """Drain the durable inbox for *agent_name*."""
        key = (agent_name, session_id)
        async with self._lock:
            mailbox = self._mailboxes.get(key)
            if mailbox is None:
                return []
            queue = mailbox.queue
        messages: list[AgentMessage] = []
        while True:
            with contextlib.suppress(asyncio.QueueEmpty):
                item = queue.get_nowait()
                if item is _CLOSE_SENTINEL:
                    break
                messages.append(_as_message(item))
                continue
            break
        return messages

    async def shutdown(self) -> None:
        """Prevent new sends and close all subscriptions."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            watchers = [subscription for items in self._watchers.values() for subscription in items]
            self._watchers.clear()
            mailboxes = list(self._mailboxes.values())
        for subscription in watchers:
            with contextlib.suppress(asyncio.QueueFull):
                subscription._queue.put_nowait(_CLOSE_SENTINEL)
        for mailbox in mailboxes:
            with contextlib.suppress(asyncio.QueueFull):
                mailbox.queue.put_nowait(_CLOSE_SENTINEL)

    def dead_letters(self) -> list[DeadLetterRecord]:
        """Return a snapshot of dead-letter entries."""
        return list(self._dead_letters)

    def record_dead_letter(self, record: DeadLetterRecord) -> None:
        """Record a dead-letter entry."""
        self._dead_letters.append(record)

    async def _unsubscribe(self, subscription: MessageSubscription) -> None:
        key = (subscription.agent_name, subscription.session_id)
        async with self._lock:
            items = self._watchers.get(key)
            if items is None:
                return
            items.discard(subscription)
            if not items:
                self._watchers.pop(key, None)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent message transport is shut down")

    @staticmethod
    def _matches_session(target_session: str | None, message_session: str | None) -> bool:
        if message_session is None:
            return True
        return target_session == message_session


class AgentMessageBus:
    """High-level messaging facade over an :class:`AgentMessageTransport`."""

    def __init__(
        self,
        *,
        transport: AgentMessageTransport | None = None,
        serializer: AgentMessageSerializer | None = None,
        middleware: Sequence[AgentMessageMiddleware] | None = None,
        signals: Any | None = None,
        capacity: int = 256,
    ) -> None:
        self._serializer = serializer or JSONAgentMessageSerializer()
        self._transport = transport or InMemoryAgentMessageTransport(capacity=capacity)
        self._middleware = list(middleware or ())
        self._signals = signals
        self._closed = False

    @property
    def serializer(self) -> AgentMessageSerializer:
        """Serializer used by the bus."""
        return self._serializer

    async def register_agent(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        capacity: int | None = None,
        include_broadcast: bool = True,
        topics: Sequence[str] | None = None,
    ) -> None:
        """Ensure a durable inbox exists for *agent_name*."""
        await self._transport.register_agent(
            agent_name,
            session_id=session_id,
            capacity=capacity,
            include_broadcast=include_broadcast,
            topics=topics,
        )

    async def subscribe(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
        capacity: int | None = None,
        include_broadcast: bool = True,
        topics: Sequence[str] | None = None,
    ) -> MessageSubscription:
        """Create an ephemeral subscription for *agent_name*."""
        subscription = await self._transport.subscribe(
            agent_name,
            session_id=session_id,
            capacity=capacity,
            include_broadcast=include_broadcast,
            topics=topics,
        )
        if self._middleware:

            async def _receive_hook(message: AgentMessage) -> AgentMessage:
                return await self._apply_receive_middleware(message, recipient=subscription.agent_name)

            subscription.set_receive_hook(_receive_hook)
        return subscription

    async def send(self, message: AgentMessage) -> DeliveryResult:
        """Send a message after applying middleware and validation."""
        outbound = message
        try:
            self._assert_open()
            # Serializer round-trip validates JSON compatibility early.
            try:
                outbound = self._serializer.deserialize(self._serializer.serialize(outbound))
            except Exception as exc:
                raise MessageValidationError(
                    "AgentMessage must be serializable by the configured serializer",
                    cause=exc,
                ) from exc
            for item in self._middleware:
                outbound = await item.on_send(outbound)
            result = await self._transport.send(outbound)
            await self._emit_signal(
                "AgentMessageSent",
                message_id=outbound.id,
                from_agent=outbound.from_agent,
                to=outbound.to,
                topic=outbound.topic,
                session_id=outbound.session_id,
                task_id=outbound.task_id,
                correlation_id=outbound.correlation_id,
                message_type=outbound.message_type,
                receiver_count=result.receiver_count,
                dropped_count=result.dropped_count,
                attempt=outbound.attempt,
            )
            return result
        except Exception as exc:
            await self._emit_error(exc, outbound)
            raise

    async def broadcast(
        self,
        *,
        from_agent: str,
        message_type: AgentMessageType,
        payload: dict[str, Any],
        session_id: str | None = None,
        task_id: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeliveryResult:
        """Broadcast a message to all registered broadcast subscribers."""
        return await self.send(
            AgentMessage(
                from_agent=from_agent,
                message_type=message_type,
                payload=payload,
                session_id=session_id,
                task_id=task_id,
                headers=dict(headers or {}),
                metadata=dict(metadata or {}),
            )
        )

    async def publish(
        self,
        topic: str,
        *,
        from_agent: str,
        message_type: AgentMessageType,
        payload: dict[str, Any],
        session_id: str | None = None,
        task_id: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeliveryResult:
        """Publish a topic-scoped message."""
        return await self.send(
            AgentMessage(
                from_agent=from_agent,
                message_type=message_type,
                payload=payload,
                topic=topic,
                session_id=session_id,
                task_id=task_id,
                headers=dict(headers or {}),
                metadata=dict(metadata or {}),
            )
        )

    async def request(
        self,
        target: str,
        request: AgentMessage,
        *,
        timeout: float,
        retry_policy: MessageRetryPolicy | None = None,
    ) -> AgentMessage:
        """Send *request* and await a correlated non-stream response."""
        self._assert_open()
        policy = retry_policy or MessageRetryPolicy()
        started_at = time.monotonic()
        deadline = started_at + timeout
        attempts_made = 0
        prepared_request = self._prepare_request(target=target, request=request)
        subscription = await self.subscribe(
            prepared_request.reply_to or prepared_request.from_agent,
            session_id=prepared_request.session_id,
            include_broadcast=False,
        )
        try:
            delays = [0.0, *policy.delays()]
            for attempt, delay in enumerate(delays, start=1):
                if delay > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(delay, remaining))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                outbound = prepared_request.for_retry(attempt=attempt)
                attempts_made = attempt
                await self.send(outbound)
                try:
                    remaining_attempts = max(1, policy.max_attempts - attempt + 1)
                    attempt_deadline = time.monotonic() + (remaining / remaining_attempts)
                    while True:
                        remaining = min(deadline, attempt_deadline) - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError
                        reply = await subscription.get(timeout=remaining)
                        if reply.correlation_id != prepared_request.id:
                            continue
                        await self._emit_signal(
                            "AgentMessageRequestCompleted",
                            request_id=prepared_request.id,
                            from_agent=prepared_request.from_agent,
                            target=target,
                            session_id=prepared_request.session_id,
                            task_id=prepared_request.task_id,
                            elapsed_ms=(time.monotonic() - started_at) * 1000,
                            attempts=attempt,
                            timed_out=False,
                        )
                        return reply
                except TimeoutError:
                    if attempt >= policy.max_attempts:
                        break
            elapsed_ms = (time.monotonic() - started_at) * 1000
            await self._emit_signal(
                "AgentMessageRequestCompleted",
                request_id=prepared_request.id,
                from_agent=prepared_request.from_agent,
                target=target,
                session_id=prepared_request.session_id,
                task_id=prepared_request.task_id,
                elapsed_ms=elapsed_ms,
                attempts=attempts_made,
                timed_out=True,
            )
            raise MessageBusTimeoutError(target=target, elapsed_ms=elapsed_ms)
        finally:
            await subscription.close()

    async def request_stream(
        self,
        target: str,
        request: AgentMessage,
        *,
        timeout: float,
        propagate_cancel: bool = True,
    ) -> AsyncIterator[AgentMessage]:
        """Send *request* and yield correlated streaming replies."""
        self._assert_open()
        started_at = time.monotonic()
        deadline = started_at + timeout
        prepared_request = self._prepare_request(target=target, request=request)
        subscription = await self.subscribe(
            prepared_request.reply_to or prepared_request.from_agent,
            session_id=prepared_request.session_id,
            include_broadcast=False,
        )
        try:
            await self.send(prepared_request)
        except Exception:
            await subscription.close()
            raise

        async def _iterator() -> AsyncIterator[AgentMessage]:
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    reply = await subscription.get(timeout=remaining)
                    if reply.correlation_id != prepared_request.id:
                        continue
                    if reply.message_type == AgentMessageType.STREAM_END:
                        break
                    yield reply
                await self._emit_signal(
                    "AgentMessageRequestCompleted",
                    request_id=prepared_request.id,
                    from_agent=prepared_request.from_agent,
                    target=target,
                    session_id=prepared_request.session_id,
                    task_id=prepared_request.task_id,
                    elapsed_ms=(time.monotonic() - started_at) * 1000,
                    attempts=1,
                    timed_out=False,
                )
            except TimeoutError as exc:
                elapsed_ms = (time.monotonic() - started_at) * 1000
                await self._emit_signal(
                    "AgentMessageRequestCompleted",
                    request_id=prepared_request.id,
                    from_agent=prepared_request.from_agent,
                    target=target,
                    session_id=prepared_request.session_id,
                    task_id=prepared_request.task_id,
                    elapsed_ms=elapsed_ms,
                    attempts=1,
                    timed_out=True,
                )
                raise MessageBusTimeoutError(target=target, elapsed_ms=elapsed_ms) from exc
            except (asyncio.CancelledError, GeneratorExit):
                if propagate_cancel:
                    with contextlib.suppress(Exception):
                        await self.cancel(
                            from_agent=prepared_request.from_agent,
                            target=target,
                            correlation_id=prepared_request.id,
                            session_id=prepared_request.session_id,
                            task_id=prepared_request.task_id,
                        )
                raise
            finally:
                await subscription.close()

        return _iterator()

    async def reply(
        self,
        request: AgentMessage,
        *,
        from_agent: str,
        message_type: AgentMessageType,
        payload: dict[str, Any],
        to: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeliveryResult:
        """Reply to *request* using its correlation and scope metadata."""
        return await self.send(
            AgentMessage(
                from_agent=from_agent,
                to=to or request.reply_to or request.from_agent,
                message_type=message_type,
                payload=payload,
                correlation_id=request.id,
                causation_id=request.id,
                session_id=request.session_id,
                task_id=request.task_id,
                stream_id=request.stream_id,
                headers=dict(headers or {}),
                metadata=dict(metadata or {}),
            )
        )

    async def stream_chunk(
        self,
        request: AgentMessage,
        *,
        from_agent: str,
        payload: dict[str, Any],
        to: str | None = None,
        stream_id: str | None = None,
    ) -> DeliveryResult:
        """Emit one correlated streaming chunk."""
        return await self.reply(
            request,
            from_agent=from_agent,
            message_type=AgentMessageType.STREAM_CHUNK,
            payload=payload,
            to=to,
            metadata={"stream_id": stream_id or request.stream_id} if stream_id else {},
        )

    async def end_stream(
        self,
        request: AgentMessage,
        *,
        from_agent: str,
        to: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> DeliveryResult:
        """Signal the end of a correlated stream."""
        return await self.reply(
            request,
            from_agent=from_agent,
            message_type=AgentMessageType.STREAM_END,
            payload=dict(payload or {}),
            to=to,
        )

    async def cancel(
        self,
        *,
        from_agent: str,
        target: str,
        correlation_id: uuid.UUID,
        session_id: str | None = None,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> DeliveryResult:
        """Send a cancellation message tied to *correlation_id*."""
        return await self.send(
            AgentMessage(
                from_agent=from_agent,
                to=target,
                message_type=AgentMessageType.CANCEL,
                payload=dict(payload or {}),
                correlation_id=correlation_id,
                session_id=session_id,
                task_id=task_id,
            )
        )

    async def drain(
        self,
        agent_name: str,
        *,
        session_id: str | None = None,
    ) -> list[AgentMessage]:
        """Drain the durable inbox for *agent_name*."""
        messages = await self._transport.drain(agent_name, session_id=session_id)
        if not self._middleware:
            return messages
        return [await self._apply_receive_middleware(message, recipient=agent_name) for message in messages]

    async def shutdown(self) -> None:
        """Gracefully shut down the bus and its transport."""
        if self._closed:
            return
        self._closed = True
        await self._transport.shutdown()

    def dead_letters(self) -> list[DeadLetterRecord]:
        """Return dead-lettered messages recorded by the transport."""
        return self._transport.dead_letters()

    async def _emit_signal(self, signal_name: str, **kwargs: Any) -> None:
        if self._signals is None:
            return
        try:
            from lauren_ai import _signals  # noqa: PLC0415

            signal_cls = getattr(_signals, signal_name, None)
            if signal_cls is None:
                return
            await self._signals.emit(signal_cls(**kwargs))
        except Exception:
            return

    async def _emit_error(self, error: Exception, message: AgentMessage | None) -> None:
        for item in self._middleware:
            with contextlib.suppress(Exception):
                await item.on_error(error, message=message)
        if message is None:
            return
        if isinstance(error, MessageValidationError):
            reason = DeadLetterReason.VALIDATION_ERROR
        elif self._closed:
            reason = DeadLetterReason.BUS_CLOSED
        else:
            reason = DeadLetterReason.INTERNAL_ERROR
        self._record_dead_letter(
            DeadLetterRecord(
                message=message,
                reason=reason,
                target=message.to,
                error=str(error),
            )
        )

    async def _apply_receive_middleware(
        self,
        message: AgentMessage,
        *,
        recipient: str,
    ) -> AgentMessage:
        try:
            inbound = message
            for item in self._middleware:
                inbound = await item.on_receive(inbound, recipient=recipient)
            return inbound
        except Exception as exc:
            await self._emit_error(exc, message)
            raise

    def _prepare_request(self, *, target: str, request: AgentMessage) -> AgentMessage:
        if request.to is not None and request.to != target:
            raise MessageValidationError(
                f"Request target mismatch: request.to={request.to!r} does not match target={target!r}",
                field_name="to",
            )
        return replace(request, to=target, expects_response=True)

    def _record_dead_letter(self, record: DeadLetterRecord) -> None:
        if isinstance(self._transport, _DeadLetterRecorder):
            self._transport.record_dead_letter(record)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentMessageBus is shut down")
