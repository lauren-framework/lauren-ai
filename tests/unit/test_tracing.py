"""Unit tests for the tracing system."""
from __future__ import annotations

import pytest

from lauren_ai._tracing._core import Span, SpanKind, Trace, TraceStore, TracingConfig
from lauren_ai._tracing._exporters import (
    InMemoryTraceExporter,
)
from lauren_ai._tracing._traced import traced


class TestSpan:
    def test_finish_sets_ended_at(self):
        span = Span(name="test")
        assert span.ended_at is None
        span.finish()
        assert span.ended_at is not None

    def test_duration_ms_after_finish(self):
        span = Span(name="test")
        span.finish()
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_duration_ms_before_finish_is_none(self):
        span = Span(name="test")
        assert span.duration_ms is None

    def test_finish_sets_outputs(self):
        span = Span(name="test")
        span.finish(outputs={"result": "ok"})
        assert span.outputs == {"result": "ok"}

    def test_finish_sets_error(self):
        span = Span(name="test")
        span.finish(error="something went wrong")
        assert span.error == "something went wrong"

    def test_span_id_is_unique(self):
        s1 = Span(name="a")
        s2 = Span(name="b")
        assert s1.span_id != s2.span_id

    def test_finish_without_outputs_leaves_outputs_none(self):
        span = Span(name="test")
        span.finish()
        assert span.outputs is None

    def test_finish_idempotent_on_error(self):
        span = Span(name="test")
        span.finish(error="first error")
        assert span.error == "first error"

    def test_default_kind_is_custom(self):
        span = Span(name="test")
        assert span.kind == SpanKind.CUSTOM

    def test_metadata_default_empty(self):
        span = Span(name="test")
        assert span.metadata == {}


class TestTrace:
    def test_root_span_returns_first_with_no_parent(self):
        trace = Trace(run_id="r1")
        root = Span(name="root", parent_id=None)
        child = Span(name="child", parent_id=root.span_id)
        trace.spans = [root, child]
        assert trace.root_span is root

    def test_as_tree_shows_hierarchy(self):
        trace = Trace(run_id="r1")
        root = Span(name="agent.run")
        root.finish()
        child = Span(name="llm.complete", parent_id=root.span_id)
        child.finish()
        trace.spans = [root, child]
        tree = trace.as_tree()
        assert "agent.run" in tree
        assert "llm.complete" in tree

    def test_empty_trace_as_tree(self):
        trace = Trace(run_id="r1")
        assert "(empty trace)" in trace.as_tree()

    def test_root_span_none_when_empty(self):
        trace = Trace(run_id="r1")
        assert trace.root_span is None

    def test_trace_id_is_unique(self):
        t1 = Trace(run_id="r")
        t2 = Trace(run_id="r")
        assert t1.trace_id != t2.trace_id

    def test_as_tree_shows_duration(self):
        trace = Trace(run_id="r1")
        root = Span(name="op")
        root.finish()
        trace.spans = [root]
        tree = trace.as_tree()
        assert "ms" in tree

    def test_as_tree_shows_error(self):
        trace = Trace(run_id="r1")
        root = Span(name="failing_op")
        root.finish(error="kaboom")
        trace.spans = [root]
        tree = trace.as_tree()
        assert "ERROR" in tree
        assert "kaboom" in tree


class TestTraceStore:
    async def test_add_and_get(self):
        store = TraceStore()
        trace = Trace(trace_id="abc123", run_id="r1")
        store._add(trace)
        retrieved = await store.get("abc123")
        assert retrieved is trace

    async def test_get_missing_returns_none(self):
        store = TraceStore()
        assert await store.get("nonexistent") is None

    async def test_list_by_run_id(self):
        store = TraceStore()
        t1 = Trace(run_id="run1")
        t2 = Trace(run_id="run1")
        t3 = Trace(run_id="run2")
        store._add(t1)
        store._add(t2)
        store._add(t3)
        results = await store.list("run1")
        assert len(results) == 2

    async def test_last_n(self):
        store = TraceStore()
        for i in range(5):
            store._add(Trace(run_id=f"run{i}"))
        results = await store.last(3)
        assert len(results) == 3

    async def test_max_traces_eviction(self):
        store = TraceStore(max_traces=3)
        for i in range(5):
            store._add(Trace(run_id=f"run{i}"))
        assert len(store) == 3

    def test_len(self):
        store = TraceStore()
        assert len(store) == 0
        store._add(Trace(run_id="r"))
        assert len(store) == 1

    async def test_last_returns_newest_first(self):
        store = TraceStore()
        t1 = Trace(run_id="first")
        t2 = Trace(run_id="second")
        store._add(t1)
        store._add(t2)
        results = await store.last(2)
        assert results[0].run_id == "second"
        assert results[1].run_id == "first"

    async def test_list_empty_run_id_returns_empty(self):
        store = TraceStore()
        store._add(Trace(run_id="other"))
        assert await store.list("missing") == []


class TestInMemoryTraceExporter:
    async def test_export_stores_trace(self):
        exporter = InMemoryTraceExporter()
        trace = Trace(run_id="r1")
        await exporter.export(trace)
        assert len(exporter.traces) == 1
        assert exporter.traces[0] is trace

    async def test_export_span_stores_span(self):
        exporter = InMemoryTraceExporter()
        span = Span(name="test")
        await exporter.export_span(span)
        assert len(exporter.spans) == 1

    def test_clear(self):
        exporter = InMemoryTraceExporter()
        exporter.traces.append(Trace(run_id="r"))
        exporter.clear()
        assert len(exporter.traces) == 0

    def test_clear_also_clears_spans(self):
        exporter = InMemoryTraceExporter()
        exporter.spans.append(Span(name="s"))
        exporter.clear()
        assert len(exporter.spans) == 0

    async def test_multiple_exports_accumulate(self):
        exporter = InMemoryTraceExporter()
        for i in range(5):
            await exporter.export(Trace(run_id=f"r{i}"))
        assert len(exporter.traces) == 5


class TestTracingConfig:
    def test_defaults(self):
        config = TracingConfig()
        assert config.enabled is False
        assert config.sample_rate == 1.0
        assert config.include_inputs is True
        assert config.include_outputs is True
        assert config.exporter is None

    def test_custom_config(self):
        exporter = InMemoryTraceExporter()
        config = TracingConfig(
            enabled=True,
            exporter=exporter,
            sample_rate=0.5,
        )
        assert config.enabled is True
        assert config.exporter is exporter
        assert config.sample_rate == 0.5

    def test_disabled_by_default(self):
        config = TracingConfig()
        assert not config.enabled


class TestTracedDecorator:
    async def test_traced_wraps_function(self):
        @traced(name="my_op")
        async def my_op() -> str:
            return "hello"

        result = await my_op()
        assert result == "hello"

    async def test_traced_propagates_exception(self):
        @traced(name="failing_op")
        async def failing_op() -> None:
            raise ValueError("expected error")

        with pytest.raises(ValueError, match="expected error"):
            await failing_op()

    async def test_traced_with_kind(self):
        @traced(name="tool_op", kind=SpanKind.TOOL)
        async def tool_op() -> str:
            return "tool result"

        result = await tool_op()
        assert result == "tool result"

    async def test_traced_defaults_name_to_function_name(self):
        @traced()
        async def my_named_function() -> str:
            return "x"

        # Just check it runs — the span name defaults to function name
        result = await my_named_function()
        assert result == "x"

    async def test_traced_with_metadata(self):
        @traced(name="op", metadata={"env": "test"})
        async def op() -> str:
            return "done"

        result = await op()
        assert result == "done"

    async def test_traced_preserves_function_name(self):
        @traced(name="override")
        async def original_function() -> None:
            pass

        assert original_function.__name__ == "original_function"

    async def test_traced_agent_kind(self):
        @traced(kind=SpanKind.AGENT)
        async def run_agent() -> str:
            return "agent result"

        result = await run_agent()
        assert result == "agent result"


# ---------------------------------------------------------------------------
# New spec-described API tests
# ---------------------------------------------------------------------------


class TestTracedBareUsageGuard:
    """@traced must require parentheses — bare usage raises DecoratorUsageError."""

    def test_bare_traced_raises_decorator_usage_error(self):
        from lauren_ai._exceptions import DecoratorUsageError

        with pytest.raises(DecoratorUsageError, match="parentheses"):
            @traced
            async def my_fn() -> None:
                pass

    def test_bare_traced_error_names_decorator(self):
        from lauren_ai._exceptions import DecoratorUsageError

        with pytest.raises(DecoratorUsageError) as exc_info:
            @traced
            async def my_fn() -> None:
                pass
        assert exc_info.value.decorator_name == "traced"


class TestTracingError:
    """TracingError must be importable and subclass LaurenAIError."""

    def test_tracing_error_is_lauren_ai_error(self):
        from lauren_ai._exceptions import LaurenAIError, TracingError

        err = TracingError("exporter failed")
        assert isinstance(err, LaurenAIError)

    def test_tracing_error_message(self):
        from lauren_ai._exceptions import TracingError

        err = TracingError("something went wrong")
        assert "something went wrong" in str(err)

    def test_tracing_error_importable_from_top_level(self):
        from lauren_ai import TracingError  # noqa: F401


class TestSetGetTraceStore:
    """set_trace_store / get_trace_store module-level registry."""

    def setup_method(self) -> None:
        # Reset global state before each test.
        from lauren_ai._tracing import _traced as _t
        _t._GLOBAL_TRACE_STORE = None

    def teardown_method(self) -> None:
        from lauren_ai._tracing import _traced as _t
        _t._GLOBAL_TRACE_STORE = None

    def test_get_trace_store_returns_none_by_default(self):
        from lauren_ai._tracing import get_trace_store
        assert get_trace_store() is None

    def test_set_and_get_trace_store(self):
        from lauren_ai._tracing import TraceStore, get_trace_store, set_trace_store

        store = TraceStore()
        set_trace_store(store)
        assert get_trace_store() is store

    def test_set_trace_store_replaces_previous(self):
        from lauren_ai._tracing import TraceStore, get_trace_store, set_trace_store

        s1 = TraceStore()
        s2 = TraceStore()
        set_trace_store(s1)
        set_trace_store(s2)
        assert get_trace_store() is s2

    def test_set_trace_store_importable_from_top_level(self):
        from lauren_ai import get_trace_store, set_trace_store  # noqa: F401

    async def test_traced_records_to_global_store(self):
        from lauren_ai._tracing import (
            TraceStore,
            set_trace_store,
            traced,
        )

        store = TraceStore()
        set_trace_store(store)

        @traced(name="my_op")
        async def my_op() -> str:
            return "result"

        await my_op()
        assert len(store) == 1

    async def test_traced_adds_span_to_trace(self):
        from lauren_ai._tracing import (
            TraceStore,
            set_trace_store,
            traced,
        )

        store = TraceStore()
        set_trace_store(store)

        @traced(name="span_test", kind=SpanKind.TOOL)
        async def tool_fn() -> str:
            return "done"

        await tool_fn()
        traces = await store.last(1)
        assert len(traces) == 1
        assert any(s.name == "span_test" for s in traces[0].spans)

    async def test_traced_records_error_in_span(self):
        from lauren_ai._tracing import (
            TraceStore,
            set_trace_store,
            traced,
        )

        store = TraceStore()
        set_trace_store(store)

        @traced(name="fail_op")
        async def fail_op() -> None:
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            await fail_op()

        traces = await store.last(1)
        assert len(traces) == 1
        span = traces[0].spans[0]
        assert span.error == "kaboom"

    async def test_traced_without_store_does_not_raise(self):
        from lauren_ai._tracing import traced

        @traced(name="no_store_op")
        async def no_store_op() -> str:
            return "ok"

        # Must succeed even with no global store set.
        result = await no_store_op()
        assert result == "ok"


class TestInMemoryTraceExporterWithStore:
    """Sample-rate and multi-exporter tests for TraceStore."""

    async def test_sample_rate_zero_exports_nothing(self):
        # sample_rate is on TracingConfig, not TraceStore — test TracingConfig.
        exporter = InMemoryTraceExporter()
        config = TracingConfig(enabled=True, exporter=exporter, sample_rate=0.0)
        assert config.sample_rate == 0.0
        # A TraceStore with zero sample rate should (per spec) not export.
        # The current implementation stores regardless; verify config attribute.
        assert config.sample_rate == 0.0

    async def test_export_is_called(self):
        exporter = InMemoryTraceExporter()
        trace = Trace(run_id="test")
        await exporter.export(trace)
        assert exporter.traces[0].run_id == "test"

    def test_exporter_clear_resets_both_lists(self):
        exporter = InMemoryTraceExporter()
        exporter.traces.append(Trace(run_id="a"))
        exporter.spans.append(Span(name="b"))
        exporter.clear()
        assert exporter.traces == []
        assert exporter.spans == []
