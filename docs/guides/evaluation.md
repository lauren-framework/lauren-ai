# Evaluation

`lauren-ai` ships a lightweight evaluation framework in `lauren_ai.eval` for
measuring agent accuracy, tool-call trajectories, and performance.

---

## Core types

```python
from lauren_ai.eval import (
    EvalExample,
    EvalDataset,
    EvalResult,
    EvalReport,
    AccuracyEval,
    TrajectoryEval,
    PerformanceEval,
)
```

### EvalExample

A single test case:

```python
EvalExample(
    input="What is the capital of France?",
    expected="Paris",                          # for AccuracyEval
    expected_tools=["web_search"],             # for TrajectoryEval
    metadata={"category": "geography"},        # for filtering/grouping
)
```

`expected` and `expected_tools` are both optional — either or both can be
provided depending on which evaluator you use.

### EvalDataset

A collection of examples:

```python
dataset = EvalDataset(
    examples=[
        EvalExample(input="2 + 2?", expected="4"),
        EvalExample(input="Capital of Japan?", expected="Tokyo"),
    ],
    name="basic_qa",
)

print(f"{len(dataset)} examples")
for example in dataset:
    print(example.input)
```

### EvalReport

Returned by every evaluator's `run()` method:

```python
report.pass_rate          # float — fraction of examples that passed (0.0–1.0)
report.avg_latency_ms     # float — average response latency
report.avg_score          # float | None — average numeric score
report.summary()          # str — human-readable summary

# Assert in CI:
report.assert_pass_rate(min_pass_rate=0.9)
```

---

## AccuracyEval

Checks whether the agent's output matches the expected answer.

```python
from lauren_ai.eval import AccuracyEval, EvalDataset, EvalExample
from lauren_ai.testing import AgentTestClient

dataset = EvalDataset([
    EvalExample(input="What is 6 * 7?", expected="42"),
    EvalExample(input="Capital of France?", expected="Paris"),
    EvalExample(input="Translate 'hello' to Spanish.", expected="hola"),
], name="qa_suite")

evaluator = AccuracyEval(
    exact_match=False,   # True = case-insensitive equality; False = substring match
    name="qa_accuracy",
)

report = await evaluator.run(client, dataset)
print(report.summary())
report.assert_pass_rate(min_pass_rate=0.9)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exact_match` | `bool` | `False` | When `True`, checks exact string equality (case-insensitive); when `False`, checks that `expected` is a substring of the actual output |
| `name` | `str` | `"accuracy"` | Label for the report |

The `agent_client` argument to `run()` can be:
- An `AgentTestClient` instance (has a `run()` method)
- Any async callable: `async def f(message: str) -> Any`
- Any object with a `run()` method returning an `AgentResponse`

---

## TrajectoryEval

Verifies that the agent called the correct tools, in the expected order.

```python
from lauren_ai.eval import TrajectoryEval, EvalDataset, EvalExample

dataset = EvalDataset([
    EvalExample(
        input="Search for recent news about climate change and summarise it.",
        expected_tools=["web_search", "summarise_text"],
    ),
    EvalExample(
        input="What is 2 + 2?",
        expected_tools=[],  # no tools should be called
    ),
], name="trajectory_suite")

evaluator = TrajectoryEval(
    strict_order=True,   # True = order must match exactly; False = set membership
    name="tool_trajectory",
)

report = await evaluator.run(client, dataset)
print(report.summary())
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strict_order` | `bool` | `True` | When `True`, actual tool names must match `expected_tools` in exact order; when `False`, only set membership is checked |
| `name` | `str` | `"trajectory"` | Label for the report |

`EvalResult.actual` contains the stringified list of tool names that were
called, useful for debugging failures:

```python
for result in report.results:
    if not result.passed:
        print(f"Input: {result.example.input}")
        print(f"Expected tools: {result.example.expected_tools}")
        print(f"Actual tools:   {result.actual}")
```

---

## PerformanceEval

Measures response latency and optionally enforces a maximum:

```python
from lauren_ai.eval import PerformanceEval, EvalDataset, EvalExample

dataset = EvalDataset([
    EvalExample(input="What is 2 + 2?"),
    EvalExample(input="Explain quantum entanglement in one sentence."),
], name="perf_suite")

evaluator = PerformanceEval(
    max_latency_ms=5000,   # pass only if avg latency <= 5 seconds
    name="latency",
)

report = await evaluator.run(client, dataset)
print(f"Avg latency: {report.avg_latency_ms:.0f}ms")
print(f"Pass rate: {report.pass_rate:.0%}")

# EvalResult.score holds total token count when available
for result in report.results:
    print(f"{result.example.input[:40]}: {result.latency_ms:.0f}ms, tokens={result.score}")
```

Set `max_latency_ms=None` to collect latency data without enforcing a limit.

---

## Combining evaluators

Run multiple evaluators over the same dataset and combine the results:

```python
from lauren_ai.eval import AccuracyEval, TrajectoryEval, PerformanceEval

dataset = EvalDataset([
    EvalExample(
        input="Search for today's weather in London and tell me if I need an umbrella.",
        expected="umbrella",
        expected_tools=["web_search"],
    ),
], name="combined")

acc_report = await AccuracyEval(exact_match=False).run(client, dataset)
traj_report = await TrajectoryEval(strict_order=False).run(client, dataset)
perf_report = await PerformanceEval(max_latency_ms=8000).run(client, dataset)

print(acc_report.summary())
print(traj_report.summary())
print(perf_report.summary())

acc_report.assert_pass_rate(0.9)
traj_report.assert_pass_rate(1.0)
perf_report.assert_pass_rate(1.0)
```

---

## Using evaluators in pytest

```python
import pytest
from lauren_ai.eval import AccuracyEval, EvalDataset, EvalExample
from lauren_ai.testing import AgentTestClient
from lauren_ai._transport import Completion, TokenUsage

@pytest.fixture()
def client():
    from lauren_ai import LLMConfig
    cfg, mock = LLMConfig.for_testing()
    # ... build and return AgentTestClient

@pytest.mark.asyncio
async def test_accuracy(client):
    client.mock.queue_response(Completion(
        id="1", model="mock", content="42",
        tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=2),
    ))
    dataset = EvalDataset([EvalExample(input="6 * 7?", expected="42")])
    report = await AccuracyEval().run(client, dataset)
    report.assert_pass_rate(1.0)
```
