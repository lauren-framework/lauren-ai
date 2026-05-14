---
name: prompt-versioning
description: Manages multiple named prompt versions with A/B routing by user-id hash. Use when running controlled prompt experiments, when rolling out a revised system prompt to a percentage of users, or when you need reproducible rollback to a prior prompt version.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Prompt Versioning & A/B Testing

## Pattern

```python
# NOTE: future annotations are allowed, but tool signature types must resolve at import time.
import hashlib
from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    version: str
    system_prompt: str
    description: str = ""


class PromptVersionRegistry:
    def __init__(self):
        self._versions: dict[str, PromptVersion] = {}
        self._default: str | None = None

    def register(self, version: PromptVersion, default: bool = False) -> None:
        self._versions[version.version] = version
        if default or not self._default:
            self._default = version.version

    def get(self, version: str) -> PromptVersion:
        if version not in self._versions:
            raise KeyError(f"Prompt version '{version}' not found")
        return self._versions[version]

    def get_default(self) -> PromptVersion:
        if self._default is None:
            raise RuntimeError("No versions registered")
        return self._versions[self._default]

    def ab_select(
        self,
        user_id: str,
        variants: list[str],
        weights: list[float] | None = None,
    ) -> str:
        """Deterministically select a variant for user_id based on an MD5 bucket.

        The same user_id always maps to the same variant as long as the
        variants list and weights do not change.
        """
        if weights is None:
            weights = [1.0 / len(variants)] * len(variants)
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        cumulative = 0.0
        for variant, weight in zip(variants, weights):
            cumulative += weight * 100
            if bucket < cumulative:
                return variant
        return variants[-1]
```

## Full example

```python
registry = PromptVersionRegistry()
registry.register(
    PromptVersion(
        version="v1",
        system_prompt="You are a helpful assistant.",
        description="Baseline prompt",
    ),
    default=True,
)
registry.register(
    PromptVersion(
        version="v2",
        system_prompt="You are a concise, helpful assistant. Prefer bullet points.",
        description="Concise variant",
    ),
)

# Route a user to one of the two variants (50/50 split)
def get_system_for_user(user_id: str) -> str:
    version_name = registry.ab_select(user_id, ["v1", "v2"])
    return registry.get(version_name).system_prompt
```

## Using the selected prompt in an agent

```python
from lauren_ai import agent, AgentRunnerBase, LLMConfig

def make_agent_for_user(user_id: str):
    system = get_system_for_user(user_id)

    @agent(model="claude-haiku-4-5", system=system)
    class UserAgent: ...

    return UserAgent
```

## Weighted A/B split

```python
# 80% control (v1), 20% experimental (v2)
version_name = registry.ab_select(
    user_id,
    variants=["v1", "v2"],
    weights=[0.8, 0.2],
)
```

## Tracking which version was used

```python
from dataclasses import dataclass

@dataclass
class VersionedResponse:
    content: str
    version_used: str
    user_id: str

async def run_versioned(user_id: str, message: str, runner) -> VersionedResponse:
    version_name = registry.ab_select(user_id, ["v1", "v2"])
    version = registry.get(version_name)
    # Create agent dynamically or cache by version name
    result = await runner.run(agent_for_version(version), message)
    return VersionedResponse(
        content=result.content,
        version_used=version_name,
        user_id=user_id,
    )
```

## Pitfalls

- The MD5 bucket is stable as long as you do not change the `variants` list
  order or the weights.  Adding a new variant re-buckets some users — plan
  accordingly.
- Do not use MD5 for security-sensitive bucketing; it is fine for load
  distribution but is not cryptographically random.
- Store `version_used` alongside model responses in your database so you can
  segment analytics by prompt version.
