"""Integration tests for the prompt-versioning skill (Skill 25).

Verifies PromptVersionRegistry lookup, default handling, and A/B routing
consistency per user_id.
"""
import hashlib
import pytest

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------


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
        if weights is None:
            weights = [1.0 / len(variants)] * len(variants)
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        cumulative = 0.0
        for variant, weight in zip(variants, weights):
            cumulative += weight * 100
            if bucket < cumulative:
                return variant
        return variants[-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populated_registry() -> PromptVersionRegistry:
    registry = PromptVersionRegistry()
    registry.register(
        PromptVersion(version="v1", system_prompt="You are a helpful assistant.", description="Baseline"),
        default=True,
    )
    registry.register(
        PromptVersion(version="v2", system_prompt="You are a concise assistant. Use bullets.", description="Concise"),
    )
    registry.register(
        PromptVersion(version="v3", system_prompt="You are a detailed assistant.", description="Verbose"),
    )
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromptVersionRegistry:
    def test_register_and_get(self):
        registry = _populated_registry()
        v1 = registry.get("v1")
        assert v1.version == "v1"
        assert "helpful" in v1.system_prompt

    def test_get_unknown_version_raises_key_error(self):
        registry = _populated_registry()
        with pytest.raises(KeyError, match="v99"):
            registry.get("v99")

    def test_first_registered_is_default(self):
        registry = PromptVersionRegistry()
        registry.register(PromptVersion(version="first", system_prompt="First"))
        registry.register(PromptVersion(version="second", system_prompt="Second"))
        assert registry.get_default().version == "first"

    def test_explicit_default_overrides_first(self):
        registry = PromptVersionRegistry()
        registry.register(PromptVersion(version="a", system_prompt="A"))
        registry.register(PromptVersion(version="b", system_prompt="B"), default=True)
        assert registry.get_default().version == "b"

    def test_get_default_raises_when_empty(self):
        registry = PromptVersionRegistry()
        with pytest.raises(RuntimeError, match="No versions"):
            registry.get_default()

    def test_version_descriptions_stored(self):
        registry = _populated_registry()
        assert registry.get("v1").description == "Baseline"
        assert registry.get("v2").description == "Concise"


class TestAbSelect:
    def test_ab_select_returns_valid_variant(self):
        registry = _populated_registry()
        for uid in ["user1", "user2", "user3", "user4", "user5"]:
            selected = registry.ab_select(uid, ["v1", "v2"])
            assert selected in {"v1", "v2"}

    def test_ab_select_is_deterministic(self):
        """Same user_id always maps to the same variant."""
        registry = _populated_registry()
        for uid in ["alice", "bob", "charlie", "dave"]:
            first = registry.ab_select(uid, ["v1", "v2"])
            second = registry.ab_select(uid, ["v1", "v2"])
            assert first == second, f"Non-deterministic for user_id={uid}"

    def test_ab_select_distributes_users(self):
        """With 100 distinct user IDs and equal weights, both variants appear."""
        registry = _populated_registry()
        results = set()
        for i in range(100):
            uid = f"user_{i:04d}"
            results.add(registry.ab_select(uid, ["v1", "v2"]))
        assert len(results) == 2, "Both variants should appear across 100 users"

    def test_ab_select_with_weights(self):
        """Heavily weighted towards v1; v2 should appear but rarely."""
        registry = _populated_registry()
        counts = {"v1": 0, "v2": 0}
        for i in range(100):
            uid = f"weighted_user_{i}"
            selected = registry.ab_select(uid, ["v1", "v2"], weights=[0.9, 0.1])
            counts[selected] += 1
        assert counts["v1"] > counts["v2"], "v1 should dominate with 90% weight"

    def test_ab_select_three_variants(self):
        registry = _populated_registry()
        results = set()
        for i in range(200):
            uid = f"multi_{i}"
            results.add(registry.ab_select(uid, ["v1", "v2", "v3"]))
        assert len(results) == 3, "All three variants should appear"
