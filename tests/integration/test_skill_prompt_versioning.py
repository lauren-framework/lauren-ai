"""Integration tests for the prompt-versioning skill (Skill 25).

Verifies PromptVersionRegistry lookup, default handling, and A/B routing
consistency per user_id, via direct calls.
"""

import hashlib
import pytest
from dataclasses import dataclass


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


def _make_registry():
    """Create a fresh registry with v1, v2, v3 registered."""
    reg = PromptVersionRegistry()
    reg.register(
        PromptVersion("v1", "You are a helpful assistant.", description="Baseline"), default=True
    )
    reg.register(
        PromptVersion("v2", "You are a concise assistant. Use bullets.", description="Concise")
    )
    reg.register(PromptVersion("v3", "You are a detailed assistant.", description="Verbose"))
    return reg


# ---------------------------------------------------------------------------
# Tests: PromptVersionRegistry
# ---------------------------------------------------------------------------


class TestPromptVersionRegistry:
    def test_register_and_get(self):
        reg = PromptVersionRegistry()
        v = PromptVersion(version="v1", system_prompt="Test prompt")
        reg.register(v, default=True)
        assert reg.get("v1").system_prompt == "Test prompt"

    def test_get_unknown_version_raises(self):
        reg = PromptVersionRegistry()
        with pytest.raises(KeyError):
            reg.get("v99")

    def test_register_sets_default(self):
        reg = PromptVersionRegistry()
        reg.register(PromptVersion("first", "First"), default=True)
        assert reg.get_default().version == "first"

    def test_version_descriptions_stored(self):
        reg = _make_registry()
        assert reg.get("v1").description == "Baseline"
        assert reg.get("v2").description == "Concise"

    def test_register_multiple_versions(self):
        reg = _make_registry()
        for v in ["v1", "v2", "v3"]:
            assert reg.get(v).version == v

    def test_get_default_returns_first_registered_when_no_explicit_default(self):
        reg = PromptVersionRegistry()
        reg.register(PromptVersion("a", "Prompt A"))
        reg.register(PromptVersion("b", "Prompt B"))
        assert reg.get_default().version == "a"

    def test_default_can_be_overridden(self):
        reg = PromptVersionRegistry()
        reg.register(PromptVersion("a", "Prompt A"))
        reg.register(PromptVersion("b", "Prompt B"), default=True)
        assert reg.get_default().version == "b"


# ---------------------------------------------------------------------------
# Tests: A/B selection
# ---------------------------------------------------------------------------


class TestAbSelect:
    def test_ab_select_returns_valid_variant(self):
        reg = _make_registry()
        for uid in ["user1", "user2", "user3", "user4", "user5"]:
            selected = reg.ab_select(uid, ["v1", "v2"])
            assert selected in {"v1", "v2"}

    def test_ab_select_is_deterministic(self):
        reg = _make_registry()
        for uid in ["alice", "bob", "charlie", "dave"]:
            r1 = reg.ab_select(uid, ["v1", "v2"])
            r2 = reg.ab_select(uid, ["v1", "v2"])
            assert r1 == r2, f"Non-deterministic for {uid}"

    def test_ab_select_distributes_users(self):
        reg = _make_registry()
        results = set()
        for i in range(100):
            uid = f"user_{i:04d}"
            results.add(reg.ab_select(uid, ["v1", "v2"]))
        assert len(results) == 2, "Both variants should appear across 100 users"

    def test_ab_select_with_weights(self):
        reg = _make_registry()
        counts = {"v1": 0, "v2": 0}
        for i in range(100):
            uid = f"weighted_user_{i}"
            counts[reg.ab_select(uid, ["v1", "v2"], weights=[0.9, 0.1])] += 1
        assert counts["v1"] > counts["v2"], "v1 should dominate with 90% weight"

    def test_ab_select_three_variants(self):
        reg = _make_registry()
        results = set()
        for i in range(200):
            uid = f"multi_{i}"
            results.add(reg.ab_select(uid, ["v1", "v2", "v3"]))
        assert len(results) == 3, "All three variants should appear"
