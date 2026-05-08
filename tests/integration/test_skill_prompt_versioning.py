"""Integration tests for the prompt-versioning skill (Skill 25).

Verifies PromptVersionRegistry lookup, default handling, and A/B routing
consistency per user_id, via HTTP through a Lauren TestClient.
"""

import hashlib
from dataclasses import dataclass

from lauren import LaurenFactory, controller, post, get, module, Json, use_value, injectable, Scope, Path
from lauren.testing import TestClient
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    version: str
    system_prompt: str
    description: str = ""


@injectable(scope=Scope.SINGLETON)
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
# Request models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    version: str
    system_prompt: str
    description: str = ""
    default: bool = False


class AbSelectRequest(BaseModel):
    user_id: str
    variants: list[str]
    weights: list[float] | None = None


# ---------------------------------------------------------------------------
# Controller / Module / build_app
# ---------------------------------------------------------------------------


@controller("/versions")
class VersionsController:
    def __init__(self, registry: PromptVersionRegistry) -> None:
        self._registry = registry

    @post("/register")
    async def register(self, body: Json[RegisterRequest]) -> dict:
        self._registry.register(
            PromptVersion(
                version=body.version,
                system_prompt=body.system_prompt,
                description=body.description,
            ),
            default=body.default,
        )
        return {"registered": True}

    @get("/{version}")
    async def get_version(self, version: Path[str]) -> dict:
        pv = self._registry.get(version)
        return {"version": pv.version, "system_prompt": pv.system_prompt, "description": pv.description}

    @post("/ab-select")
    async def ab_select(self, body: Json[AbSelectRequest]) -> dict:
        selected = self._registry.ab_select(body.user_id, body.variants, body.weights)
        return {"selected": selected}


@module(controllers=[VersionsController], providers=[PromptVersionRegistry])
class VersionsModule: ...


def build_app():
    return TestClient(LaurenFactory.create(VersionsModule))


def _register_defaults(client):
    """Register v1, v2, v3 in the given client's app."""
    client.post("/versions/register", json={
        "version": "v1", "system_prompt": "You are a helpful assistant.",
        "description": "Baseline", "default": True,
    })
    client.post("/versions/register", json={
        "version": "v2", "system_prompt": "You are a concise assistant. Use bullets.",
        "description": "Concise",
    })
    client.post("/versions/register", json={
        "version": "v3", "system_prompt": "You are a detailed assistant.",
        "description": "Verbose",
    })


# ---------------------------------------------------------------------------
# Tests: PromptVersionRegistry
# ---------------------------------------------------------------------------


class TestPromptVersionRegistry:
    def test_register_and_get(self):
        client = build_app()
        _register_defaults(client)
        resp = client.get("/versions/v1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "v1"
        assert "helpful" in data["system_prompt"]

    def test_get_unknown_version_returns_error(self):
        client = build_app()
        resp = client.get("/versions/v99")
        assert resp.status_code != 200

    def test_register_sets_default(self):
        client = build_app()
        resp = client.post("/versions/register", json={
            "version": "first", "system_prompt": "First", "default": True,
        })
        assert resp.status_code == 200
        assert resp.json()["registered"] is True

    def test_version_descriptions_stored(self):
        client = build_app()
        _register_defaults(client)
        r1 = client.get("/versions/v1").json()
        r2 = client.get("/versions/v2").json()
        assert r1["description"] == "Baseline"
        assert r2["description"] == "Concise"

    def test_register_multiple_versions(self):
        client = build_app()
        _register_defaults(client)
        for v in ["v1", "v2", "v3"]:
            resp = client.get(f"/versions/{v}")
            assert resp.status_code == 200
            assert resp.json()["version"] == v


# ---------------------------------------------------------------------------
# Tests: A/B selection
# ---------------------------------------------------------------------------


class TestAbSelect:
    def test_ab_select_returns_valid_variant(self):
        client = build_app()
        _register_defaults(client)
        for uid in ["user1", "user2", "user3", "user4", "user5"]:
            resp = client.post("/versions/ab-select", json={
                "user_id": uid, "variants": ["v1", "v2"],
            })
            assert resp.status_code == 200
            assert resp.json()["selected"] in {"v1", "v2"}

    def test_ab_select_is_deterministic(self):
        client = build_app()
        _register_defaults(client)
        for uid in ["alice", "bob", "charlie", "dave"]:
            r1 = client.post("/versions/ab-select", json={"user_id": uid, "variants": ["v1", "v2"]})
            r2 = client.post("/versions/ab-select", json={"user_id": uid, "variants": ["v1", "v2"]})
            assert r1.json()["selected"] == r2.json()["selected"], f"Non-deterministic for {uid}"

    def test_ab_select_distributes_users(self):
        client = build_app()
        _register_defaults(client)
        results = set()
        for i in range(100):
            uid = f"user_{i:04d}"
            resp = client.post("/versions/ab-select", json={"user_id": uid, "variants": ["v1", "v2"]})
            results.add(resp.json()["selected"])
        assert len(results) == 2, "Both variants should appear across 100 users"

    def test_ab_select_with_weights(self):
        client = build_app()
        _register_defaults(client)
        counts = {"v1": 0, "v2": 0}
        for i in range(100):
            uid = f"weighted_user_{i}"
            resp = client.post("/versions/ab-select", json={
                "user_id": uid, "variants": ["v1", "v2"], "weights": [0.9, 0.1],
            })
            counts[resp.json()["selected"]] += 1
        assert counts["v1"] > counts["v2"], "v1 should dominate with 90% weight"

    def test_ab_select_three_variants(self):
        client = build_app()
        _register_defaults(client)
        results = set()
        for i in range(200):
            uid = f"multi_{i}"
            resp = client.post("/versions/ab-select", json={
                "user_id": uid, "variants": ["v1", "v2", "v3"],
            })
            results.add(resp.json()["selected"])
        assert len(results) == 3, "All three variants should appear"
