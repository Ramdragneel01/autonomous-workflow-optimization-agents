
"""Integration tests for agentic-research-assistant API endpoints."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from api import main as main_module
from api.main import app

client = TestClient(app)


def _override_settings(monkeypatch, **changes):
    """Apply temporary runtime setting overrides for API security tests."""

    monkeypatch.setattr(main_module, "settings", replace(main_module.settings, **changes))


@pytest.fixture(autouse=True)
def reset_runtime_state(monkeypatch):
    """Reset limiter and auth settings between tests for deterministic behavior."""

    main_module.limiter.clear()
    _override_settings(monkeypatch, api_key="", rate_limit_per_minute=90)

    yield

    main_module.limiter.clear()


def test_health_endpoint_reports_runtime_limits():
    """Health endpoint should return status and key runtime constraints."""

    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["rate_limit_per_minute"] > 0
    assert payload["max_query_length"] >= 100


def test_research_run_returns_answer_contract():
    """Synchronous research endpoint should return answer, critique, and traces."""

    response = client.post(
        "/research/run",
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["query"]
    assert payload["summary"]
    assert isinstance(payload["sources"], list)
    assert isinstance(payload["trace"], list)
    assert payload["trace_count"] == len(payload["trace"])


def test_research_sse_stream_returns_answer_event():
    """Streaming endpoint should emit event-stream payload with final answer event."""

    response = client.get("/research", params={"query": "What is zero trust architecture?", "max_sources": 4})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: answer" in response.text


def test_research_run_requires_api_key_when_configured(monkeypatch):
    """Synchronous research endpoint should require API key when configured."""

    _override_settings(monkeypatch, api_key="secret-key")

    unauthorized = client.post(
        "/research/run",
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/research/run",
        headers={"X-API-Key": "secret-key"},
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert authorized.status_code == 200


def test_health_is_public_when_api_key_enabled(monkeypatch):
    """Health endpoint should remain public for uptime checks."""

    _override_settings(monkeypatch, api_key="secret-key")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_run_rate_limit_returns_429(monkeypatch):
    """Synchronous research endpoint should return 429 after limit is reached."""

    _override_settings(monkeypatch, rate_limit_per_minute=1)

    first = client.post(
        "/research/run",
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert first.status_code == 200

    second = client.post(
        "/research/run",
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert second.status_code == 429
