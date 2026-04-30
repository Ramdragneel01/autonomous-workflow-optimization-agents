
"""Integration tests for autonomous-workflow-optimization-agents API endpoints."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from api import main as main_module
from api.main import app

client = TestClient(app)


def _assert_error_contract(
    response,
    expected_status: int,
    expected_code: str,
    expected_message: str | None = None,
    expect_details: bool = False,
) -> None:
    """Validate normalized API error contract fields."""

    assert response.status_code == expected_status
    payload = response.json()
    assert isinstance(payload.get("error"), dict)
    assert payload["error"]["code"] == expected_code
    assert isinstance(payload["error"]["message"], str)
    if expected_message is not None:
        assert payload["error"]["message"] == expected_message
    assert isinstance(payload["error"]["request_id"], str)
    assert payload["error"]["request_id"]
    if expect_details:
        assert "details" in payload["error"]


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


def test_probe_alias_endpoints_return_runtime_state():
    """Readiness and probe aliases should be available for deployment checks."""

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    health_alias = client.get("/healthz")
    assert health_alias.status_code == 200
    assert health_alias.json()["status"] == "ok"

    ready_alias = client.get("/readyz")
    assert ready_alias.status_code == 200
    assert ready_alias.json()["status"] == "ready"


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
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

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

    ready_response = client.get("/ready")
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"


def test_phase1_auth_required_contract(monkeypatch):
    """Protected endpoints should require API key with normalized unauthorized errors."""

    _override_settings(monkeypatch, api_key="phase1-secret")

    unauthorized = client.post(
        "/research/run",
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    invalid_key = client.post(
        "/research/run",
        headers={"X-API-Key": "wrong"},
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    _assert_error_contract(
        invalid_key,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    authorized = client.post(
        "/research/run",
        headers={"X-API-Key": "phase1-secret"},
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert authorized.status_code == 200


def test_phase1_error_contract_response():
    """Missing body should return normalized validation error payload."""

    response = client.post("/research/run")
    _assert_error_contract(
        response,
        expected_status=422,
        expected_code="validation_error",
        expected_message="request_validation_failed",
        expect_details=True,
    )


def test_error_responses_include_request_and_security_headers(monkeypatch):
    """Error responses should carry request tracing and baseline security headers."""

    _override_settings(monkeypatch, api_key="header-secret")
    response = client.post(
        "/research/run",
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )

    assert response.status_code == 401
    assert response.headers.get("X-Request-ID")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"


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
    _assert_error_contract(
        second,
        expected_status=429,
        expected_code="rate_limited",
        expected_message="rate_limited",
    )
    assert second.headers.get("Retry-After") == "60"


def test_metrics_requires_api_key_when_configured(monkeypatch):
    """Metrics endpoint should enforce optional API key when configured."""

    _override_settings(monkeypatch, metrics_api_key="metrics-secret")

    unauthorized = client.get("/metrics")
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    authorized = client.get("/metrics", headers={"X-Metrics-Key": "metrics-secret"})
    assert authorized.status_code == 200


def test_rate_limit_uses_forwarded_for_when_enabled(monkeypatch):
    """Rate limit keys should use X-Forwarded-For when trust setting is enabled."""

    _override_settings(monkeypatch, rate_limit_per_minute=1, trust_proxy_headers=True)

    first = client.post(
        "/research/run",
        headers={"X-Forwarded-For": "203.0.113.10"},
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert first.status_code == 200

    different_client = client.post(
        "/research/run",
        headers={"X-Forwarded-For": "198.51.100.2"},
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert different_client.status_code == 200

    same_client_again = client.post(
        "/research/run",
        headers={"X-Forwarded-For": "203.0.113.10"},
        json={"query": "Compare modern API gateway patterns", "max_sources": 4},
    )
    assert same_client_again.status_code == 429


def test_health_includes_hardening_headers():
    """Health endpoint responses should include baseline hardening headers."""

    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in response.headers
