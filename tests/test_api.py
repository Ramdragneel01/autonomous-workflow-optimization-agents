
"""Integration tests for agentic-research-assistant API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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
