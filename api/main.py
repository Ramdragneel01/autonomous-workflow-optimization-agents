
"""FastAPI API exposing a streaming research endpoint over SSE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.graph import build_graph, run_research, stream_research


@dataclass(frozen=True)
class Settings:
    """Runtime settings for API security controls and execution defaults."""

    app_name: str
    app_version: str
    cors_origins: list[str]
    rate_limit_per_minute: int
    max_query_length: int
    default_max_sources: int


class InMemoryRateLimiter:
    """Sliding-window limiter used to protect research endpoints from abuse."""

    def __init__(self, window_seconds: int = 60) -> None:
        """Initialize in-memory limiter state."""

        self._window_seconds = window_seconds
        self._store: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> bool:
        """Return True when current request is within configured rate limit."""

        now = perf_counter()
        with self._lock:
            queue = self._store.setdefault(key, deque())
            cutoff = now - self._window_seconds
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= limit:
                return False
            queue.append(now)
        return True


class HealthResponse(BaseModel):
    """Health payload used by uptime checks and runtime diagnostics."""

    status: str
    timestamp: str
    app_version: str
    rate_limit_per_minute: int
    max_query_length: int


class ResearchRequest(BaseModel):
    """Request body for synchronous research execution endpoint."""

    query: str = Field(min_length=3, max_length=800)
    max_sources: int = Field(default=5, ge=1, le=8)


class ResearchAnswer(BaseModel):
    """Normalized final answer payload returned by research endpoints."""

    query: str
    summary: str
    critique: dict[str, object]
    sources: list[dict[str, str]]
    trace: list[dict[str, object]]
    trace_count: int
    duration_ms: float
    request_id: str


def _load_settings() -> Settings:
    """Read runtime settings from environment with safe defaults."""

    origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:4175,http://localhost:4175")
    return Settings(
        app_name=os.getenv("APP_NAME", "agentic-research-assistant"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        cors_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "90")),
        max_query_length=int(os.getenv("MAX_QUERY_LENGTH", "800")),
        default_max_sources=int(os.getenv("DEFAULT_MAX_SOURCES", "5")),
    )


def _guard_query(query: str, max_query_length: int) -> str:
    """Validate and sanitize user query for safe agent execution."""

    sanitized = re.sub(r"\s+", " ", query).strip()
    if len(sanitized) < 3:
        raise HTTPException(status_code=422, detail="Query must contain at least 3 characters")
    if len(sanitized) > max_query_length:
        raise HTTPException(status_code=422, detail=f"Query exceeds max length {max_query_length}")

    blocked_patterns = [r"ignore\s+previous\s+instructions", r"system\s+prompt", r"bypass\s+guardrails"]
    if any(re.search(pattern, sanitized, flags=re.IGNORECASE) for pattern in blocked_patterns):
        raise HTTPException(status_code=400, detail="Query contains disallowed instruction pattern")

    return sanitized


def _request_key(request: Request) -> str:
    """Build a rate-limit key from request client information."""

    return request.client.host if request.client else "unknown"


settings = _load_settings()
compiled_graph = build_graph()
limiter = InMemoryRateLimiter(window_seconds=60)

REQUEST_COUNTER = Counter(
    "research_api_requests_total",
    "Research API request count",
    labelnames=("method", "endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "research_api_request_duration_seconds",
    "Research API request duration in seconds",
    labelnames=("method", "endpoint"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach security headers, request id, and request-level metrics."""

    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    started_at = perf_counter()

    response = await call_next(request)

    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(perf_counter() - started_at)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=()"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured API errors including request id for debugging."""

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc.detail),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.get("/health")
def health() -> HealthResponse:
    """Return service health for monitoring and deployment checks."""

    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=settings.app_version,
        rate_limit_per_minute=settings.rate_limit_per_minute,
        max_query_length=settings.max_query_length,
    )


@app.post("/research/run", response_model=ResearchAnswer)
def research_run(request: Request, payload: ResearchRequest) -> ResearchAnswer:
    """Run research workflow synchronously for automation and integration tests."""

    if not limiter.allow(_request_key(request), settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    query = _guard_query(payload.query, settings.max_query_length)
    result = run_research(query=query, max_sources=payload.max_sources, compiled_graph=compiled_graph)
    request_id = str(request.state.request_id)
    return ResearchAnswer(
        query=result["query"],
        summary=result["summary"],
        critique=result["critique"],
        sources=result["sources"],
        trace=result["trace"],
        trace_count=len(result["trace"]),
        duration_ms=float(result["duration_ms"]),
        request_id=request_id,
    )


@app.get("/research")
def research(
    request: Request,
    query: str = Query(..., min_length=3, max_length=800),
    max_sources: int = Query(default=settings.default_max_sources, ge=1, le=8),
) -> EventSourceResponse:
    """Stream agent-by-agent execution trace and final answer for a query."""

    if not limiter.allow(_request_key(request), settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    sanitized_query = _guard_query(query, settings.max_query_length)
    request_id = str(request.state.request_id)

    async def event_generator():
        yield {
            "event": "metadata",
            "data": json.dumps(
                {
                    "request_id": request_id,
                    "query": sanitized_query,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=True,
            ),
        }

        try:
            for item in stream_research(sanitized_query, max_sources=max_sources, compiled_graph=compiled_graph):
                if item["type"] == "trace":
                    payload = dict(item["payload"])
                    payload["request_id"] = request_id
                    yield {
                        "event": "trace",
                        "data": json.dumps(payload, ensure_ascii=True),
                    }
                    continue

                answer = dict(item["payload"])
                answer["request_id"] = request_id
                answer["trace_count"] = len(answer.get("trace", []))
                yield {
                    "event": "answer",
                    "data": json.dumps(answer, ensure_ascii=True),
                }
        except Exception as exc:  # pragma: no cover
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "request_id": request_id,
                        "detail": f"Research workflow failed: {exc}",
                    },
                    ensure_ascii=True,
                ),
            }

    return EventSourceResponse(event_generator())


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for API and workflow monitoring."""

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
