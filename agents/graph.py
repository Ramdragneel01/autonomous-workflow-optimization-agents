
"""LangGraph StateGraph orchestration for Searcher -> Summarizer -> Critic."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .critic import critique
from .searcher import search_sources
from .summarizer import summarize


class TraceEvent(TypedDict):
    """Represents one trace record emitted by an agent step."""

    agent: str
    stage: str
    message: str
    timestamp: str
    elapsed_ms: float
    payload: dict[str, object]


class AgentState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes."""

    query: str
    max_sources: int
    sources: list[dict[str, str]]
    summary: str
    critique: dict[str, object]
    traces: list[TraceEvent]


def _utc_now() -> str:
    """Return current timestamp in ISO-8601 UTC format."""

    return datetime.now(timezone.utc).isoformat()


def _trace(agent: str, stage: str, message: str, payload: dict[str, object], started_at: float) -> TraceEvent:
    """Build a normalized trace object for frontend and API consumers."""

    return {
        "agent": agent,
        "stage": stage,
        "message": message,
        "timestamp": _utc_now(),
        "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
        "payload": payload,
    }


def searcher_node(state: AgentState) -> AgentState:
    """Fetch candidate sources for the user query."""

    started_at = perf_counter()
    query = state.get("query", "")
    max_sources = int(state.get("max_sources", 5) or 5)
    sources = search_sources(query, limit=max_sources)

    event = _trace(
        agent="searcher",
        stage="retrieval",
        message="Retrieved candidate evidence sources",
        payload={
            "source_count": len(sources),
            "titles": [item.get("title", "Untitled") for item in sources[:4]],
        },
        started_at=started_at,
    )
    return {"sources": sources, "traces": [event]}


def summarizer_node(state: AgentState) -> AgentState:
    """Generate structured summary from search evidence."""

    started_at = perf_counter()
    summary = summarize(
        state.get("query", ""),
        state.get("sources", []),
    )
    event = _trace(
        agent="summarizer",
        stage="synthesis",
        message="Built structured research brief",
        payload={
            "summary_length": len(summary),
            "contains_references": "References:" in summary,
        },
        started_at=started_at,
    )
    return {"summary": summary, "traces": [event]}


def critic_node(state: AgentState) -> AgentState:
    """Evaluate the summary and assign confidence metadata."""

    started_at = perf_counter()
    critique_payload = critique(
        state.get("summary", ""),
        source_count=len(state.get("sources", [])),
    )
    event = _trace(
        agent="critic",
        stage="review",
        message="Evaluated evidence coverage and confidence",
        payload={
            "verdict": critique_payload.get("verdict", "unknown"),
            "confidence": critique_payload.get("confidence", 0.0),
            "gap_count": len(critique_payload.get("gaps", [])),
        },
        started_at=started_at,
    )
    return {"critique": critique_payload, "traces": [event]}


def build_graph():
    """Build and compile LangGraph workflow for research assistant."""

    graph = StateGraph(AgentState)
    graph.add_node("searcher", searcher_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("searcher")
    graph.add_edge("searcher", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


def run_research(query: str, max_sources: int, compiled_graph=None) -> dict[str, object]:
    """Execute LangGraph workflow once and return final answer plus trace events."""

    graph = compiled_graph or build_graph()
    bounded_sources = max(1, min(max_sources, 8))

    initial_state: AgentState = {
        "query": query,
        "max_sources": bounded_sources,
        "traces": [],
    }

    final_state: AgentState = {
        "query": query,
        "max_sources": bounded_sources,
        "sources": [],
        "summary": "",
        "critique": {},
        "traces": [],
    }
    all_traces: list[TraceEvent] = []
    started_at = perf_counter()

    for step in graph.stream(initial_state):
        for _, payload in step.items():
            if "sources" in payload:
                final_state["sources"] = payload["sources"]
            if "summary" in payload:
                final_state["summary"] = payload["summary"]
            if "critique" in payload:
                final_state["critique"] = payload["critique"]
            traces = payload.get("traces", [])
            all_traces.extend(traces)

    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    return {
        "query": query,
        "sources": final_state.get("sources", []),
        "summary": final_state.get("summary", ""),
        "critique": final_state.get("critique", {}),
        "trace": all_traces,
        "duration_ms": duration_ms,
    }


def stream_research(query: str, max_sources: int, compiled_graph=None):
    """Yield trace events and final answer from a single workflow execution."""

    graph = compiled_graph or build_graph()
    bounded_sources = max(1, min(max_sources, 8))

    initial_state: AgentState = {
        "query": query,
        "max_sources": bounded_sources,
        "traces": [],
    }

    final_state: AgentState = {
        "query": query,
        "max_sources": bounded_sources,
        "sources": [],
        "summary": "",
        "critique": {},
        "traces": [],
    }
    all_traces: list[TraceEvent] = []
    started_at = perf_counter()

    for step in graph.stream(initial_state):
        for _, payload in step.items():
            if "sources" in payload:
                final_state["sources"] = payload["sources"]
            if "summary" in payload:
                final_state["summary"] = payload["summary"]
            if "critique" in payload:
                final_state["critique"] = payload["critique"]

            for trace in payload.get("traces", []):
                all_traces.append(trace)
                yield {"type": "trace", "payload": trace}

    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    answer = {
        "query": query,
        "sources": final_state.get("sources", []),
        "summary": final_state.get("summary", ""),
        "critique": final_state.get("critique", {}),
        "trace": all_traces,
        "duration_ms": duration_ms,
    }
    yield {"type": "answer", "payload": answer}
