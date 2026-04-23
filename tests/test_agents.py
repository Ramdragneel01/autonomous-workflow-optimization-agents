
"""Unit tests for Searcher, Summarizer, and Critic agents."""

from __future__ import annotations

from agents.critic import critique
from agents.searcher import search_sources
from agents.summarizer import summarize


def test_search_sources_returns_fallback_when_no_key(monkeypatch):
    """Searcher should return deterministic fallback sources without Tavily key."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    sources = search_sources("enterprise secrets management", limit=3)

    assert len(sources) == 3
    assert all("title" in item for item in sources)
    assert all("content" in item for item in sources)


def test_summarizer_includes_findings_and_references():
    """Summarizer should include expected structural sections for grounding."""

    summary = summarize(
        "evaluate migration strategy",
        [
            {
                "title": "NIST reference",
                "url": "https://example.com/nist",
                "content": "Detailed guidance on identity and controls.",
            },
            {
                "title": "OWASP reference",
                "url": "https://example.com/owasp",
                "content": "API and validation recommendations.",
            },
        ],
    )

    assert "Key Findings:" in summary
    assert "References:" in summary


def test_critic_detects_low_evidence_summary():
    """Critic should return revision verdict for low-information summaries."""

    payload = critique("No sources were retrieved for this query.", source_count=0)

    assert payload["verdict"] == "needs_revision"
    assert payload["confidence"] < 0.88
    assert len(payload["gaps"]) > 0
