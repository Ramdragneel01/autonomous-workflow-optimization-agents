
"""Searcher agent that retrieves grounded web evidence via Tavily."""

from __future__ import annotations

from typing import Any
import os
import re

import requests


DEFAULT_LIMIT = 5
MAX_LIMIT = 8
CONTENT_CHAR_LIMIT = 460

FALLBACK_SOURCES = [
    {
        "title": "NIST Digital Identity Guidelines Overview",
        "url": "https://www.nist.gov/itl/iam/digital-identity-guidelines",
        "content": "NIST digital identity guidance outlines assurance levels, risk management, and practical implementation trade-offs.",
    },
    {
        "title": "OWASP Cheat Sheet Series",
        "url": "https://cheatsheetseries.owasp.org/",
        "content": "OWASP cheat sheets provide secure implementation patterns for web security, APIs, validation, and threat mitigation.",
    },
    {
        "title": "CISA Cybersecurity Advisories",
        "url": "https://www.cisa.gov/news-events/cybersecurity-advisories",
        "content": "CISA advisories summarize emerging threats, indicators, and mitigation actions relevant for enterprise decision makers.",
    },
    {
        "title": "MITRE ATT&CK Knowledge Base",
        "url": "https://attack.mitre.org/",
        "content": "ATT&CK maps adversarial tactics and techniques, useful for structured reasoning about controls and residual risk.",
    },
    {
        "title": "OECD AI Principles",
        "url": "https://oecd.ai/en/ai-principles",
        "content": "OECD principles emphasize trustworthy AI, transparency, accountability, robustness, and governance.",
    },
]


def _clean_text(value: str) -> str:
    """Normalize whitespace and strip potentially noisy control characters."""

    compact = re.sub(r"\s+", " ", value).strip()
    return compact[:600]


def _normalize_item(item: dict[str, Any]) -> dict[str, str]:
    """Convert Tavily payload item into a stable source schema."""

    title = _clean_text(str(item.get("title", "Untitled")))
    url = _clean_text(str(item.get("url", "")))
    content = _clean_text(str(item.get("content", "")))
    return {
        "title": title or "Untitled",
        "url": url,
        "content": content[:CONTENT_CHAR_LIMIT],
    }


def _offline_sources(query: str, limit: int) -> list[dict[str, str]]:
    """Return deterministic fallback evidence when live search is unavailable."""

    contextual = []
    for item in FALLBACK_SOURCES:
        contextual.append(
            {
                "title": item["title"],
                "url": item["url"],
                "content": f"{item['content']} Query context: {query[:120]}",
            }
        )
    return contextual[:limit]


def search_sources(query: str, limit: int = DEFAULT_LIMIT) -> list[dict[str, str]]:
    """Return top web sources for a query using Tavily search API."""

    sanitized_query = _clean_text(query)
    if not sanitized_query:
        return []

    bounded_limit = max(1, min(limit, MAX_LIMIT))
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return _offline_sources(sanitized_query, bounded_limit)

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": sanitized_query, "max_results": bounded_limit},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return _offline_sources(sanitized_query, bounded_limit)

    results = payload.get("results", [])
    normalized = [_normalize_item(item) for item in results[:bounded_limit]]
    return normalized if normalized else _offline_sources(sanitized_query, bounded_limit)
