
"""Critic agent that checks summary quality and identifies missing evidence."""

from __future__ import annotations


def critique(summary: str, source_count: int) -> dict[str, object]:
    """Score summary confidence and list obvious gaps."""

    gaps: list[str] = []
    recommendations: list[str] = []

    if "No sources" in summary:
        gaps.append("No supporting sources were available.")
        recommendations.append("Run broader retrieval query or enable external search provider.")
    if len(summary) < 120:
        gaps.append("Summary is too short to be decision-grade.")
        recommendations.append("Add more findings and cite evidence per claim.")
    if source_count < 2:
        gaps.append("Evidence diversity is low (fewer than 2 sources).")
        recommendations.append("Seek at least two independent corroborating references.")
    if "References:" not in summary:
        gaps.append("Summary does not include explicit references section.")
        recommendations.append("Include source references in standardized format.")

    confidence = 0.88
    if gaps:
        confidence = max(0.25, 0.88 - 0.16 * len(gaps))

    return {
        "confidence": round(confidence, 2),
        "gaps": gaps,
        "recommendations": recommendations,
        "evidence_sources": source_count,
        "verdict": "needs_revision" if gaps else "sufficient",
    }
