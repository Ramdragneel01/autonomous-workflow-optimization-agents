
"""Summarizer agent that builds structured brief from evidence."""

from __future__ import annotations


def _build_findings(sources: list[dict[str, str]], limit: int = 4) -> list[str]:
    """Construct concise grounded findings from retrieved sources."""

    findings: list[str] = []
    for item in sources[:limit]:
        title = item.get("title", "Untitled")
        snippet = item.get("content", "")
        snippet = snippet.replace("\n", " ").strip()
        findings.append(f"- {title}: {snippet[:220]}")
    return findings


def _build_source_refs(sources: list[dict[str, str]], limit: int = 4) -> list[str]:
    """Render source references in markdown-like citation style."""

    references: list[str] = []
    for idx, item in enumerate(sources[:limit], start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        references.append(f"[{idx}] {title} - {url}")
    return references


def summarize(query: str, sources: list[dict[str, str]]) -> str:
    """Create a concise research brief from retrieved sources."""

    if not sources:
        return (
            f"Research brief for '{query}':\n"
            "Key Findings:\n"
            "- No high-confidence sources were retrieved.\n"
            "Caveats:\n"
            "- Validate with trusted domain sources before making decisions.\n"
            "References:\n"
            "- None"
        )

    findings = _build_findings(sources)
    references = _build_source_refs(sources)

    caveats = [
        "- Source freshness may vary; verify publication dates for time-sensitive topics.",
        "- Cross-check claims that appear in only one source.",
    ]

    sections = [
        f"Research brief for '{query}':",
        "Key Findings:",
        *findings,
        "Caveats:",
        *caveats,
        "References:",
        *references,
    ]
    return "\n".join(sections)
