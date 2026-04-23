
"""PDF loading utilities for research assistant workflows."""

from __future__ import annotations

import importlib


def read_pdf_text(file_path: str, max_pages: int = 10) -> str:
    """Extract text from first pages of a PDF file for contextual analysis."""

    pdf_module = importlib.import_module("pypdf")
    PdfReader = pdf_module.PdfReader
    reader = PdfReader(file_path)
    pages = reader.pages[:max_pages]
    return "\n".join(page.extract_text() or "" for page in pages)
