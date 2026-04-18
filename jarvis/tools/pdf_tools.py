"""PDF tools — extract text, tables, metadata from PDF files."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _read_pdf(path: str, pages: str = "") -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            page_nums = _parse_pages(pages, len(pdf.pages))
            parts = []
            for i in page_nums:
                page = pdf.pages[i]
                text = page.extract_text() or ""
                parts.append(f"--- Page {i+1} ---\n{text}")
        return "\n".join(parts)[:10000]
    except ImportError:
        return "pdfplumber not installed. Run: pip install pdfplumber"
    except Exception as exc:
        return f"PDF read error: {exc}"


def _extract_pdf_tables(path: str, page: int = 0) -> str:
    try:
        import pdfplumber, json
        with pdfplumber.open(path) as pdf:
            p = pdf.pages[page]
            tables = p.extract_tables()
            if not tables:
                return "No tables found on this page."
            return json.dumps(tables, indent=2)[:5000]
    except ImportError:
        return "pdfplumber not installed. Run: pip install pdfplumber"
    except Exception as exc:
        return f"Table extraction error: {exc}"


def _pdf_metadata(path: str) -> str:
    try:
        import pdfplumber, json
        with pdfplumber.open(path) as pdf:
            meta = pdf.metadata or {}
            return json.dumps({"pages": len(pdf.pages), **meta}, indent=2)
    except Exception as exc:
        return f"PDF metadata error: {exc}"


def _parse_pages(spec: str, total: int) -> list[int]:
    if not spec:
        return list(range(min(total, 10)))
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a)-1, min(int(b), total)))
        else:
            pages.append(int(part)-1)
    return [p for p in pages if 0 <= p < total]


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="read_pdf", description="Extract text from a PDF file (all or specific pages).",
        input_schema={"type":"object","properties":{"path":{"type":"string"},"pages":{"type":"string","description":"e.g. '1-5,8' (optional)"}},"required":["path"]},
        fn=_read_pdf, category="files"))
    registry.register(Tool(name="extract_pdf_tables", description="Extract tables from a PDF page as JSON.",
        input_schema={"type":"object","properties":{"path":{"type":"string"},"page":{"type":"integer","default":0}},"required":["path"]},
        fn=_extract_pdf_tables, category="files"))
    registry.register(Tool(name="pdf_metadata", description="Get PDF metadata: page count, author, title, etc.",
        input_schema={"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
        fn=_pdf_metadata, category="files"))
