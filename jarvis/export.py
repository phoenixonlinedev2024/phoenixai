"""Conversation export — markdown and PDF from session history."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.memory.store import MemoryStore


def export_markdown(memory: "MemoryStore", session_id: str, output_path: str | None = None) -> str:
    """Export a conversation session as a Markdown file."""
    history = memory.get_history(session_id, limit=500)
    if not history:
        return "No conversation history found for this session."

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    filename = output_path or f"jarvis_session_{ts}.md"

    lines = [
        f"# JARVIS Session — {ts}\n",
        f"Session ID: `{session_id}`\n",
        "---\n",
    ]
    for msg in history:
        role = "**You**" if msg["role"] == "user" else "**JARVIS**"
        lines.append(f"{role}:\n{msg['content']}\n\n---\n")

    content = "\n".join(lines)
    Path(filename).write_text(content, encoding="utf-8")
    return f"Exported {len(history)} messages to {filename}"


def export_pdf(memory: "MemoryStore", session_id: str, output_path: str | None = None) -> str:
    """Export a conversation session as a PDF file."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib import colors
    except ImportError:
        return "reportlab not installed. Run: pip install reportlab"

    history = memory.get_history(session_id, limit=500)
    if not history:
        return "No conversation history found for this session."

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    filename = output_path or f"jarvis_session_{ts}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    user_style = ParagraphStyle("User", parent=styles["Normal"], textColor=colors.darkblue, fontName="Helvetica-Bold")
    jarvis_style = ParagraphStyle("Jarvis", parent=styles["Normal"], textColor=colors.darkgreen, fontName="Helvetica")
    story = [Paragraph(f"JARVIS Session — {ts}", styles["Title"]), Spacer(1, 0.5*cm)]

    for msg in history:
        if msg["role"] == "user":
            story.append(Paragraph("You:", user_style))
            style = styles["Normal"]
        else:
            story.append(Paragraph("JARVIS:", jarvis_style))
            style = styles["Normal"]
        # Escape HTML special chars
        text = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(text, style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return f"Exported {len(history)} messages to {filename}"
