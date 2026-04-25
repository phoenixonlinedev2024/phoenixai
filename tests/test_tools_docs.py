"""Tests for jarvis.tools.pdf_tools and spreadsheet_tools."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

from jarvis.tools.pdf_tools import _parse_pages  # noqa: E402
from jarvis.tools.spreadsheet_tools import (  # noqa: E402
    _read_csv,
    _read_spreadsheet,
    _write_spreadsheet,
)


# ── _parse_pages (pure logic) ─────────────────────────────────────────────────

def test_parse_pages_empty_defaults_to_first_10():
    assert _parse_pages("", 50) == list(range(10))


def test_parse_pages_empty_when_total_small():
    assert _parse_pages("", 3) == [0, 1, 2]


def test_parse_pages_single_page():
    # 1-indexed input, 0-indexed output
    assert _parse_pages("5", 20) == [4]


def test_parse_pages_range():
    assert _parse_pages("2-4", 10) == [1, 2, 3]


def test_parse_pages_multiple_specs():
    result = _parse_pages("1,3-5,8", 20)
    assert result == [0, 2, 3, 4, 7]


def test_parse_pages_clamps_to_total():
    result = _parse_pages("1-100", 5)
    assert result == [0, 1, 2, 3, 4]


def test_parse_pages_filters_out_of_range():
    # page 99 (index 98) should be dropped if total=5
    result = _parse_pages("99", 5)
    assert result == []


# ── _read_csv ─────────────────────────────────────────────────────────────────

def test_read_csv_basic(tmp_path):
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age"])
        writer.writerow(["Alice", "30"])
        writer.writerow(["Bob", "25"])

    result = _read_csv(str(csv_path))
    data = json.loads(result)
    assert data[0] == ["name", "age"]
    assert data[1] == ["Alice", "30"]
    assert len(data) == 3


def test_read_csv_respects_max_rows(tmp_path):
    csv_path = tmp_path / "big.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(500):
            writer.writerow([f"row{i}", i])

    result = _read_csv(str(csv_path), max_rows=10)
    data = json.loads(result)
    assert len(data) == 10


def test_read_csv_handles_missing_file():
    result = _read_csv("/nonexistent/path/file.csv")
    assert "error" in result.lower()


# ── _read_spreadsheet (delegates to CSV on .csv) ──────────────────────────────

def test_read_spreadsheet_csv_path(tmp_path):
    csv_path = tmp_path / "sheet.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["a", "b"])
        csv.writer(f).writerow(["1", "2"])

    result = _read_spreadsheet(str(csv_path))
    data = json.loads(result)
    assert data[0] == ["a", "b"]


def test_read_spreadsheet_xlsx_missing_openpyxl_gracefully(monkeypatch, tmp_path):
    """When openpyxl is unavailable for a real .xlsx, show helpful message."""
    fake_xlsx = tmp_path / "dummy.xlsx"
    fake_xlsx.write_bytes(b"not a real xlsx")

    # Mock openpyxl to raise ImportError when load_workbook is called
    fake_openpyxl = MagicMock()
    fake_openpyxl.load_workbook = MagicMock(side_effect=ImportError("no module"))
    monkeypatch.setitem(sys.modules, "openpyxl", fake_openpyxl)

    result = _read_spreadsheet(str(fake_xlsx))
    assert "not installed" in result.lower() or "error" in result.lower()


# ── _write_spreadsheet ────────────────────────────────────────────────────────

def test_write_spreadsheet_csv(tmp_path):
    path = tmp_path / "out.csv"
    data = [["name", "age"], ["Alice", 30], ["Bob", 25]]
    result = _write_spreadsheet(str(path), data)
    assert "3 rows" in result
    assert path.exists()

    # Verify contents
    with open(path) as f:
        lines = f.readlines()
    assert "Alice" in lines[1]


def test_write_spreadsheet_empty_data(tmp_path):
    path = tmp_path / "empty.csv"
    result = _write_spreadsheet(str(path), [])
    assert "0 rows" in result


def test_write_spreadsheet_unwritable_path():
    result = _write_spreadsheet("/nonexistent/deeply/nested/path.csv", [["a"]])
    assert "error" in result.lower()


# ── register_tools ────────────────────────────────────────────────────────────

def test_pdf_tools_register():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.pdf_tools import register_tools
    registry = build_registry()
    register_tools(registry)
    assert registry.get("read_pdf") is not None
    assert registry.get("extract_pdf_tables") is not None
    assert registry.get("pdf_metadata") is not None


def test_spreadsheet_tools_register():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.spreadsheet_tools import register_tools
    registry = build_registry()
    register_tools(registry)
    assert registry.get("read_spreadsheet") is not None
    assert registry.get("write_spreadsheet") is not None
    assert registry.get("list_sheets") is not None


def test_read_spreadsheet_xlsx_max_rows_break(monkeypatch):
    """Line 23: break fires when xlsx has more rows than max_rows."""
    import json
    fake_row1 = (1, 2, 3)
    fake_row2 = (4, 5, 6)
    fake_row3 = (7, 8, 9)

    fake_ws = MagicMock()
    fake_ws.iter_rows = MagicMock(return_value=iter([fake_row1, fake_row2, fake_row3]))

    fake_wb = MagicMock()
    fake_wb.active = fake_ws

    fake_openpyxl = MagicMock()
    fake_openpyxl.load_workbook = MagicMock(return_value=fake_wb)

    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _read_spreadsheet("dummy.xlsx", max_rows=2)

    rows = json.loads(result)
    assert len(rows) == 2
