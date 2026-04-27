"""Tests for jarvis.tools.spreadsheet_tools — CSV, Excel (openpyxl mocked)."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from jarvis.tools.spreadsheet_tools import (
    _read_csv,
    _read_spreadsheet,
    _write_spreadsheet,
    _list_sheets,
)


# ── CSV (pure Python, no openpyxl) ───────────────────────────────────────────

def test_read_csv_basic(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,age\nAlice,30\nBob,25\n")
    result = _read_csv(str(p))
    rows = json.loads(result)
    assert rows[0] == ["name", "age"]
    assert rows[1] == ["Alice", "30"]


def test_read_csv_max_rows(tmp_path):
    p = tmp_path / "big.csv"
    p.write_text("\n".join(f"row{i}" for i in range(50)))
    result = _read_csv(str(p), max_rows=10)
    rows = json.loads(result)
    assert len(rows) == 10


def test_read_csv_missing_file():
    result = _read_csv("/nonexistent/path.csv")
    assert "CSV read error" in result


def test_read_spreadsheet_routes_csv(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n")
    result = _read_spreadsheet(str(p))
    rows = json.loads(result)
    assert rows[0] == ["a", "b"]


def test_write_spreadsheet_csv(tmp_path):
    p = tmp_path / "out.csv"
    result = _write_spreadsheet(str(p), [["h1", "h2"], ["v1", "v2"]])
    assert "Written 2 rows" in result
    content = p.read_text()
    assert "h1,h2" in content
    assert "v1,v2" in content


def test_write_spreadsheet_csv_empty(tmp_path):
    p = tmp_path / "empty.csv"
    result = _write_spreadsheet(str(p), [])
    assert "Written 0 rows" in result


# ── openpyxl paths (mocked) ──────────────────────────────────────────────────

def _make_fake_openpyxl(rows=None):
    """Return a fake openpyxl module with a workbook that yields given rows."""
    rows = rows or [["col1", "col2"], ["val1", "val2"]]
    fake_ws = MagicMock()
    fake_ws.iter_rows.return_value = [tuple(r) for r in rows]
    fake_wb = MagicMock()
    fake_wb.active = fake_ws
    fake_wb.__getitem__ = MagicMock(return_value=fake_ws)
    fake_wb.sheetnames = ["Sheet1", "Sheet2"]
    fake_openpyxl = MagicMock()
    fake_openpyxl.load_workbook.return_value = fake_wb
    fake_openpyxl.Workbook.return_value = MagicMock()
    return fake_openpyxl, fake_ws, fake_wb


def test_read_spreadsheet_xlsx_active_sheet(tmp_path):
    fake_openpyxl, fake_ws, fake_wb = _make_fake_openpyxl()
    p = tmp_path / "data.xlsx"
    p.touch()
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _read_spreadsheet(str(p))
    data = json.loads(result)
    assert data[0] == ["col1", "col2"]


def test_read_spreadsheet_xlsx_named_sheet(tmp_path):
    fake_openpyxl, fake_ws, fake_wb = _make_fake_openpyxl()
    p = tmp_path / "data.xlsx"
    p.touch()
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _read_spreadsheet(str(p), sheet="Sheet1")
    data = json.loads(result)
    assert isinstance(data, list)


def test_read_spreadsheet_no_openpyxl(tmp_path):
    p = tmp_path / "data.xlsx"
    p.touch()
    with patch.dict(sys.modules, {"openpyxl": None}):
        result = _read_spreadsheet(str(p))
    assert "openpyxl not installed" in result


def test_read_spreadsheet_exception(tmp_path):
    fake_openpyxl = MagicMock()
    fake_openpyxl.load_workbook.side_effect = RuntimeError("corrupt file")
    p = tmp_path / "bad.xlsx"
    p.touch()
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _read_spreadsheet(str(p))
    assert "Spreadsheet read error" in result


def test_write_spreadsheet_xlsx(tmp_path):
    fake_openpyxl = MagicMock()
    fake_wb = MagicMock()
    fake_ws = MagicMock()
    fake_wb.active = fake_ws
    fake_openpyxl.Workbook.return_value = fake_wb
    p = tmp_path / "out.xlsx"
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _write_spreadsheet(str(p), [["a", "b"], ["1", "2"]], sheet="MySheet")
    assert "Written 2 rows" in result
    fake_wb.save.assert_called_once_with(str(p))


def test_write_spreadsheet_no_openpyxl(tmp_path):
    p = tmp_path / "out.xlsx"
    with patch.dict(sys.modules, {"openpyxl": None}):
        result = _write_spreadsheet(str(p), [["a"]])
    assert "openpyxl not installed" in result


def test_write_spreadsheet_exception(tmp_path):
    fake_openpyxl = MagicMock()
    fake_wb = MagicMock()
    fake_wb.active = MagicMock()
    fake_wb.save.side_effect = RuntimeError("disk full")
    fake_openpyxl.Workbook.return_value = fake_wb
    p = tmp_path / "err.xlsx"
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _write_spreadsheet(str(p), [["x"]])
    assert "Write error" in result


def test_list_sheets_success(tmp_path):
    fake_openpyxl, _, fake_wb = _make_fake_openpyxl()
    p = tmp_path / "wb.xlsx"
    p.touch()
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _list_sheets(str(p))
    assert "Sheet1" in result
    assert "Sheet2" in result


def test_list_sheets_exception():
    fake_openpyxl = MagicMock()
    fake_openpyxl.load_workbook.side_effect = RuntimeError("bad")
    with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
        result = _list_sheets("/tmp/x.xlsx")
    assert "Error" in result


def test_spreadsheet_tools_register():
    from jarvis.tools.spreadsheet_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert {"read_spreadsheet", "write_spreadsheet", "list_sheets"} <= names


# ── CSV edge cases ────────────────────────────────────────────────────────────

def test_read_csv_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    result = _read_csv(str(p))
    rows = json.loads(result)
    assert rows == []


def test_read_csv_single_column(tmp_path):
    p = tmp_path / "single.csv"
    p.write_text("header\nval1\nval2\n")
    result = _read_csv(str(p))
    rows = json.loads(result)
    assert rows[0] == ["header"]
    assert rows[1] == ["val1"]


def test_read_csv_unicode_content(tmp_path):
    p = tmp_path / "unicode.csv"
    p.write_text("name,city\nJérôme,Montréal\n", encoding="utf-8")
    result = _read_csv(str(p))
    rows = json.loads(result)
    assert "Jérôme" in rows[1][0]


def test_write_spreadsheet_csv_roundtrip(tmp_path):
    p = tmp_path / "roundtrip.csv"
    data = [["a", "b", "c"], [1, 2, 3], [4, 5, 6]]
    _write_spreadsheet(str(p), data)
    result = _read_csv(str(p))
    rows = json.loads(result)
    assert rows[0] == ["a", "b", "c"]
    assert len(rows) == 3


def test_write_spreadsheet_zero_rows_csv(tmp_path):
    p = tmp_path / "zero.csv"
    result = _write_spreadsheet(str(p), [])
    assert "0 rows" in result
    # File should exist but be empty
    assert p.exists()


# ── _read_spreadsheet CSV routing ─────────────────────────────────────────────

def test_read_spreadsheet_routes_csv_extension(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("x,y\n10,20\n")
    result = _read_spreadsheet(str(p))
    rows = json.loads(result)
    assert rows[0] == ["x", "y"]


def test_read_spreadsheet_non_existent_file_returns_error(tmp_path):
    result = _read_spreadsheet(str(tmp_path / "missing.xlsx"))
    # Either an error message or "openpyxl not installed" — both indicate failure
    assert "error" in result.lower() or "not installed" in result.lower()
