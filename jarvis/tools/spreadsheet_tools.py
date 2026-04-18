"""Spreadsheet tools — read/write Excel, CSV, Google Sheets."""

from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _read_spreadsheet(path: str, sheet: str = "", max_rows: int = 100) -> str:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return _read_csv(path, max_rows)
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet] if sheet else wb.active
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= max_rows:
                break
            rows.append([str(c) if c is not None else "" for c in row])
        return json.dumps(rows, ensure_ascii=False)
    except ImportError:
        return "openpyxl not installed. Run: pip install openpyxl"
    except Exception as exc:
        return f"Spreadsheet read error: {exc}"


def _read_csv(path: str, max_rows: int = 200) -> str:
    try:
        import csv
        rows = []
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
        return json.dumps(rows, ensure_ascii=False)
    except Exception as exc:
        return f"CSV read error: {exc}"


def _write_spreadsheet(path: str, data: list, sheet: str = "Sheet1") -> str:
    p = Path(path)
    try:
        if p.suffix.lower() == ".csv":
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(data)
            return f"Written {len(data)} rows to {path}"
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet
        for row in data:
            ws.append(row)
        wb.save(path)
        return f"Written {len(data)} rows to {path}"
    except ImportError:
        return "openpyxl not installed. Run: pip install openpyxl"
    except Exception as exc:
        return f"Write error: {exc}"


def _list_sheets(path: str) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        return "Sheets: " + ", ".join(wb.sheetnames)
    except Exception as exc:
        return f"Error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="read_spreadsheet", description="Read Excel (.xlsx) or CSV file. Returns rows as JSON array.",
        input_schema={"type":"object","properties":{"path":{"type":"string"},"sheet":{"type":"string"},"max_rows":{"type":"integer","default":100}},"required":["path"]},
        fn=_read_spreadsheet, category="files"))
    registry.register(Tool(name="write_spreadsheet", description="Write a 2D array to an Excel or CSV file.",
        input_schema={"type":"object","properties":{"path":{"type":"string"},"data":{"type":"array","items":{"type":"array"}},"sheet":{"type":"string","default":"Sheet1"}},"required":["path","data"]},
        fn=_write_spreadsheet, category="files"))
    registry.register(Tool(name="list_sheets", description="List sheet names in an Excel workbook.",
        input_schema={"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
        fn=_list_sheets, category="files"))
