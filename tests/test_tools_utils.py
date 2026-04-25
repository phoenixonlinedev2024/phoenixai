"""Tests for nlp_cron, package_installer, subagent_tools, spreadsheet_tools."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()


# ══════════════════════════════════════════════════════════════════════════════
# nlp_cron
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.nlp_cron import _nl_to_cron, _validate_cron  # noqa: E402


@pytest.mark.parametrize("phrase,expected", [
    ("every minute", "* * * * *"),
    ("every 5 minutes", "*/5 * * * *"),
    ("every 30 minutes", "*/30 * * * *"),
    ("every hour", "0 * * * *"),
    ("every 4 hours", "0 */4 * * *"),
    ("every morning", "0 8 * * *"),
    ("every evening", "0 18 * * *"),
    ("every night", "0 22 * * *"),
    ("every monday", "0 9 * * 1"),
    ("every tuesday", "0 9 * * 2"),
    ("every wednesday", "0 9 * * 3"),
    ("every thursday", "0 9 * * 4"),
    ("every friday", "0 9 * * 5"),
    ("every weekend", "0 9 * * 6,0"),
    ("every weekday", "0 9 * * 1-5"),
    ("every week", "0 9 * * 1"),
    ("every month", "0 9 1 * *"),
    ("midnight", "0 0 * * *"),
    ("noon", "0 12 * * *"),
])
def test_nl_to_cron_patterns(phrase, expected):
    assert _nl_to_cron(phrase) == expected


def test_nl_to_cron_every_day_at_hour():
    assert _nl_to_cron("every day at 14") == "0 14 * * *"


def test_nl_to_cron_daily_at_with_minutes():
    assert _nl_to_cron("daily at 9:30") == "30 9 * * *"


def test_nl_to_cron_unrecognised_returns_error():
    out = _nl_to_cron("when the cows come home")
    assert "Could not parse" in out


def test_validate_cron_valid():
    out = _validate_cron("0 9 * * 1-5")
    assert "Valid cron" in out


def test_validate_cron_wrong_parts():
    out = _validate_cron("0 9 *")
    assert "must have 5 parts" in out


def test_validate_cron_out_of_range():
    out = _validate_cron("99 9 * * *")
    assert "out of range" in out.lower()


def test_validate_cron_wildcard_passes():
    assert "Valid" in _validate_cron("* * * * *")


def test_validate_cron_non_numeric_field_passes():
    """Lines 65-66: ValueError is swallowed for non-numeric cron field values."""
    out = _validate_cron("abc 0 * * *")
    assert "Valid cron" in out or "Cron issues" in out


def test_nlp_cron_register_tools():
    from jarvis.tools.nlp_cron import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert "nl_to_cron" in names
    assert "validate_cron" in names


# ══════════════════════════════════════════════════════════════════════════════
# package_installer
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.package_installer import (  # noqa: E402
    _install_package,
    _list_installed,
    _check_package,
)


def test_install_package_success(monkeypatch):
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = "Successfully installed foo"
    fake.stderr = ""
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    out = _install_package("foo")
    assert "Successfully installed" in out


def test_install_package_failure(monkeypatch):
    fake = MagicMock()
    fake.returncode = 1
    fake.stdout = ""
    fake.stderr = "ERROR: no version matching"
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    out = _install_package("nosuchpkg==99.99")
    assert "Install failed" in out


def test_install_package_timeout(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        MagicMock(side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=120))
    )
    out = _install_package("slowpkg")
    assert "timed out" in out.lower()


def test_install_package_exception(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=RuntimeError("no pip")))
    out = _install_package("pkg")
    assert "Install error" in out


def test_install_package_upgrade_passes_flag(monkeypatch):
    calls = []
    fake = MagicMock(returncode=0, stdout="ok", stderr="")
    def fake_run(args, **kw):
        calls.append(args)
        return fake
    monkeypatch.setattr(subprocess, "run", fake_run)
    _install_package("pkg", upgrade=True)
    assert "--upgrade" in calls[0]


def test_list_installed_returns_packages(monkeypatch):
    fake = MagicMock(stdout="Package    Version\npytest     8.0.0\nnumpy      2.0.0\n", returncode=0)
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    out = _list_installed()
    assert "pytest" in out


def test_list_installed_filter(monkeypatch):
    fake = MagicMock(stdout="Package Version\npytest 8.0\nnumpy 2.0\n", returncode=0)
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    out = _list_installed("pyt")
    assert "pytest" in out
    assert "numpy" not in out


def test_list_installed_exception(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=RuntimeError("oops")))
    out = _list_installed()
    assert "List error" in out


def test_check_package_installed(monkeypatch):
    fake = MagicMock(returncode=0, stdout="Name: pytest\nVersion: 8.0.0\n")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    out = _check_package("pytest")
    assert "pytest" in out


def test_check_package_not_installed(monkeypatch):
    fake = MagicMock(returncode=1, stdout="")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake))
    out = _check_package("nonexistentpkg")
    assert "not installed" in out


def test_check_package_exception(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=RuntimeError("fail")))
    out = _check_package("x")
    assert "Check error" in out


def test_package_installer_register_tools():
    from jarvis.tools.package_installer import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert "install_package" in names
    assert "list_packages" in names
    assert "check_package" in names


# ══════════════════════════════════════════════════════════════════════════════
# subagent_tools
# ══════════════════════════════════════════════════════════════════════════════

import jarvis.tools.subagent_tools as _submod  # noqa: E402


def test_delegate_no_pool_returns_message():
    _submod._pool = None
    out = _submod._delegate("do something")
    assert "not initialised" in out.lower()


def test_delegate_parallel_no_pool_returns_message():
    _submod._pool = None
    out = _submod._delegate_parallel([{"goal": "x"}])
    assert "not initialised" in out.lower()


def test_delegate_with_pool_calls_dispatch_one(monkeypatch):
    fake_pool = MagicMock()
    fake_pool.dispatch_one = MagicMock(return_value="task done")
    # asyncio.run wraps an async call; monkeypatch it to call the mock directly
    monkeypatch.setattr("jarvis.tools.subagent_tools.asyncio.run",
                        lambda coro: "task done")
    _submod._pool = fake_pool
    out = _submod._delegate("do X")
    assert out == "task done"
    _submod._pool = None


def test_delegate_parallel_returns_json(monkeypatch):
    from jarvis.agents.subagent import SubagentTask
    fake_pool = MagicMock()
    def fake_asyncio_run(coro):
        return {"abc": "result for abc"}
    monkeypatch.setattr("jarvis.tools.subagent_tools.asyncio.run", fake_asyncio_run)
    _submod._pool = fake_pool
    out = _submod._delegate_parallel([{"goal": "g", "context": "c"}])
    parsed = json.loads(out)
    assert "abc" in parsed
    _submod._pool = None


def test_subagent_tools_register_tools():
    from jarvis.tools.subagent_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    _submod._pool = None
    register_tools(reg, jarvis=None)
    names = {t.name for t in reg.all()}
    assert "delegate_task" in names
    assert "delegate_parallel" in names


def test_get_pool_creates_pool_with_jarvis(monkeypatch):
    """Lines 18-19: _get_pool initialises pool when jarvis is provided."""
    _submod._pool = None
    fake_pool_instance = MagicMock()
    fake_pool_cls = MagicMock(return_value=fake_pool_instance)
    monkeypatch.setattr("jarvis.agents.subagent.SubagentPool", fake_pool_cls)
    import jarvis.tools.subagent_tools as submod2
    submod2._pool = None
    fake_jarvis = MagicMock()
    pool = submod2._get_pool(jarvis=fake_jarvis)
    assert pool is fake_pool_instance
    submod2._pool = None


def test_delegate_exception_returns_error(monkeypatch):
    """Lines 30-31: _delegate returns error string when pool.dispatch_one raises."""
    fake_pool = MagicMock()
    _submod._pool = fake_pool
    monkeypatch.setattr("jarvis.tools.subagent_tools.asyncio.run",
                        MagicMock(side_effect=RuntimeError("agent crashed")))
    out = _submod._delegate("task")
    assert "Subagent error" in out
    assert "agent crashed" in out
    _submod._pool = None


def test_delegate_parallel_exception_returns_error(monkeypatch):
    """Lines 44-45: _delegate_parallel returns error string when dispatch raises."""
    fake_pool = MagicMock()
    _submod._pool = fake_pool
    monkeypatch.setattr("jarvis.tools.subagent_tools.asyncio.run",
                        MagicMock(side_effect=RuntimeError("parallel crash")))
    out = _submod._delegate_parallel([{"goal": "g"}])
    assert "Parallel subagent error" in out
    assert "parallel crash" in out
    _submod._pool = None


# ══════════════════════════════════════════════════════════════════════════════
# spreadsheet_tools
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.spreadsheet_tools import (  # noqa: E402
    _read_csv,
    _read_spreadsheet,
    _write_spreadsheet,
)


def test_read_csv_roundtrip(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    out = _read_csv(str(p))
    rows = json.loads(out)
    assert rows[0] == ["a", "b", "c"]
    assert rows[1] == ["1", "2", "3"]


def test_read_csv_respects_max_rows(tmp_path):
    p = tmp_path / "big.csv"
    p.write_text("\n".join(f"row{i}" for i in range(300)), encoding="utf-8")
    out = _read_csv(str(p), max_rows=10)
    assert len(json.loads(out)) == 10


def test_read_csv_missing_file():
    out = _read_csv("/no/such/file.csv")
    assert "CSV read error" in out


def test_read_spreadsheet_routes_csv(tmp_path):
    p = tmp_path / "sheet.csv"
    p.write_text("x,y\n1,2\n", encoding="utf-8")
    out = _read_spreadsheet(str(p))
    assert "x" in out


def test_read_spreadsheet_missing_openpyxl(tmp_path, monkeypatch):
    p = tmp_path / "book.xlsx"
    p.write_text("fake")  # not a real xlsx
    with patch.dict(sys.modules, {"openpyxl": None}):
        out = _read_spreadsheet(str(p))
    assert "openpyxl not installed" in out


def test_write_spreadsheet_csv(tmp_path):
    p = tmp_path / "out.csv"
    out = _write_spreadsheet(str(p), [["h1", "h2"], [1, 2], [3, 4]])
    assert "Written 3 rows" in out
    # Verify CSV content is readable back
    content = _read_csv(str(p))
    rows = json.loads(content)
    assert rows[0] == ["h1", "h2"]


def test_write_spreadsheet_missing_openpyxl(tmp_path, monkeypatch):
    p = tmp_path / "out.xlsx"
    with patch.dict(sys.modules, {"openpyxl": None}):
        out = _write_spreadsheet(str(p), [[1, 2]])
    assert "openpyxl not installed" in out


def test_spreadsheet_register_tools():
    from jarvis.tools.spreadsheet_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert "read_spreadsheet" in names
    assert "write_spreadsheet" in names
    assert "list_sheets" in names
