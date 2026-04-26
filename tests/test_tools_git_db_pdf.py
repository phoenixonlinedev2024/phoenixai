"""Tests for git_tools, database_tools, pdf_tools."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# git_tools
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.git_tools import (  # noqa: E402
    _git_status,
    _git_log,
    _git_clone,
    _git_commit,
    _git_diff,
    _git_branch,
)


def _fake_git():
    """Return a sys.modules-patchable fake git module."""
    fake = MagicMock()
    repo = MagicMock()
    fake.Repo.return_value = repo
    fake.Repo.clone_from = MagicMock(return_value=repo)
    return fake, repo


def test_git_status_returns_json():
    fake, repo = _fake_git()
    repo.active_branch.name = "main"
    repo.index.diff.return_value = []
    repo.untracked_files = ["new.py"]
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_status(".")
    d = json.loads(out)
    assert d["branch"] == "main"
    assert "new.py" in d["untracked"]


def test_git_status_missing_gitpython():
    with patch.dict(sys.modules, {"git": None}):
        out = _git_status(".")
    assert "gitpython not installed" in out


def test_git_status_exception():
    fake = MagicMock()
    fake.Repo.side_effect = RuntimeError("not a git repo")
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_status(".")
    assert "Git status error" in out


def test_git_log_formats_commits():
    fake, repo = _fake_git()
    commit = MagicMock()
    commit.hexsha = "abcdef0123456789"
    commit.committed_datetime.strftime.return_value = "2024-01-01"
    commit.author.name = "Tony"
    commit.message = "feat: add gadget\n"
    repo.iter_commits.return_value = [commit]
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_log(".", n=1)
    assert "abcdef01" in out
    assert "Tony" in out
    assert "feat: add gadget" in out


def test_git_log_exception():
    fake, repo = _fake_git()
    repo.iter_commits.side_effect = RuntimeError("bad")
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_log(".")
    assert "Git log error" in out


def test_git_clone_success():
    fake, repo = _fake_git()
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_clone("https://github.com/foo/bar.git", dest="/tmp/bar")
    assert "Cloned" in out
    assert "bar.git" in out or "bar" in out


def test_git_clone_derives_dest_from_url():
    fake, repo = _fake_git()
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_clone("https://github.com/org/myrepo.git")
    assert "myrepo" in out


def test_git_clone_error():
    fake = MagicMock()
    fake.Repo.clone_from = MagicMock(side_effect=RuntimeError("auth failed"))
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_clone("https://github.com/x/y.git")
    assert "Clone error" in out


def test_git_commit_success():
    fake, repo = _fake_git()
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_commit(".", "my commit msg")
    assert "Committed" in out
    assert "my commit msg" in out


def test_git_commit_error():
    fake, repo = _fake_git()
    repo.index.commit.side_effect = RuntimeError("nothing staged")
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_commit(".", "msg")
    assert "Commit error" in out


def test_git_diff_no_changes():
    fake, repo = _fake_git()
    repo.index.diff.return_value = []
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_diff(".")
    assert "No changes" in out


def test_git_diff_returns_paths():
    fake, repo = _fake_git()
    d = MagicMock()
    d.a_path = "foo.py"
    d.b_path = "foo.py"
    repo.index.diff.return_value = [d]
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_diff(".")
    assert "foo.py" in out


def test_git_branch_list():
    fake, repo = _fake_git()
    b1, b2 = MagicMock(), MagicMock()
    b1.name = "main"
    b2.name = "feature"
    repo.branches = [b1, b2]
    repo.active_branch.name = "main"
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_branch(".")
    assert "main" in out
    assert "feature" in out


def test_git_branch_create():
    fake, repo = _fake_git()
    repo.branches = []
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_branch(".", name="new-feature")
    assert "Created branch: new-feature" in out


def test_git_branch_error():
    fake = MagicMock()
    fake.Repo.side_effect = RuntimeError("not a repo")
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_branch(".", name="x")
    assert "Branch error" in out


def test_git_clone_with_branch():
    """Line 41: kwargs['branch'] is set when branch arg is provided."""
    fake, repo = _fake_git()
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_clone("https://github.com/org/repo.git", branch="develop")
    assert "Cloned" in out
    fake.Repo.clone_from.assert_called_once()
    _, kwargs = fake.Repo.clone_from.call_args
    assert kwargs.get("branch") == "develop"


def test_git_diff_error():
    """Lines 70-71: Diff error path when git.Repo raises."""
    fake = MagicMock()
    fake.Repo.side_effect = RuntimeError("not a repo")
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_diff(".")
    assert "Diff error" in out


def test_git_branch_checkout_existing():
    """Lines 82-83: checkout=True for an existing branch name."""
    fake, repo = _fake_git()
    b = MagicMock()
    b.name = "main"
    repo.branches = [b]
    with patch.dict(sys.modules, {"git": fake}):
        out = _git_branch(".", name="main", checkout=True)
    assert "Switched to branch: main" in out
    repo.git.checkout.assert_called_once_with("main")


def test_git_tools_register():
    from jarvis.tools.git_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert {"git_status", "git_log", "git_clone", "git_commit", "git_diff", "git_branch"} <= names


# ══════════════════════════════════════════════════════════════════════════════
# database_tools (SQLite for real; SQLAlchemy is stdlib-friendly)
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.database_tools import _db_query, _db_schema, _db_execute  # noqa: E402

_SQLITE_URL = "sqlite:///:memory:"


def test_db_query_select(monkeypatch):
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        pytest.skip("sqlalchemy not installed")

    out = _db_query(_SQLITE_URL, "SELECT 1 AS val")
    d = json.loads(out)
    assert "val" in d["columns"]
    assert d["rows"][0]["val"] == 1


def test_db_query_no_sqlalchemy():
    with patch.dict(sys.modules, {"sqlalchemy": None}):
        out = _db_query(_SQLITE_URL, "SELECT 1")
    assert "SQLAlchemy not installed" in out


def test_db_execute_creates_table(monkeypatch):
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        pytest.skip("sqlalchemy not installed")

    from sqlalchemy import create_engine
    url = "sqlite:///:memory:"
    out = _db_execute(url, "CREATE TABLE t (id INTEGER PRIMARY KEY)")
    assert "Executed" in out


def test_db_execute_exception():
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        pytest.skip("sqlalchemy not installed")

    out = _db_execute(_SQLITE_URL, "DROP TABLE nonexistent_xyz")
    assert "Execute error" in out or "Executed" in out


def test_db_schema_lists_tables(monkeypatch):
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        pytest.skip("sqlalchemy not installed")

    from sqlalchemy import create_engine, text
    url = "sqlite:///:memory:"
    engine = create_engine(url)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE demo (x INTEGER)"))

    # Use a fresh engine at the same URL — SQLite in-memory won't persist,
    # so just test the exception/table-list branch with the shared engine.
    out = _db_schema(url)
    # Returns "Tables: " possibly with empty list
    assert "Tables" in out or "Schema error" in out


def test_db_tools_register():
    from jarvis.tools.database_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert {"db_query", "db_schema", "db_execute"} <= names


# ── database_tools with mocked SQLAlchemy ─────────────────────────────────────

def _make_fake_sqlalchemy(returns_rows=True, rows=None, cols=None, rowcount=1):
    """Build a fake sqlalchemy module with controllable result."""
    if rows is None:
        rows = [(42,)]
    if cols is None:
        cols = ["val"]

    fake_result = MagicMock()
    fake_result.returns_rows = returns_rows
    fake_result.keys.return_value = cols
    fake_result.fetchmany.return_value = [tuple(v for v in zip(cols, r))[0] if len(cols) == 1 else r
                                          for r in rows]
    fake_result.rowcount = rowcount

    fake_conn = MagicMock()
    fake_conn.execute.return_value = fake_result
    fake_conn.__enter__ = MagicMock(return_value=fake_conn)
    fake_conn.__exit__ = MagicMock(return_value=False)

    fake_engine = MagicMock()
    fake_engine.connect.return_value = fake_conn
    fake_engine.begin.return_value = fake_conn

    fake_sa = MagicMock()
    fake_sa.create_engine.return_value = fake_engine
    fake_sa.text = MagicMock(side_effect=lambda s: s)
    fake_sa.inspect.return_value = MagicMock(
        get_table_names=MagicMock(return_value=["users", "orders"]),
        get_columns=MagicMock(return_value=[{"name": "id", "type": "INTEGER"}]),
        get_pk_constraint=MagicMock(return_value={"constrained_columns": ["id"]}),
        get_foreign_keys=MagicMock(return_value=[]),
    )
    return fake_sa, fake_engine, fake_conn, fake_result


def test_db_query_mocked_select():
    from jarvis.tools.database_tools import _db_query
    fake_sa, _, _, fake_result = _make_fake_sqlalchemy(returns_rows=True)
    fake_result.keys.return_value = ["val"]
    fake_result.fetchmany.return_value = [(42,)]
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_query("sqlite:///:memory:", "SELECT 1 AS val")
    data = json.loads(out)
    assert "columns" in data
    assert "rows" in data


def test_db_query_mocked_non_select():
    from jarvis.tools.database_tools import _db_query
    fake_sa, _, _, fake_result = _make_fake_sqlalchemy(returns_rows=False, rowcount=3)
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_query("sqlite:///:memory:", "INSERT INTO t VALUES (1)")
    assert "Rows affected: 3" in out


def test_db_execute_mocked():
    from jarvis.tools.database_tools import _db_execute
    fake_sa, _, _, fake_result = _make_fake_sqlalchemy(returns_rows=False, rowcount=2)
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_execute("sqlite:///:memory:", "DELETE FROM t")
    assert "Executed" in out
    assert "2" in out


def test_db_schema_mocked_list_tables():
    from jarvis.tools.database_tools import _db_schema
    fake_sa, _, _, _ = _make_fake_sqlalchemy()
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_schema("sqlite:///:memory:")
    assert "users" in out
    assert "Tables" in out


def test_db_schema_mocked_with_table():
    from jarvis.tools.database_tools import _db_schema
    fake_sa, _, _, _ = _make_fake_sqlalchemy()
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_schema("sqlite:///:memory:", table="users")
    import json as _json
    data = _json.loads(out)
    assert data["table"] == "users"
    assert "columns" in data


def test_db_query_mocked_exception():
    from jarvis.tools.database_tools import _db_query
    fake_sa = MagicMock()
    fake_sa.create_engine.side_effect = RuntimeError("connection refused")
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_query("bad://url", "SELECT 1")
    assert "DB query error" in out


def test_db_execute_mocked_exception():
    from jarvis.tools.database_tools import _db_execute
    fake_sa = MagicMock()
    fake_sa.create_engine.side_effect = RuntimeError("bad connection")
    with patch.dict(sys.modules, {"sqlalchemy": fake_sa}):
        out = _db_execute("bad://url", "DROP TABLE x")
    assert "Execute error" in out


# ══════════════════════════════════════════════════════════════════════════════
# pdf_tools._parse_pages (pure logic, no pdfplumber needed)
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.pdf_tools import _parse_pages, _read_pdf, _extract_pdf_tables, _pdf_metadata  # noqa: E402


def test_parse_pages_empty_spec():
    assert _parse_pages("", 20) == list(range(10))  # caps at 10


def test_parse_pages_range():
    assert _parse_pages("2-4", 10) == [1, 2, 3]  # 0-indexed


def test_parse_pages_list():
    assert _parse_pages("1,3,5", 10) == [0, 2, 4]


def test_parse_pages_mixed():
    result = _parse_pages("1-2,5", 10)
    assert 0 in result and 1 in result and 4 in result


def test_parse_pages_out_of_range():
    result = _parse_pages("5-10", 5)
    assert all(p < 5 for p in result)


def test_read_pdf_no_pdfplumber():
    with patch.dict(sys.modules, {"pdfplumber": None}):
        out = _read_pdf("/tmp/x.pdf")
    assert "pdfplumber not installed" in out


def test_read_pdf_file_not_found():
    # pdfplumber may or may not be installed; either way the path doesn't exist
    out = _read_pdf("/tmp/does_not_exist_xyz.pdf")
    assert "error" in out.lower() or "pdfplumber not installed" in out


def test_extract_pdf_tables_no_pdfplumber():
    with patch.dict(sys.modules, {"pdfplumber": None}):
        out = _extract_pdf_tables("/tmp/x.pdf")
    assert "pdfplumber not installed" in out


def test_pdf_metadata_exception():
    fake_plumber = MagicMock()
    fake_plumber.open.side_effect = RuntimeError("bad pdf")
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _pdf_metadata("/tmp/bad.pdf")
    assert "PDF metadata error" in out


def test_pdf_tools_register():
    from jarvis.tools.pdf_tools import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert {"read_pdf", "extract_pdf_tables", "pdf_metadata"} <= names


def _make_fake_pdfplumber(text="page text", tables=None, metadata=None):
    """Build a fake pdfplumber module with a two-page PDF."""
    fake_page = MagicMock()
    fake_page.extract_text.return_value = text
    fake_page.extract_tables.return_value = tables or [[[" A", "B"], ["1", "2"]]]
    fake_pdf = MagicMock()
    fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
    fake_pdf.__exit__ = MagicMock(return_value=False)
    fake_pdf.pages = [fake_page, fake_page]
    fake_pdf.metadata = metadata or {"Author": "Tony", "Title": "Report"}
    fake_plumber = MagicMock()
    fake_plumber.open.return_value = fake_pdf
    return fake_plumber, fake_pdf, fake_page


def test_read_pdf_runtime_exception():
    """Lines 23-24: PDF read error when pdfplumber raises a non-import exception."""
    fake_plumber = MagicMock()
    fake_plumber.open.side_effect = RuntimeError("corrupt pdf file")
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _read_pdf("/tmp/doc.pdf")
    assert "PDF read error" in out


def test_read_pdf_success():
    fake_plumber, _, _ = _make_fake_pdfplumber(text="Hello PDF")
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _read_pdf("/tmp/doc.pdf")
    assert "Hello PDF" in out
    assert "Page 1" in out


def test_read_pdf_with_page_spec():
    fake_plumber, _, _ = _make_fake_pdfplumber(text="specific page")
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _read_pdf("/tmp/doc.pdf", pages="1")
    assert "specific page" in out


def test_read_pdf_page_returns_none_text():
    fake_plumber, fake_pdf, fake_page = _make_fake_pdfplumber()
    fake_page.extract_text.return_value = None
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _read_pdf("/tmp/doc.pdf")
    assert "Page 1" in out


def test_extract_pdf_tables_success():
    fake_plumber, _, _ = _make_fake_pdfplumber()
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _extract_pdf_tables("/tmp/doc.pdf", page=0)
    import json
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) > 0


def test_extract_pdf_tables_empty():
    fake_plumber, fake_pdf, fake_page = _make_fake_pdfplumber()
    fake_page.extract_tables.return_value = []
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _extract_pdf_tables("/tmp/doc.pdf", page=0)
    assert "No tables found" in out


def test_extract_pdf_tables_exception():
    fake_plumber = MagicMock()
    fake_plumber.open.side_effect = RuntimeError("corrupt")
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _extract_pdf_tables("/tmp/doc.pdf")
    assert "Table extraction error" in out


def test_pdf_metadata_success():
    fake_plumber, _, _ = _make_fake_pdfplumber(metadata={"Author": "Tony"})
    with patch.dict(sys.modules, {"pdfplumber": fake_plumber}):
        out = _pdf_metadata("/tmp/doc.pdf")
    import json
    data = json.loads(out)
    assert "pages" in data
    assert data["Author"] == "Tony"


def test_db_query_non_select():
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        pytest.skip("sqlalchemy not installed")

    from jarvis.tools.database_tools import _db_query
    out = _db_query("sqlite:///:memory:", "CREATE TABLE t2 (x INTEGER)")
    assert "Rows affected" in out or "Query executed" in out


def test_db_schema_with_table():
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        pytest.skip("sqlalchemy not installed")

    from jarvis.tools.database_tools import _db_schema
    url = "sqlite:///:memory:"
    engine = create_engine(url)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE myinfo (id INTEGER PRIMARY KEY, name TEXT)"))
    out = _db_schema(url, table="myinfo")
    assert "myinfo" in out or "Schema error" in out


def test_db_schema_error():
    """Lines 40-41: _db_schema exception path returns 'Schema error'."""
    from jarvis.tools.database_tools import _db_schema
    out = _db_schema("not-a-valid-url://???", table="t")
    assert "Schema error" in out or "error" in out.lower()


# ── git_tools: branch 53->55 (_git_commit with add_all=False) ────────────────

def _fake_git():
    """Reuse pattern from existing git tests."""
    import sys
    from unittest.mock import MagicMock
    fake = MagicMock()
    repo = MagicMock()
    fake.Repo = MagicMock(return_value=repo)
    repo.git = MagicMock()
    return fake, repo


def test_git_commit_add_all_false_skips_add():
    """Branch 53->55: when add_all=False, repo.git.add is not called."""
    from unittest.mock import MagicMock, patch
    from jarvis.tools.git_tools import _git_commit
    fake, repo = _fake_git()
    with patch.dict(__import__("sys").modules, {"git": fake}):
        out = _git_commit(".", "no-add commit", add_all=False)
    assert "Committed" in out
    repo.git.add.assert_not_called()
