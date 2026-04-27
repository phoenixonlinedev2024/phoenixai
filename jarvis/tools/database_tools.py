"""Database tools — query any SQL database via SQLAlchemy."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _db_query(connection_url: str, sql: str, max_rows: int = 100) -> str:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(connection_url, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            if result.returns_rows:
                cols = list(result.keys())
                rows = [dict(zip(cols, row)) for row in result.fetchmany(max_rows)]
                return json.dumps({"columns": cols, "rows": rows, "count": len(rows)}, default=str, indent=2)
            return f"Query executed. Rows affected: {result.rowcount}"
    except ImportError:
        return "SQLAlchemy not installed. Run: pip install sqlalchemy"
    except Exception as exc:
        return f"DB query error: {exc}"


def _db_schema(connection_url: str, table: str = "") -> str:
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(connection_url)
        inspector = inspect(engine)
        if table:
            cols = inspector.get_columns(table)
            pk = inspector.get_pk_constraint(table)
            fks = inspector.get_foreign_keys(table)
            return json.dumps({"table": table, "columns": cols, "pk": pk, "fks": fks}, default=str, indent=2)
        tables = inspector.get_table_names()
        return "Tables: " + ", ".join(tables)
    except Exception as exc:
        return f"Schema error: {exc}"


def _db_execute(connection_url: str, sql: str) -> str:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(connection_url)
        with engine.begin() as conn:
            result = conn.execute(text(sql))
            return f"Executed. Rows affected: {result.rowcount}"
    except Exception as exc:
        return f"Execute error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="db_query", description="Execute a SELECT query on any SQL database (SQLite, PostgreSQL, MySQL, etc.). connection_url format: 'postgresql://user:pass@host/db'.",
        input_schema={"type":"object","properties":{"connection_url":{"type":"string"},"sql":{"type":"string"},"max_rows":{"type":"integer","default":100}},"required":["connection_url","sql"]},
        fn=_db_query, category="api"))
    registry.register(Tool(name="db_schema", description="Inspect a database schema: list tables or describe a table's columns.",
        input_schema={"type":"object","properties":{"connection_url":{"type":"string"},"table":{"type":"string"}},"required":["connection_url"]},
        fn=_db_schema, category="api"))
    registry.register(Tool(name="db_execute", description="Execute a non-SELECT SQL statement (INSERT, UPDATE, DELETE, CREATE).",
        input_schema={"type":"object","properties":{"connection_url":{"type":"string"},"sql":{"type":"string"}},"required":["connection_url","sql"]},
        fn=_db_execute, category="api"))
