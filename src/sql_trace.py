from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import re

import pandas as pd


SQL_FORBIDDEN_PATTERNS = (
    "attach",
    "copy",
    "create",
    "delete",
    "drop",
    "export",
    "insert",
    "install",
    "load",
    "merge",
    "pragma",
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "replace",
    "update",
)
SQL_RESULT_LIMIT = 1_000


@dataclass(frozen=True)
class SQLResult:
    sql: str
    df: pd.DataFrame


def run_sql(events: pd.DataFrame, sql: str) -> pd.DataFrame:
    import duckdb

    safe_sql = validate_select_query(sql)
    con = duckdb.connect(database=":memory:")
    con.register("events", events)
    return con.execute(safe_sql).df()


def validate_event_name(event: str) -> str:
    event = str(event).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", event):
        raise ValueError("event name must contain only letters, digits, '_' or '-'")
    return event


def validate_select_query(sql: str) -> str:
    query = str(sql).strip()
    normalized = query.lower()

    if not normalized:
        raise ValueError("query must not be empty")
    if ";" in query.rstrip(";"):
        raise ValueError("only a single SELECT statement is allowed")
    if not normalized.startswith("select") and not normalized.startswith("with"):
        raise ValueError("only SELECT queries are allowed")
    if any(re.search(rf"\b{pattern}\b", normalized) for pattern in SQL_FORBIDDEN_PATTERNS):
        raise ValueError("query contains a forbidden SQL operation")
    if re.search(r"\blimit\b", normalized):
        return query
    return f"{query.rstrip(';')} LIMIT {SQL_RESULT_LIMIT}"


def built_in_queries(pay_event: str = "pay") -> Dict[str, str]:
    pay_event = validate_event_name(pay_event)
    return {
        "Users per variant": """
            SELECT variant, COUNT(DISTINCT user_id) AS n_users
            FROM events
            GROUP BY variant
            ORDER BY variant;
        """,
        "Conversion to pay": f"""
            WITH users AS (
              SELECT user_id, ANY_VALUE(variant) AS variant
              FROM events
              GROUP BY user_id
            ),
            payers AS (
              SELECT DISTINCT user_id
              FROM events
              WHERE event = '{pay_event}'
            )
            SELECT
              u.variant,
              COUNT(*) AS n_users,
              SUM(CASE WHEN p.user_id IS NOT NULL THEN 1 ELSE 0 END) AS paying_users,
              1.0 * SUM(CASE WHEN p.user_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) AS conversion
            FROM users u
            LEFT JOIN payers p USING (user_id)
            GROUP BY u.variant
            ORDER BY u.variant;
        """,
        "ARPU (user-level revenue)": f"""
            WITH users AS (
              SELECT user_id, ANY_VALUE(variant) AS variant
              FROM events
              GROUP BY user_id
            ),
            rev AS (
              SELECT user_id, SUM(COALESCE(amount,0)) AS revenue
              FROM events
              WHERE event = '{pay_event}'
              GROUP BY user_id
            )
            SELECT
              u.variant,
              AVG(COALESCE(r.revenue,0)) AS arpu,
              SUM(COALESCE(r.revenue,0)) AS total_revenue
            FROM users u
            LEFT JOIN rev r USING (user_id)
            GROUP BY u.variant
            ORDER BY u.variant;
        """,
        "Daily events count": """
            SELECT DATE_TRUNC('day', CAST(ts AS TIMESTAMP)) AS day, variant, event, COUNT(*) AS n
            FROM events
            GROUP BY 1,2,3
            ORDER BY day, variant, event;
        """,
    }
