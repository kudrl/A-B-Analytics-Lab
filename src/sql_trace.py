from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import re

import pandas as pd


@dataclass(frozen=True)
class SQLResult:
    sql: str
    df: pd.DataFrame


def run_sql(events: pd.DataFrame, sql: str) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.register("events", events)
    return con.execute(sql).df()


def validate_event_name(event: str) -> str:
    event = str(event).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", event):
        raise ValueError("event name must contain only letters, digits, '_' or '-'")
    return event


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
