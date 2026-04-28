import pytest

from src.sql_trace import built_in_queries, validate_event_name, validate_select_query


def test_sql_rejects_bad_event_name():
    with pytest.raises(ValueError, match="event name"):
        validate_event_name("pay'; DROP TABLE events; --")

    with pytest.raises(ValueError, match="event name"):
        built_in_queries("pay event")


def test_sql_rejects_non_select_query():
    with pytest.raises(ValueError, match="only SELECT queries are allowed"):
        validate_select_query("DELETE FROM events")

    with pytest.raises(ValueError, match="forbidden SQL operation"):
        validate_select_query("SELECT * FROM read_csv('secret.csv')")


def test_sql_applies_default_limit():
    safe_query = validate_select_query("SELECT * FROM events")

    assert safe_query.endswith("LIMIT 1000")
