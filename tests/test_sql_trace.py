import pytest

from src.sql_trace import built_in_queries, validate_event_name


def test_sql_rejects_bad_event_name():
    with pytest.raises(ValueError, match="event name"):
        validate_event_name("pay'; DROP TABLE events; --")

    with pytest.raises(ValueError, match="event name"):
        built_in_queries("pay event")
