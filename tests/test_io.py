from io import StringIO

from src.io import load_and_validate_csv


def test_validate_csv_missing_columns():
    csv = StringIO("user_id,variant,ts\n1,A,2024-01-01\n")

    result = load_and_validate_csv(csv)

    assert not result.ok
    assert result.df is None
    assert "Missing columns" in (result.error or "")
