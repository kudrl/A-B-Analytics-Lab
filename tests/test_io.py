from io import StringIO

import pandas as pd

from src.io import (
    MAX_UPLOAD_COLUMNS,
    MAX_UPLOAD_ROWS,
    escape_excel_formula,
    load_and_validate_csv,
    sanitize_csv_export,
)


def test_validate_csv_missing_columns():
    csv = StringIO("user_id,variant,ts\n1,A,2024-01-01\n")

    result = load_and_validate_csv(csv)

    assert not result.ok
    assert result.df is None
    assert "Missing columns" in (result.error or "")


def test_validate_csv_too_many_rows():
    lines = ["user_id,variant,ts,event"] + [
        f"{index},A,2024-01-01,signup" for index in range(MAX_UPLOAD_ROWS + 1)
    ]
    csv = StringIO("\n".join(lines))

    result = load_and_validate_csv(csv)

    assert not result.ok
    assert result.error == "File has too many rows"


def test_validate_csv_too_many_columns():
    extra_columns = [f"extra_{index}" for index in range(MAX_UPLOAD_COLUMNS - 3)]
    header = ["user_id", "variant", "ts", "event", *extra_columns]
    row = ["1", "A", "2024-01-01", "signup", *(["x"] * len(extra_columns))]
    csv = StringIO(",".join(header) + "\n" + ",".join(row) + "\n")

    result = load_and_validate_csv(csv)

    assert not result.ok
    assert result.error == "File has too many columns"


def test_escape_excel_formula_in_csv_export():
    df = pd.DataFrame({"event": ["=SUM(A1:A2)", "safe"], "user_id": ["@cmd", "42"]})

    safe_df = sanitize_csv_export(df)

    assert safe_df.loc[0, "event"] == "'=SUM(A1:A2)"
    assert safe_df.loc[0, "user_id"] == "'@cmd"
    assert escape_excel_formula("safe") == "safe"
