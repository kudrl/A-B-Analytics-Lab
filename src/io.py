from __future__ import annotations



from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import io
import json
import zipfile

import pandas as pd

REQUIRED_COLS: Set[str] = {"user_id", "variant", "ts", "event"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 200_000
MAX_UPLOAD_COLUMNS = 50


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    df: Optional[pd.DataFrame]
    error: Optional[str]


def _get_uploaded_file_size(file) -> Optional[int]:
    size = getattr(file, "size", None)
    if isinstance(size, int):
        return size
    if hasattr(file, "getbuffer"):
        try:
            return int(file.getbuffer().nbytes)
        except Exception:
            return None
    return None


def load_and_validate_csv(file) -> ValidationResult:
    file_size = _get_uploaded_file_size(file)
    if file_size is not None and file_size > MAX_UPLOAD_BYTES:
        return ValidationResult(False, None, "File too large")

    try:
        df = pd.read_csv(file, nrows=MAX_UPLOAD_ROWS + 1)
    except Exception as e:
        return ValidationResult(False, None, f"Failed to read CSV: {e}")

    if len(df) > MAX_UPLOAD_ROWS:
        return ValidationResult(False, None, "File has too many rows")
    if len(df.columns) > MAX_UPLOAD_COLUMNS:
        return ValidationResult(False, None, "File has too many columns")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return ValidationResult(False, None, f"Missing columns: {sorted(missing)}")

    df = df.copy()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].isna().any():
        return ValidationResult(False, None, "Column 'ts' has invalid datetime values")

    df["variant"] = df["variant"].astype(str).str.upper().str.strip()
    if not set(df["variant"].unique()).issubset({"A", "B"}):
        return ValidationResult(False, None, "Column 'variant' must contain only 'A'/'B'")

    if "amount" not in df.columns:
        df["amount"] = 0.0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df["event"] = df["event"].astype(str).str.strip()
    df["date"] = df["ts"].dt.floor("D")

    return ValidationResult(True, df, None)


def escape_excel_formula(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def sanitize_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_columns = out.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        out[column] = out[column].map(escape_excel_formula)
    return out


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    safe_df = sanitize_csv_export(df)
    return safe_df.to_csv(index=False).encode("utf-8")


def obj_to_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def make_zip_bytes(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()
