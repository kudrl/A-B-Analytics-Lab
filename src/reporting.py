from __future__ import annotations

from typing import Optional

import pandas as pd

from src.analysis import BootstrapResult, ConversionTestResult, SRMResult, Verdict
from src.io import df_to_csv_bytes, make_zip_bytes, obj_to_json_bytes


def build_stats_payload(
    pay_event: str,
    srm: SRMResult,
    conv_test: ConversionTestResult,
    arpu_boot: BootstrapResult,
) -> dict:
    return {
        "pay_event": pay_event,
        "srm": {"ok": bool(srm.ok), "p_value": float(srm.p_value), "observed": dict(srm.observed)},
        "conversion": {
            "p_value": float(conv_test.p_value),
            "conv_a": float(conv_test.conv_a),
            "conv_b": float(conv_test.conv_b),
            "abs_diff": float(conv_test.abs_diff),
            "rel_lift": float(conv_test.rel_lift),
            "ci95_abs": [float(conv_test.ci95_abs[0]), float(conv_test.ci95_abs[1])],
        },
        "arpu_bootstrap": {
            "diff_mean": float(arpu_boot.diff_mean),
            "ci95": [float(arpu_boot.ci95[0]), float(arpu_boot.ci95[1])],
            "p_value_two_sided": float(arpu_boot.p_value_two_sided),
            "n_boot": int(arpu_boot.n_boot),
        },
    }


def build_run_meta(
    source: str,
    pay_event: str,
    scenario: Optional[str] = None,
    n_users: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    run_meta = {"source": source, "pay_event": pay_event}
    if source == "Generate synthetic":
        run_meta.update(
            {
                "scenario": scenario,
                "n_users": int(n_users) if n_users is not None else None,
                "seed": int(seed) if seed is not None else None,
            }
        )
    return run_meta


def build_report_md(pay_event: str, verdict: Verdict) -> str:
    return "\n".join(
        [
            "# A/B Analytics Lab - Exported Report",
            "",
            f"**Pay event:** `{pay_event}`",
            "",
            "## Auto conclusion",
            "",
            verdict.body.strip(),
            "",
            "## Included files",
            "- events.csv: raw events (event-level)",
            "- users.csv: user-level aggregation used for KPIs/tests",
            "- kpi.csv: KPI summary by variant",
            "- funnel.csv: funnel (if computed)",
            "- retention.csv: retention curve (if computed)",
            "- stats.json: SRM + conversion test + ARPU bootstrap",
            "- run_meta.json: run parameters for reproducibility",
            "",
        ]
    )


def make_export_bundle(
    *,
    df: pd.DataFrame,
    users: pd.DataFrame,
    kpis: pd.DataFrame,
    srm: SRMResult,
    conv_test: ConversionTestResult,
    arpu_boot: BootstrapResult,
    verdict: Verdict,
    pay_event: str,
    source: str,
    scenario: Optional[str] = None,
    n_users: Optional[int] = None,
    seed: Optional[int] = None,
    funnel_by_variant: Optional[pd.DataFrame] = None,
    ret_df: Optional[pd.DataFrame] = None,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {
        "events.csv": df_to_csv_bytes(df),
        "users.csv": df_to_csv_bytes(users),
        "kpi.csv": df_to_csv_bytes(kpis),
    }

    if funnel_by_variant is not None and len(funnel_by_variant) > 0:
        files["funnel.csv"] = df_to_csv_bytes(funnel_by_variant)

    if ret_df is not None and len(ret_df) > 0:
        files["retention.csv"] = df_to_csv_bytes(ret_df)

    files["stats.json"] = obj_to_json_bytes(build_stats_payload(pay_event, srm, conv_test, arpu_boot))
    files["run_meta.json"] = obj_to_json_bytes(build_run_meta(source, pay_event, scenario, n_users, seed))
    files["report.md"] = build_report_md(pay_event, verdict).encode("utf-8")
    files["ab_results.zip"] = make_zip_bytes(files)

    return files
