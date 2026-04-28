from __future__ import annotations

import streamlit as st
import pandas as pd

from src.io import load_and_validate_csv
from src import generator
from src.analysis import (
    build_user_table,
    compute_basic_kpis,
    compute_funnel,
    compute_retention_curve,
    check_srm,
    conversion_ztest_with_ci,
    arpu_bootstrap,
    generate_verdict,
)
from src.sql_trace import built_in_queries, run_sql
from src.reporting import make_export_bundle


st.set_page_config(page_title="A/B Analytics Lab", layout="wide")
st.title("A/B Analytics Lab")


# -----------------------------
# Source
# -----------------------------
st.sidebar.header("Data")

source = st.sidebar.selectbox("Source", ["Generate synthetic", "Upload CSV"], index=0)

pay_event = st.sidebar.text_input("Pay event name", value="pay").strip() or "pay"

df: pd.DataFrame | None = None
scenario = None
n_users = None
seed = None

if source == "Generate synthetic":
    st.sidebar.subheader("Synthetic settings")

    scenario = st.sidebar.selectbox(
        "Scenario",
        ["Conversion lift", "ARPU trade-off", "Simpson paradox"],
        index=0,
    )
    n_users = st.sidebar.slider("n_users", 500, 200_000, 20_000, step=500)
    seed = st.sidebar.number_input("seed", value=42, step=1)

    if scenario == "Conversion lift":
        base_conv = st.sidebar.slider("base_conv (A)", 0.001, 0.5, 0.10, step=0.001)
        lift_rel = st.sidebar.slider("lift_rel (B vs A)", 0.0, 2.0, 0.15, step=0.01)
        base_open = st.sidebar.slider("base_open (open_app prob)", 0.0, 1.0, 0.75, step=0.01)
        max_days = st.sidebar.slider("max_days", 3, 60, 14, step=1)

        df = generator.generate_conversion_lift(
            n_users=int(n_users),
            base_conv=float(base_conv),
            lift_rel=float(lift_rel),
            base_open=float(base_open),
            max_days=int(max_days),
            seed=int(seed),
        )

    elif scenario == "ARPU trade-off":
        conv_a = st.sidebar.slider("conv_a", 0.001, 0.5, 0.12, step=0.001)
        conv_b = st.sidebar.slider("conv_b", 0.001, 0.5, 0.10, step=0.001)
        amount_mean_a = st.sidebar.slider(
            "amount_mean_a (lognormal mean)", 1.0, 6.0, 2.8, step=0.05
        )
        amount_mean_b = st.sidebar.slider(
            "amount_mean_b (lognormal mean)", 1.0, 6.0, 3.25, step=0.05
        )
        max_days = st.sidebar.slider("max_days", 3, 60, 14, step=1)

        df = generator.generate_arpu_tradeoff(
            n_users=int(n_users),
            conv_a=float(conv_a),
            conv_b=float(conv_b),
            amount_mean_a=float(amount_mean_a),
            amount_mean_b=float(amount_mean_b),
            max_days=int(max_days),
            seed=int(seed),
        )

    else:  # симпс
        df = generator.generate_simpson_paradox(
            n_users=int(n_users),
            seed=int(seed),
        )

   
    if pay_event != "pay":
        df = df.copy()
        df.loc[df["event"] == "pay", "event"] = pay_event

else:
    st.sidebar.subheader("Upload CSV")
    uploaded = st.sidebar.file_uploader("events.csv", type=["csv"])
    if uploaded is not None:
        vr = load_and_validate_csv(uploaded)
        if not vr.ok:
            st.error(vr.error or "CSV validation failed")
            st.stop()
        df = vr.df


if df is None or len(df) == 0:
    st.info("Load data in the sidebar to start.")
    st.stop()


# -----------------------------
# ядро вычислений
# -----------------------------
users = build_user_table(df, pay_event=pay_event)
kpis = compute_basic_kpis(users)

srm = check_srm(users)
conv_test = conversion_ztest_with_ci(users)
arpu_boot = arpu_bootstrap(users)

verdict = generate_verdict(
    kpis=kpis,
    srm=srm,
    conv_test=conv_test,
    arpu_boot=arpu_boot,
)

funnel_by_variant = None
ret_df = None


# -----------------------------
# Tabs
# -----------------------------
tab_metrics, tab_stats, tab_conc, tab_sql, tab_dl = st.tabs(
    ["Metrics", "Stats", "Conclusion", "SQL Trace", "Downloads"]
)

with tab_metrics:
    st.subheader("KPIs (user-level)")
    st.dataframe(kpis, use_container_width=True)

    st.divider()
    st.subheader("Funnel")

    all_events = sorted(df["event"].astype(str).unique().tolist())

    default_steps = []
    if "signup" in all_events:
        default_steps.append("signup")
    if pay_event in all_events:
        default_steps.append(pay_event)

    steps = st.multiselect(
        "Funnel steps (ordered)",
        options=all_events,
        default=default_steps,
        key="funnel_steps",
    )

    if len(steps) >= 2:
        funnel_res = compute_funnel(df, steps=steps)
        funnel_by_variant = funnel_res.by_variant
        st.dataframe(funnel_by_variant, use_container_width=True)
    else:
        st.info("Select at least 2 steps to compute funnel.")

    st.divider()
    st.subheader("Retention (cohort)")

    max_day = st.slider("Max day", 7, 60, 14, key="ret_max_day")
    try:
        ret_df = compute_retention_curve(df, active_event=None, max_day=max_day)
        st.dataframe(ret_df, use_container_width=True)
    except Exception as e:
        ret_df = None
        st.error(f"Retention failed: {e}")


with tab_stats:
    st.subheader("SRM check (50/50)")

    st.write(
        {
            "ok": bool(srm.ok),
            "p_value": float(srm.p_value),
            "observed": dict(srm.observed),
        }
    )

    st.divider()
    st.subheader("Conversion test (z-test)")

    st.write(
        {
            "p_value": float(conv_test.p_value),
            "conv_a": float(conv_test.conv_a),
            "conv_b": float(conv_test.conv_b),
            "abs_diff": float(conv_test.abs_diff),
            "rel_lift": float(conv_test.rel_lift),
            "ci95_abs": [float(conv_test.ci95_abs[0]), float(conv_test.ci95_abs[1])],
        }
    )

    st.divider()
    st.subheader("ARPU bootstrap")

    st.write(
        {
            "diff_mean": float(arpu_boot.diff_mean),
            "ci95": [float(arpu_boot.ci95[0]), float(arpu_boot.ci95[1])],
            "p_value_two_sided": float(arpu_boot.p_value_two_sided),
            "n_boot": int(arpu_boot.n_boot),
        }
    )


with tab_conc:
    st.subheader("Auto conclusion")
    st.markdown(f"### {verdict.title}")
    st.write(verdict.body)


with tab_sql:
    st.subheader("SQL Trace (DuckDB over current events table)")
    st.caption("Table name: events")

    try:
        presets = built_in_queries(pay_event=pay_event)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    preset_name = st.selectbox("Preset", list(presets.keys()), index=0)
    sql_text = st.text_area("SQL", value=presets[preset_name], height=180)

    if st.button("Run SQL", type="primary"):
        try:
            out = run_sql(df, sql_text)
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(f"SQL error: {e}")

    st.divider()
    st.subheader("Raw events preview")
    st.dataframe(df.head(200), use_container_width=True)


with tab_dl:
    st.subheader("Export results")
    st.caption("Download raw + computed tables to re-check everything outside the app.")

    files = make_export_bundle(
        df=df,
        users=users,
        kpis=kpis,
        srm=srm,
        conv_test=conv_test,
        arpu_boot=arpu_boot,
        verdict=verdict,
        pay_event=pay_event,
        source=source,
        scenario=scenario,
        n_users=int(n_users) if n_users is not None else None,
        seed=int(seed) if seed is not None else None,
        funnel_by_variant=funnel_by_variant,
        ret_df=ret_df,
    )

    st.markdown("### Download individual files")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ events.csv", files["events.csv"], "events.csv", "text/csv")
        st.download_button("⬇️ users.csv", files["users.csv"], "users.csv", "text/csv")
    with c2:
        st.download_button("⬇️ kpi.csv", files["kpi.csv"], "kpi.csv", "text/csv")
        if "funnel.csv" in files:
            st.download_button("⬇️ funnel.csv", files["funnel.csv"], "funnel.csv", "text/csv")
    with c3:
        if "retention.csv" in files:
            st.download_button("⬇️ retention.csv", files["retention.csv"], "retention.csv", "text/csv")
        st.download_button("⬇️ stats.json", files["stats.json"], "stats.json", "application/json")

    st.download_button("⬇️ report.md", files["report.md"], "report.md", "text/markdown")
    st.download_button("⬇️ run_meta.json", files["run_meta.json"], "run_meta.json", "application/json")

    st.divider()
    st.markdown("### Download everything as one archive")

    st.download_button(
        "⬇️ ab_results.zip",
        data=files["ab_results.zip"],
        file_name="ab_results.zip",
        mime="application/zip",
    )
