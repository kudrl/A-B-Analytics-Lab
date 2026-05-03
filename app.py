from __future__ import annotations

import hmac
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src import generator
from src.analysis import (
    arpu_bootstrap,
    build_user_table,
    check_srm,
    compute_basic_kpis,
    compute_funnel,
    compute_retention_curve,
    conversion_ztest_with_ci,
    generate_verdict,
)
from src.io import load_and_validate_csv
from src.reporting import make_export_bundle
from src.sql_trace import built_in_queries, run_sql, validate_event_name


MAX_SYNTHETIC_USERS = 100_000
ARPU_BOOTSTRAP_SAMPLES = 1_000


st.set_page_config(page_title="A/B Analytics Lab", layout="wide")
st.title("A/B Analytics Lab")


APP_PASSWORD = os.getenv("ABL_APP_PASSWORD")
ENABLE_SQL = os.getenv("ABL_ENABLE_SQL", "1").strip().lower() not in {"0", "false"}


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_money(value: float) -> str:
    return f"{value:,.2f}"


def _variant_metric(kpis: pd.DataFrame, variant: str, column: str, default: float = 0.0) -> float:
    indexed = kpis.set_index("variant")
    if variant not in indexed.index or column not in indexed.columns:
        return default
    return float(indexed.loc[variant, column])


def _prepare_viz_frame(events: pd.DataFrame) -> pd.DataFrame:
    viz = events.copy()
    viz["ts"] = pd.to_datetime(viz["ts"], errors="coerce")
    viz = viz.dropna(subset=["ts"])
    viz["date"] = viz["ts"].dt.floor("D")
    if "amount" not in viz.columns:
        viz["amount"] = 0.0
    return viz


@st.cache_data(show_spinner=False)
def _daily_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["date", "variant", "event"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
        .sort_values(["date", "variant", "event"])
    )


@st.cache_data(show_spinner=False)
def _daily_active_users(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["date", "variant"], as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "active_users"})
        .sort_values(["date", "variant"])
    )


@st.cache_data(show_spinner=False)
def _daily_revenue(events: pd.DataFrame, pay_event: str) -> pd.DataFrame:
    return (
        events.loc[events["event"].astype(str) == str(pay_event)]
        .groupby(["date", "variant"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "revenue"})
        .sort_values(["date", "variant"])
    )


def _stable_user_segment(events: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    segment_values = events[["user_id", segment_col]].dropna()
    if segment_values.empty:
        return pd.DataFrame(columns=["user_id", segment_col])
    unique_counts = segment_values.groupby("user_id")[segment_col].nunique(dropna=True)
    stable_users = unique_counts[unique_counts == 1].index
    return (
        segment_values[segment_values["user_id"].isin(stable_users)]
        .groupby("user_id", as_index=False)[segment_col]
        .first()
    )


def _build_conclusion_rows(
    kpis: pd.DataFrame,
    conv_test,
    arpu_boot,
) -> pd.DataFrame:
    conv_a = _variant_metric(kpis, "A", "conversion")
    conv_b = _variant_metric(kpis, "B", "conversion")
    arpu_a = _variant_metric(kpis, "A", "arpu")
    arpu_b = _variant_metric(kpis, "B", "arpu")
    return pd.DataFrame(
        [
            {
                "metric": "Conversion",
                "A": _format_pct(conv_a),
                "B": _format_pct(conv_b),
                "B - A": _format_pct(conv_test.abs_diff),
                "p-value": f"{conv_test.p_value:.4g}",
                "95% CI": f"{_format_pct(conv_test.ci95_abs[0])} to {_format_pct(conv_test.ci95_abs[1])}",
            },
            {
                "metric": "ARPU",
                "A": _format_money(arpu_a),
                "B": _format_money(arpu_b),
                "B - A": _format_money(arpu_b - arpu_a),
                "p-value": f"{arpu_boot.p_value_two_sided:.4g}",
                "95% CI": f"{_format_money(arpu_boot.ci95[0])} to {_format_money(arpu_boot.ci95[1])}",
            },
        ]
    )


def _decision_text(srm, conv_test, arpu_boot, kpis: pd.DataFrame) -> tuple[str, str]:
    arpu_a = _variant_metric(kpis, "A", "arpu")
    arpu_b = _variant_metric(kpis, "B", "arpu")
    conv_a = _variant_metric(kpis, "A", "conversion")
    conv_b = _variant_metric(kpis, "B", "conversion")
    arpu_ci_excludes_zero = arpu_boot.ci95[0] > 0 or arpu_boot.ci95[1] < 0

    if not srm.ok:
        return (
            "Hold decision",
            "Traffic split looks suspicious. Investigate SRM before trusting treatment effects.",
        )
    if arpu_ci_excludes_zero:
        if arpu_b > arpu_a:
            return "Ship B", "ARPU is higher for B and the bootstrap interval excludes zero."
        return "Do not ship B", "ARPU is lower for B and the bootstrap interval excludes zero."
    if conv_test.p_value < 0.05 and conv_b > conv_a:
        return "Ship B with caution", "Conversion is higher for B, while ARPU is not conclusive."
    if conv_test.p_value < 0.05 and conv_b < conv_a:
        return "Do not ship B", "Conversion is lower for B with statistical evidence."
    return "Keep testing", "No primary metric has enough evidence for a rollout decision."


def _source_sidebar() -> tuple[pd.DataFrame | None, str, str, str | None, int | None, int | None]:
    st.sidebar.header("Data")

    source = st.sidebar.selectbox("Source", ["Generate synthetic", "Upload CSV"], index=0)
    pay_event = st.sidebar.text_input("Pay event name", value="pay").strip() or "pay"
    try:
        pay_event = validate_event_name(pay_event)
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()

    scenario = None
    n_users = None
    seed = None
    events: pd.DataFrame | None = None

    if source == "Generate synthetic":
        st.sidebar.subheader("Synthetic settings")
        scenario = st.sidebar.selectbox(
            "Scenario",
            ["Conversion lift", "ARPU trade-off", "Simpson paradox"],
            index=0,
        )
        n_users = st.sidebar.slider("n_users", 500, MAX_SYNTHETIC_USERS, 20_000, step=500)
        seed = st.sidebar.number_input("seed", value=42, step=1)

        if scenario == "Conversion lift":
            base_conv = st.sidebar.slider("base_conv (A)", 0.001, 0.5, 0.10, step=0.001)
            lift_rel = st.sidebar.slider("lift_rel (B vs A)", 0.0, 2.0, 0.15, step=0.01)
            base_open = st.sidebar.slider("base_open (open_app prob)", 0.0, 1.0, 0.75, step=0.01)
            max_days = st.sidebar.slider("max_days", 3, 60, 14, step=1)
            events = generator.generate_conversion_lift(
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
            amount_mean_a = st.sidebar.slider("amount_mean_a", 1.0, 6.0, 2.8, step=0.05)
            amount_mean_b = st.sidebar.slider("amount_mean_b", 1.0, 6.0, 3.25, step=0.05)
            max_days = st.sidebar.slider("max_days", 3, 60, 14, step=1)
            events = generator.generate_arpu_tradeoff(
                n_users=int(n_users),
                conv_a=float(conv_a),
                conv_b=float(conv_b),
                amount_mean_a=float(amount_mean_a),
                amount_mean_b=float(amount_mean_b),
                max_days=int(max_days),
                seed=int(seed),
            )
        else:
            events = generator.generate_simpson_paradox(n_users=int(n_users), seed=int(seed))

        if pay_event != "pay":
            events = events.copy()
            events.loc[events["event"] == "pay", "event"] = pay_event
    else:
        st.sidebar.subheader("Upload CSV")
        uploaded = st.sidebar.file_uploader("events.csv", type=["csv"])
        if uploaded is not None:
            validation = load_and_validate_csv(uploaded)
            if not validation.ok:
                st.error(validation.error or "CSV validation failed")
                st.stop()
            events = validation.df

    return events, pay_event, source, scenario, n_users, seed


if APP_PASSWORD:
    st.sidebar.header("Access")
    st.session_state.setdefault("abl_authed", False)
    if not st.session_state["abl_authed"]:
        password = st.sidebar.text_input("Password", type="password")
        if password:
            if hmac.compare_digest(str(password), str(APP_PASSWORD)):
                st.session_state["abl_authed"] = True
            else:
                st.sidebar.error("Wrong password")
        st.stop()


df, pay_event, source, scenario, n_users, seed = _source_sidebar()

if df is None or len(df) == 0:
    st.info("Load data in the sidebar to start.")
    st.stop()

df_viz = _prepare_viz_frame(df)
if df_viz.empty:
    st.error("No valid timestamps found in the current data.")
    st.stop()

users = build_user_table(df, pay_event=pay_event)
if set(users["variant"].unique()) != {"A", "B"}:
    st.error("Data must contain both variants: A and B")
    st.stop()

kpis = compute_basic_kpis(users)
srm = check_srm(users)
conv_test = conversion_ztest_with_ci(users)
arpu_boot = arpu_bootstrap(users, n_boot=ARPU_BOOTSTRAP_SAMPLES)
verdict = generate_verdict(kpis=kpis, srm=srm, conv_test=conv_test, arpu_boot=arpu_boot)

funnel_by_variant = None
ret_df = None

tab_names = ["Dashboard", "Metrics", "Stats", "Conclusion"]
if ENABLE_SQL:
    tab_names.append("SQL Trace")
tab_names.append("Downloads")
tabs = dict(zip(tab_names, st.tabs(tab_names)))


with tabs["Dashboard"]:
    st.subheader("Experiment dashboard")

    date_min = df_viz["date"].min().date()
    date_max = df_viz["date"].max().date()
    segment_candidates = sorted(
        column
        for column in df_viz.columns
        if column not in {"user_id", "variant", "ts", "event", "amount", "date"}
    )

    controls = st.columns([1.2, 1, 1, 1])
    with controls[0]:
        dashboard_goal = st.selectbox(
            "View",
            [
                "Overview",
                "Event timeline",
                "Daily conversion",
                "Revenue and ARPU",
                "Funnel",
                "Retention",
                "Data quality",
            ],
            index=0,
        )
    with controls[1]:
        selected_event = st.selectbox(
            "Event",
            sorted(df_viz["event"].astype(str).unique().tolist()),
            index=0,
        )
    with controls[2]:
        segment_col = st.selectbox("Segment", ["None"] + segment_candidates, index=0)
    with controls[3]:
        max_rows = st.number_input("Rows", min_value=20, max_value=500, value=100, step=20)

    date_controls = st.columns(2)
    with date_controls[0]:
        date_from = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    with date_controls[1]:
        date_to = st.date_input("To", value=date_max, min_value=date_min, max_value=date_max)

    if date_from > date_to:
        st.error("From date must be earlier than or equal to To date.")
        st.stop()

    df_window = df_viz[
        (df_viz["date"] >= pd.Timestamp(date_from)) & (df_viz["date"] <= pd.Timestamp(date_to))
    ].copy()

    if df_window.empty:
        st.info("No events in the selected date window.")
        st.stop()

    users_window = build_user_table(df_window, pay_event=pay_event)
    kpis_window = compute_basic_kpis(users_window)

    conv_a = _variant_metric(kpis_window, "A", "conversion")
    conv_b = _variant_metric(kpis_window, "B", "conversion")
    arpu_a = _variant_metric(kpis_window, "A", "arpu")
    arpu_b = _variant_metric(kpis_window, "B", "arpu")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Users", f"{int(users_window['user_id'].nunique()):,}")
    summary_cols[1].metric("Conversion B - A", _format_pct(conv_b - conv_a))
    summary_cols[2].metric("ARPU B - A", _format_money(arpu_b - arpu_a))
    summary_cols[3].metric("Events", f"{len(df_window):,}")

    if dashboard_goal == "Overview":
        st.dataframe(kpis_window, width="stretch", hide_index=True)
        overview_chart = kpis_window.melt(
            id_vars="variant",
            value_vars=["conversion", "arpu", "arppu"],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            overview_chart,
            x="metric",
            y="value",
            color="variant",
            barmode="group",
            title="Core metrics by variant",
        )
        st.plotly_chart(fig, width="stretch")

    elif dashboard_goal == "Event timeline":
        daily = _daily_event_counts(df_window)
        daily = daily.loc[daily["event"].astype(str) == str(selected_event)]
        if daily.empty:
            st.info("No rows for the selected event.")
        else:
            fig = px.line(
                daily,
                x="date",
                y="events",
                color="variant",
                markers=True,
                title=f"Daily {selected_event} events",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(daily.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Daily conversion":
        active = _daily_active_users(df_window)
        payers = (
            df_window.loc[df_window["event"].astype(str) == str(pay_event)]
            .groupby(["date", "variant"], as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "payers"})
        )
        daily = active.merge(payers, on=["date", "variant"], how="left").fillna({"payers": 0})
        daily["conversion_proxy"] = daily["payers"] / daily["active_users"].where(
            daily["active_users"] > 0, pd.NA
        )
        fig = px.line(
            daily,
            x="date",
            y="conversion_proxy",
            color="variant",
            markers=True,
            title="Daily payer / active user proxy",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(daily.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Revenue and ARPU":
        active = _daily_active_users(df_window)
        revenue = _daily_revenue(df_window, pay_event)
        daily = active.merge(revenue, on=["date", "variant"], how="left").fillna({"revenue": 0.0})
        daily["arpu_proxy"] = daily["revenue"] / daily["active_users"].where(
            daily["active_users"] > 0, pd.NA
        )
        revenue_chart, arpu_chart = st.columns(2)
        with revenue_chart:
            st.plotly_chart(
                px.line(daily, x="date", y="revenue", color="variant", markers=True, title="Revenue"),
                width="stretch",
            )
        with arpu_chart:
            st.plotly_chart(
                px.line(daily, x="date", y="arpu_proxy", color="variant", markers=True, title="ARPU proxy"),
                width="stretch",
            )
        st.dataframe(daily.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Funnel":
        all_events = sorted(df_window["event"].astype(str).unique().tolist())
        default_steps = [step for step in ["signup", pay_event] if step in all_events]
        steps = st.multiselect("Steps", options=all_events, default=default_steps, key="dashboard_funnel")
        if len(steps) < 2:
            st.info("Select at least two ordered steps.")
        else:
            funnel = compute_funnel(df_window, steps=steps).by_variant
            st.dataframe(funnel, width="stretch", hide_index=True)
            fig = px.bar(
                funnel,
                x="step",
                y="users_reached",
                color="variant",
                barmode="group",
                title="Users reached by funnel step",
            )
            st.plotly_chart(fig, width="stretch")

    elif dashboard_goal == "Retention":
        retention_day = st.slider("Max day", 7, 60, 14, key="dashboard_retention_day")
        active_options = ["Any event"] + sorted(df_window["event"].astype(str).unique().tolist())
        active_event = st.selectbox("Active event", active_options, index=0)
        active_event_value = None if active_event == "Any event" else active_event
        retention = compute_retention_curve(df_window, active_event=active_event_value, max_day=retention_day)
        fig = px.line(retention, x="day", y="retention", color="variant", markers=True, title="Retention")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(retention.head(int(max_rows)), width="stretch", hide_index=True)

    else:
        quality_cols = st.columns(4)
        quality_cols[0].metric("Rows", f"{len(df_window):,}")
        quality_cols[1].metric("Users", f"{df_window['user_id'].nunique():,}")
        quality_cols[2].metric("Events", f"{df_window['event'].nunique():,}")
        quality_cols[3].metric("Missing amount", f"{int(df_window['amount'].isna().sum()):,}")
        top_events = df_window["event"].astype(str).value_counts().head(20).reset_index()
        top_events.columns = ["event", "rows"]
        st.dataframe(top_events, width="stretch", hide_index=True)

    if segment_col != "None":
        st.divider()
        st.subheader("Segment cut")
        user_segment = _stable_user_segment(df_window, segment_col)
        segmented_users = users_window.merge(user_segment, on="user_id", how="inner")
        if segmented_users.empty:
            st.info("No stable user-level segment values found for the selected column.")
        else:
            segment_kpis = (
                segmented_users.groupby([segment_col, "variant"], as_index=False)
                .agg(
                    n_users=("user_id", "nunique"),
                    conversion=("is_payer", "mean"),
                    arpu=("revenue", "mean"),
                )
                .sort_values(["n_users", segment_col], ascending=[False, True])
            )
            st.dataframe(segment_kpis.head(int(max_rows)), width="stretch", hide_index=True)
            fig = px.bar(
                segment_kpis.head(30),
                x=segment_col,
                y="conversion",
                color="variant",
                barmode="group",
                title="Conversion by segment",
            )
            st.plotly_chart(fig, width="stretch")


with tabs["Metrics"]:
    st.subheader("User-level KPIs")
    st.dataframe(kpis, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Funnel")
    all_events = sorted(df["event"].astype(str).unique().tolist())
    default_steps = [step for step in ["signup", pay_event] if step in all_events]
    steps = st.multiselect("Steps", options=all_events, default=default_steps, key="metrics_funnel")
    if len(steps) >= 2:
        funnel_result = compute_funnel(df, steps=steps)
        funnel_by_variant = funnel_result.by_variant
        st.dataframe(funnel_by_variant, width="stretch", hide_index=True)
    else:
        st.info("Select at least two ordered steps.")

    st.divider()
    st.subheader("Retention")
    max_day = st.slider("Max day", 7, 60, 14, key="metrics_retention_day")
    try:
        ret_df = compute_retention_curve(df, active_event=None, max_day=max_day)
        st.dataframe(ret_df, width="stretch", hide_index=True)
    except Exception as error:
        ret_df = None
        st.error(f"Retention calculation failed: {error}")


with tabs["Stats"]:
    st.subheader("SRM check")
    srm_cols = st.columns(3)
    srm_cols[0].metric("Status", "OK" if srm.ok else "Check")
    srm_cols[1].metric("p-value", f"{srm.p_value:.4g}")
    srm_cols[2].metric("Split", f"A={srm.observed.get('A', 0):,} / B={srm.observed.get('B', 0):,}")

    st.divider()
    st.subheader("Conversion z-test")
    st.json(
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
    st.json(
        {
            "diff_mean": float(arpu_boot.diff_mean),
            "ci95": [float(arpu_boot.ci95[0]), float(arpu_boot.ci95[1])],
            "p_value_two_sided": float(arpu_boot.p_value_two_sided),
            "n_boot": int(arpu_boot.n_boot),
        }
    )


with tabs["Conclusion"]:
    decision, explanation = _decision_text(srm, conv_test, arpu_boot, kpis)
    st.subheader("Conclusion")

    decision_cols = st.columns([1.2, 2])
    with decision_cols[0]:
        st.metric("Decision", decision)
    with decision_cols[1]:
        st.info(explanation)

    st.dataframe(_build_conclusion_rows(kpis, conv_test, arpu_boot), width="stretch", hide_index=True)

    st.divider()
    st.subheader("What to check before rollout")
    checklist = [
        ("Traffic split", "Pass" if srm.ok else "Needs investigation"),
        ("Conversion evidence", "Pass" if conv_test.p_value < 0.05 else "Not conclusive"),
        (
            "ARPU evidence",
            "Pass" if (arpu_boot.ci95[0] > 0 or arpu_boot.ci95[1] < 0) else "Not conclusive",
        ),
    ]
    st.dataframe(pd.DataFrame(checklist, columns=["check", "status"]), width="stretch", hide_index=True)


if ENABLE_SQL:
    with tabs["SQL Trace"]:
        st.subheader("SQL Trace")
        st.caption("Table name: events. Only read-only SELECT/WITH queries are allowed.")
        presets = built_in_queries(pay_event=pay_event)
        preset_name = st.selectbox("Preset", list(presets.keys()), index=0)
        sql_text = st.text_area("SQL", value=presets[preset_name], height=180)

        if st.button("Run SQL", type="primary"):
            try:
                output = run_sql(df, sql_text)
                st.dataframe(output, width="stretch", hide_index=True)
            except Exception as error:
                st.error(f"SQL query failed: {error}")

        st.divider()
        st.subheader("Raw events preview")
        st.dataframe(df.head(200), width="stretch", hide_index=True)


with tabs["Downloads"]:
    st.subheader("Export results")
    st.caption("Download raw data, computed tables, metadata, and a markdown report.")

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

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button("events.csv", files["events.csv"], "events.csv", "text/csv")
        st.download_button("users.csv", files["users.csv"], "users.csv", "text/csv")
    with download_cols[1]:
        st.download_button("kpi.csv", files["kpi.csv"], "kpi.csv", "text/csv")
        if "funnel.csv" in files:
            st.download_button("funnel.csv", files["funnel.csv"], "funnel.csv", "text/csv")
    with download_cols[2]:
        if "retention.csv" in files:
            st.download_button("retention.csv", files["retention.csv"], "retention.csv", "text/csv")
        st.download_button("stats.json", files["stats.json"], "stats.json", "application/json")

    st.download_button("report.md", files["report.md"], "report.md", "text/markdown")
    st.download_button("run_meta.json", files["run_meta.json"], "run_meta.json", "application/json")
    st.download_button(
        "ab_results.zip",
        data=files["ab_results.zip"],
        file_name="ab_results.zip",
        mime="application/zip",
    )
