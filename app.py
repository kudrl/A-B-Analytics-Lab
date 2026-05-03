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


st.set_page_config(page_title="Лаборатория A/B", layout="wide")
st.title("Лаборатория A/B")


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
                "метрика": "Конверсия",
                "A": _format_pct(conv_a),
                "B": _format_pct(conv_b),
                "B − A": _format_pct(conv_test.abs_diff),
                "p-value": f"{conv_test.p_value:.4g}",
                "95% ДИ": f"{_format_pct(conv_test.ci95_abs[0])} … {_format_pct(conv_test.ci95_abs[1])}",
            },
            {
                "метрика": "ARPU",
                "A": _format_money(arpu_a),
                "B": _format_money(arpu_b),
                "B − A": _format_money(arpu_b - arpu_a),
                "p-value": f"{arpu_boot.p_value_two_sided:.4g}",
                "95% ДИ": f"{_format_money(arpu_boot.ci95[0])} … {_format_money(arpu_boot.ci95[1])}",
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
            "Пауза",
            "Распределение трафика выглядит подозрительно (SRM). Сначала разберите SRM, потом доверяйте эффектам.",
        )
    if arpu_ci_excludes_zero:
        if arpu_b > arpu_a:
            return "Запускать B", "ARPU выше в B, и бутстрэп-интервал исключает ноль."
        return "Не запускать B", "ARPU ниже в B, и бутстрэп-интервал исключает ноль."
    if conv_test.p_value < 0.05 and conv_b > conv_a:
        return "Запускать B с осторожностью", "Конверсия выше в B, но ARPU пока неубедителен."
    if conv_test.p_value < 0.05 and conv_b < conv_a:
        return "Не запускать B", "Конверсия ниже в B со статистическим подтверждением."
    return "Продолжать тест", "Недостаточно доказательств по ключевым метрикам для решения о запуске."


def _source_sidebar() -> tuple[pd.DataFrame | None, str, str, str | None, int | None, int | None]:
    st.sidebar.header("Данные")

    source = st.sidebar.selectbox("Источник", ["Синтетика", "Загрузить CSV"], index=0)
    pay_event = st.sidebar.text_input("Событие оплаты (pay)", value="pay").strip() or "pay"
    try:
        pay_event = validate_event_name(pay_event)
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()

    scenario = None
    n_users = None
    seed = None
    events: pd.DataFrame | None = None

    if source == "Синтетика":
        st.sidebar.subheader("Настройки синтетики")
        with st.sidebar.expander("Что регулируют параметры?"):
            st.caption(
                "Синтетика генерирует события для двух вариантов (A и B). "
                "Параметры ниже задают базовые вероятности/уровни для A и отличия для B."
            )
        scenario = st.sidebar.selectbox(
            "Сценарий",
            ["Рост конверсии", "Компромисс ARPU", "Парадокс Симпсона"],
            index=0,
        )
        n_users = st.sidebar.slider("Пользователей", 500, MAX_SYNTHETIC_USERS, 20_000, step=500)
        seed = st.sidebar.number_input("Seed", value=42, step=1)

        if scenario == "Рост конверсии":
            base_conv = st.sidebar.slider(
                "Конверсия A",
                0.001,
                0.5,
                0.10,
                step=0.001,
                help="Вероятность события оплаты (pay) для варианта A.",
            )
            lift_rel = st.sidebar.slider(
                "Отн. рост B к A",
                0.0,
                2.0,
                0.15,
                step=0.01,
                help="Относительное изменение конверсии B относительно A. Например 0.15 = +15% к конверсии A.",
            )
            st.sidebar.metric("Конверсия B (расчёт)", f"{(float(base_conv) * (1 + float(lift_rel))):.2%}")
            base_open = st.sidebar.slider(
                "Вероятность open_app",
                0.0,
                1.0,
                0.75,
                step=0.01,
                help="Вероятность события open_app в выбранные дни активности.",
            )
            max_days = st.sidebar.slider("Горизонт (дни)", 3, 60, 14, step=1)
            events = generator.generate_conversion_lift(
                n_users=int(n_users),
                base_conv=float(base_conv),
                lift_rel=float(lift_rel),
                base_open=float(base_open),
                max_days=int(max_days),
                seed=int(seed),
            )
        elif scenario == "Компромисс ARPU":
            conv_a = st.sidebar.slider(
                "Конверсия A",
                0.001,
                0.5,
                0.12,
                step=0.001,
                help="Вероятность события оплаты (pay) для варианта A.",
            )
            conv_b = st.sidebar.slider(
                "Конверсия B",
                0.001,
                0.5,
                0.10,
                step=0.001,
                help="Вероятность события оплаты (pay) для варианта B.",
            )
            amount_mean_a = st.sidebar.slider(
                "Параметр mean чека (A)",
                1.0,
                6.0,
                2.8,
                step=0.05,
                help="Параметр mean логнормального распределения суммы покупки для A (чем больше, тем выше чек).",
            )
            amount_mean_b = st.sidebar.slider(
                "Параметр mean чека (B)",
                1.0,
                6.0,
                3.25,
                step=0.05,
                help="Параметр mean логнормального распределения суммы покупки для B (чем больше, тем выше чек).",
            )
            max_days = st.sidebar.slider("Горизонт (дни)", 3, 60, 14, step=1)
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
        st.sidebar.subheader("Загрузка CSV")
        uploaded = st.sidebar.file_uploader("events.csv", type=["csv"])
        if uploaded is not None:
            validation = load_and_validate_csv(uploaded)
            if not validation.ok:
                st.error(validation.error or "Ошибка валидации CSV")
                st.stop()
            events = validation.df

    return events, pay_event, source, scenario, n_users, seed


if APP_PASSWORD:
    st.sidebar.header("Доступ")
    st.session_state.setdefault("abl_authed", False)
    if not st.session_state["abl_authed"]:
        password = st.sidebar.text_input("Пароль", type="password")
        if password:
            if hmac.compare_digest(str(password), str(APP_PASSWORD)):
                st.session_state["abl_authed"] = True
            else:
                st.sidebar.error("Неверный пароль")
        st.stop()


df, pay_event, source, scenario, n_users, seed = _source_sidebar()

if df is None or len(df) == 0:
    st.info("Загрузите данные в сайдбаре, чтобы начать.")
    st.stop()

df_viz = _prepare_viz_frame(df)
if df_viz.empty:
    st.error("В данных нет валидных временных меток.")
    st.stop()

users = build_user_table(df, pay_event=pay_event)
if set(users["variant"].unique()) != {"A", "B"}:
    st.error("Данные должны содержать оба варианта: A и B.")
    st.stop()

kpis = compute_basic_kpis(users)
srm = check_srm(users)
conv_test = conversion_ztest_with_ci(users)
arpu_boot = arpu_bootstrap(users, n_boot=ARPU_BOOTSTRAP_SAMPLES)
verdict = generate_verdict(kpis=kpis, srm=srm, conv_test=conv_test, arpu_boot=arpu_boot)

funnel_by_variant = None
ret_df = None

tab_names = ["Дашборд", "Метрики", "Статистика"]
if ENABLE_SQL:
    tab_names.append("SQL")
tab_names.append("Загрузки")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

def _render_conclusion_section(
    *,
    srm_result,
    conv_result,
    arpu_result,
    kpis_for_window: pd.DataFrame,
):
    decision, explanation = _decision_text(srm_result, conv_result, arpu_result, kpis_for_window)
    st.subheader("Итог и рекомендация")

    decision_cols = st.columns([1.2, 2])
    with decision_cols[0]:
        st.metric("Решение", decision)
    with decision_cols[1]:
        st.info(explanation)

    st.dataframe(
        _build_conclusion_rows(kpis_for_window, conv_result, arpu_result),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Чек-лист перед раскаткой")
    checklist = [
        ("Распределение трафика (SRM)", "ОК" if srm_result.ok else "Проверить"),
        ("Доказательства по конверсии", "ОК" if conv_result.p_value < 0.05 else "Неубедительно"),
        (
            "Доказательства по ARPU",
            "ОК" if (arpu_result.ci95[0] > 0 or arpu_result.ci95[1] < 0) else "Неубедительно",
        ),
    ]
    st.dataframe(pd.DataFrame(checklist, columns=["проверка", "статус"]), width="stretch", hide_index=True)


with tabs["Дашборд"]:
    st.subheader("Дашборд эксперимента")

    mode = st.radio("Режим", ["Базовый", "По анкете"], horizontal=True)

    date_min = df_viz["date"].min().date()
    date_max = df_viz["date"].max().date()
    segment_candidates = sorted(
        column
        for column in df_viz.columns
        if column not in {"user_id", "variant", "ts", "event", "amount", "date"}
    )

    controls = st.columns([1.2, 1, 1, 1])
    with controls[0]:
        view_options = [
            "Обзор",
            "Лента событий",
            "Дневная конверсия",
            "Выручка и ARPU",
            "Воронка",
            "Удержание",
            "Качество данных",
            "Итог",
        ]
        view_index = 0
        if mode == "По анкете":
            with st.expander("Анкета (помогает выбрать нужные выводы)"):
                goal = st.selectbox(
                    "Какие выводы хотите сделать?",
                    [
                        "Принять решение о запуске",
                        "Оценить влияние на конверсию",
                        "Оценить влияние на выручку/ARPU",
                        "Понять воронку",
                        "Оценить удержание",
                        "Проверить качество данных",
                        "Найти сегменты",
                    ],
                    index=0,
                )
                st.session_state["dashboard_goal_ru"] = goal

            default_view = {
                "Принять решение о запуске": "Итог",
                "Оценить влияние на конверсию": "Дневная конверсия",
                "Оценить влияние на выручку/ARPU": "Выручка и ARPU",
                "Понять воронку": "Воронка",
                "Оценить удержание": "Удержание",
                "Проверить качество данных": "Качество данных",
                "Найти сегменты": "Обзор",
            }.get(goal, "Итог")
            view_index = view_options.index(default_view)

        dashboard_goal = st.selectbox(
            "Раздел",
            view_options,
            index=view_index,
        )
    with controls[1]:
        selected_event = st.selectbox(
            "Событие",
            sorted(df_viz["event"].astype(str).unique().tolist()),
            index=0,
        )
    with controls[2]:
        segment_col = st.selectbox("Сегмент", ["Нет"] + segment_candidates, index=0)
    with controls[3]:
        max_rows = st.number_input("Строк", min_value=20, max_value=500, value=100, step=20)

    date_controls = st.columns(2)
    with date_controls[0]:
        date_from = st.date_input("С", value=date_min, min_value=date_min, max_value=date_max)
    with date_controls[1]:
        date_to = st.date_input("По", value=date_max, min_value=date_min, max_value=date_max)

    if date_from > date_to:
        st.error("Дата «С» должна быть не позже даты «По».")
        st.stop()

    df_window = df_viz[
        (df_viz["date"] >= pd.Timestamp(date_from)) & (df_viz["date"] <= pd.Timestamp(date_to))
    ].copy()

    if df_window.empty:
        st.info("В выбранном окне дат событий нет.")
        st.stop()

    users_window = build_user_table(df_window, pay_event=pay_event)
    kpis_window = compute_basic_kpis(users_window)
    srm_window = check_srm(users_window)
    conv_test_window = conversion_ztest_with_ci(users_window)
    arpu_boot_window = arpu_bootstrap(users_window, n_boot=ARPU_BOOTSTRAP_SAMPLES)

    conv_a = _variant_metric(kpis_window, "A", "conversion")
    conv_b = _variant_metric(kpis_window, "B", "conversion")
    arpu_a = _variant_metric(kpis_window, "A", "arpu")
    arpu_b = _variant_metric(kpis_window, "B", "arpu")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Пользователи", f"{int(users_window['user_id'].nunique()):,}")
    summary_cols[1].metric("Конверсия B − A", _format_pct(conv_b - conv_a))
    summary_cols[2].metric("ARPU B − A", _format_money(arpu_b - arpu_a))
    summary_cols[3].metric("События", f"{len(df_window):,}")

    if dashboard_goal == "Обзор":
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
            title="Ключевые метрики по вариантам",
        )
        st.plotly_chart(fig, width="stretch")

    elif dashboard_goal == "Лента событий":
        daily = _daily_event_counts(df_window)
        daily = daily.loc[daily["event"].astype(str) == str(selected_event)]
        if daily.empty:
            st.info("По выбранному событию строк нет.")
        else:
            fig = px.line(
                daily,
                x="date",
                y="events",
                color="variant",
                markers=True,
                title=f"События {selected_event} по дням",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(daily.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Дневная конверсия":
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
            title="Доля плательщиков среди активных (по дням)",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(daily.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Выручка и ARPU":
        active = _daily_active_users(df_window)
        revenue = _daily_revenue(df_window, pay_event)
        daily = active.merge(revenue, on=["date", "variant"], how="left").fillna({"revenue": 0.0})
        daily["arpu_proxy"] = daily["revenue"] / daily["active_users"].where(
            daily["active_users"] > 0, pd.NA
        )
        revenue_chart, arpu_chart = st.columns(2)
        with revenue_chart:
            st.plotly_chart(
                px.line(daily, x="date", y="revenue", color="variant", markers=True, title="Выручка"),
                width="stretch",
            )
        with arpu_chart:
            st.plotly_chart(
                px.line(daily, x="date", y="arpu_proxy", color="variant", markers=True, title="ARPU (прокси)"),
                width="stretch",
            )
        st.dataframe(daily.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Воронка":
        all_events = sorted(df_window["event"].astype(str).unique().tolist())
        default_steps = [step for step in ["signup", pay_event] if step in all_events]
        steps = st.multiselect("Шаги", options=all_events, default=default_steps, key="dashboard_funnel")
        if len(steps) < 2:
            st.info("Выберите минимум два шага.")
        else:
            funnel = compute_funnel(df_window, steps=steps).by_variant
            st.dataframe(funnel, width="stretch", hide_index=True)
            fig = px.bar(
                funnel,
                x="step",
                y="users_reached",
                color="variant",
                barmode="group",
                title="Пользователи по шагам воронки",
            )
            st.plotly_chart(fig, width="stretch")

    elif dashboard_goal == "Удержание":
        retention_day = st.slider("Макс. день", 7, 60, 14, key="dashboard_retention_day")
        active_options = ["Любое событие"] + sorted(df_window["event"].astype(str).unique().tolist())
        active_event = st.selectbox("Активность по событию", active_options, index=0)
        active_event_value = None if active_event == "Любое событие" else active_event
        retention = compute_retention_curve(df_window, active_event=active_event_value, max_day=retention_day)
        fig = px.line(retention, x="day", y="retention", color="variant", markers=True, title="Удержание")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(retention.head(int(max_rows)), width="stretch", hide_index=True)

    elif dashboard_goal == "Качество данных":
        quality_cols = st.columns(4)
        quality_cols[0].metric("Строк", f"{len(df_window):,}")
        quality_cols[1].metric("Пользователей", f"{df_window['user_id'].nunique():,}")
        quality_cols[2].metric("Событий", f"{df_window['event'].nunique():,}")
        quality_cols[3].metric("Пустых amount", f"{int(df_window['amount'].isna().sum()):,}")
        top_events = df_window["event"].astype(str).value_counts().head(20).reset_index()
        top_events.columns = ["event", "rows"]
        st.dataframe(top_events, width="stretch", hide_index=True)
    elif dashboard_goal == "Итог":
        _render_conclusion_section(
            srm_result=srm_window,
            conv_result=conv_test_window,
            arpu_result=arpu_boot_window,
            kpis_for_window=kpis_window,
        )
    else:
        st.info("Выберите раздел для отображения.")

    if segment_col != "Нет":
        st.divider()
        st.subheader("Разрез по сегменту")
        user_segment = _stable_user_segment(df_window, segment_col)
        segmented_users = users_window.merge(user_segment, on="user_id", how="inner")
        if segmented_users.empty:
            st.info("Для выбранного столбца нет стабильных значений сегмента на уровне пользователя.")
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
                title="Конверсия по сегментам",
            )
            st.plotly_chart(fig, width="stretch")


with tabs["Метрики"]:
    st.subheader("Метрики на уровне пользователя")
    st.dataframe(kpis, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Воронка")
    all_events = sorted(df["event"].astype(str).unique().tolist())
    default_steps = [step for step in ["signup", pay_event] if step in all_events]
    steps = st.multiselect("Шаги", options=all_events, default=default_steps, key="metrics_funnel")
    if len(steps) >= 2:
        funnel_result = compute_funnel(df, steps=steps)
        funnel_by_variant = funnel_result.by_variant
        st.dataframe(funnel_by_variant, width="stretch", hide_index=True)
    else:
        st.info("Выберите минимум два шага.")

    st.divider()
    st.subheader("Удержание")
    max_day = st.slider("Макс. день", 7, 60, 14, key="metrics_retention_day")
    try:
        ret_df = compute_retention_curve(df, active_event=None, max_day=max_day)
        st.dataframe(ret_df, width="stretch", hide_index=True)
    except Exception as error:
        ret_df = None
        st.error(f"Ошибка расчёта удержания: {error}")


with tabs["Статистика"]:
    st.subheader("Проверка SRM")
    srm_cols = st.columns(3)
    srm_cols[0].metric("Статус", "ОК" if srm.ok else "Проверить")
    srm_cols[1].metric("p-value", f"{srm.p_value:.4g}")
    srm_cols[2].metric("Сплит", f"A={srm.observed.get('A', 0):,} / B={srm.observed.get('B', 0):,}")

    st.divider()
    st.subheader("z-test по конверсии")
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
    st.subheader("Бутстрэп ARPU")
    st.json(
        {
            "diff_mean": float(arpu_boot.diff_mean),
            "ci95": [float(arpu_boot.ci95[0]), float(arpu_boot.ci95[1])],
            "p_value_two_sided": float(arpu_boot.p_value_two_sided),
            "n_boot": int(arpu_boot.n_boot),
        }
    )


if ENABLE_SQL:
    with tabs["SQL"]:
        st.subheader("SQL")
        st.caption("Таблица: events. Разрешены только запросы SELECT/WITH (read-only).")
        presets = built_in_queries(pay_event=pay_event)
        preset_name = st.selectbox("Пресет", list(presets.keys()), index=0)
        sql_text = st.text_area("SQL", value=presets[preset_name], height=180)

        if st.button("Выполнить SQL", type="primary"):
            try:
                output = run_sql(df, sql_text)
                st.dataframe(output, width="stretch", hide_index=True)
            except Exception as error:
                st.error(f"Ошибка выполнения SQL: {error}")

        st.divider()
        st.subheader("Превью сырых событий")
        st.dataframe(df.head(200), width="stretch", hide_index=True)


with tabs["Загрузки"]:
    st.subheader("Экспорт результатов")
    st.caption("Скачайте сырые данные, расчётные таблицы, метаданные и markdown-отчёт.")

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
