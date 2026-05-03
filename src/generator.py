from __future__ import annotations

import numpy as np
import pandas as pd


def _lognorm_amount(rng: np.random.Generator, mean=3.0, sigma=1.0) -> float:
    return float(rng.lognormal(mean=mean, sigma=sigma))


"""
б конвертит лучше а
б конвертит хуже, но логнорм кривая
парадокс симпсона 
"""

def _date(day: int) -> str:
    return f"2023-01-{day:02d}T12:00:00"


def generate_conversion_lift(
    n_users: int = 20000,
    base_conv: float = 0.10,
    lift_rel: float = 0.15,
    base_open: float = 0.75,
    max_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=n_users)

    rows = []
    for uid, v in enumerate(variants):
        rows.append([uid, v, _date(1), "signup", 0.0])

        # activity days (open_app)
        n_open_days = rng.integers(0, max(1, max_days // 2) + 1)
        open_days = rng.choice(np.arange(1, max_days + 1), size=n_open_days, replace=False) if n_open_days > 0 else []
        for d in open_days:
            if rng.random() < base_open:
                rows.append([uid, v, _date(int(d)), "open_app", 0.0])

        p = base_conv * (1.0 + lift_rel) if v == "B" else base_conv
        if rng.random() < p:
            pay_day = int(rng.integers(2, max_days + 1))
            rows.append([uid, v, _date(pay_day), "pay", _lognorm_amount(rng)])

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount"])


def generate_arpu_tradeoff(
    n_users: int = 30000,
    conv_a: float = 0.12,
    conv_b: float = 0.10,
    amount_mean_a: float = 2.8,
    amount_mean_b: float = 3.25,
    max_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=n_users)

    rows = []
    for uid, v in enumerate(variants):
        rows.append([uid, v, _date(1), "signup", 0.0])

        p = conv_b if v == "B" else conv_a
        if rng.random() < p:
            mean = amount_mean_b if v == "B" else amount_mean_a
            pay_day = int(rng.integers(2, max_days + 1))
            rows.append([uid, v, _date(pay_day), "pay", _lognorm_amount(rng, mean=mean, sigma=1.0)])

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount"])


def generate_simpson_paradox(
    n_users: int = 40000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=n_users)

    countries = []
    for v in variants:
        if v == "A":
            countries.append(rng.choice(["RU", "DE"], p=[0.7, 0.3]))
        else:
            countries.append(rng.choice(["RU", "DE"], p=[0.3, 0.7]))

    base = {"RU": 0.14, "DE": 0.06}
    lift = 0.20

    rows = []
    for uid, (v, c) in enumerate(zip(variants, countries)):
        rows.append([uid, v, _date(1), "signup", 0.0, c])

        p = base[c] * (1.0 + lift) if v == "B" else base[c]
        if rng.random() < p:
            rows.append([uid, v, _date(2), "pay", _lognorm_amount(rng), c])

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount", "country"])


def generate_funnel_drop(
    *,
    n_users: int = 30000,
    steps: list[str] | None = None,
    step_conv_a: list[float] | None = None,
    step_lift_rel_b: list[float] | None = None,
    drop_step: str | None = None,
    drop_multiplier: float = 1.0,
    drop_variant: str = "B",
    max_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует многошаговую воронку с возможностью "уронить" конверсию
    на одном из шагов для выбранного варианта.

    - steps: упорядоченный список событий (например: signup, view, pay)
    - step_conv_a: конверсия перехода на шаг i (i>=1) для A, относительно предыдущего шага
      (для шага 0 значение игнорируется; шаг 0 считается достигнутым всеми пользователями)
    - step_lift_rel_b: относительный lift B к A по каждому переходу (i>=1): 0.2 = +20%
    - drop_step + drop_multiplier: дополнительный множитель к вероятности перехода на drop_step
      (например 0.7 = -30%) для drop_variant (A или B)
    """
    if steps is None:
        steps = ["signup", "view_product", "add_to_cart", "checkout", "pay"]
    steps = [str(s).strip() for s in steps if str(s).strip()]
    if len(steps) < 2:
        raise ValueError("steps must contain at least 2 events")

    n_steps = len(steps)
    if step_conv_a is None:
        step_conv_a = [1.0] + [0.6] * (n_steps - 2) + [0.2]
    if step_lift_rel_b is None:
        step_lift_rel_b = [0.0] * n_steps

    if len(step_conv_a) != n_steps:
        raise ValueError("step_conv_a must have same length as steps")
    if len(step_lift_rel_b) != n_steps:
        raise ValueError("step_lift_rel_b must have same length as steps")

    if not (0.0 <= float(drop_multiplier) <= 1.0):
        raise ValueError("drop_multiplier must be between 0 and 1")

    drop_variant = str(drop_variant).upper().strip() or "B"
    if drop_variant not in {"A", "B"}:
        raise ValueError("drop_variant must be 'A' or 'B'")

    rng = np.random.default_rng(seed)
    variants = rng.choice(["A", "B"], size=int(n_users))

    base_day = 1
    step_days = np.linspace(base_day, max(base_day + 1, int(max_days)), num=n_steps).astype(int).tolist()

    rows: list[list[object]] = []
    for uid, v in enumerate(variants):
        reached_prev = True
        for idx, step in enumerate(steps):
            if idx == 0:
                reached = True
            else:
                p_a = float(step_conv_a[idx])
                lift = float(step_lift_rel_b[idx]) if v == "B" else 0.0
                p = p_a * (1.0 + lift)
                if drop_step is not None and str(step) == str(drop_step).strip() and v == drop_variant:
                    p *= float(drop_multiplier)
                p = float(np.clip(p, 0.0, 1.0))
                reached = reached_prev and (rng.random() < p)

            if not reached:
                break

            amount = 0.0
            if idx == (n_steps - 1) and str(step) in {"pay", "purchase"}:
                amount = _lognorm_amount(rng)
            rows.append([uid, v, _date(int(step_days[idx])), str(step), float(amount)])
            reached_prev = reached

    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount"])
