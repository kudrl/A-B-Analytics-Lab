import math

import pandas as pd
import pytest

from src.analysis import (
    arpu_bootstrap,
    build_user_table,
    check_srm,
    compute_ordered_funnel,
    conversion_ztest_with_ci,
)


def _events(rows):
    return pd.DataFrame(rows, columns=["user_id", "variant", "ts", "event", "amount"])


def test_srm_balanced_ok():
    users = pd.DataFrame(
        {
            "user_id": range(100),
            "variant": ["A"] * 50 + ["B"] * 50,
        }
    )

    result = check_srm(users)

    assert result.ok
    assert result.p_value == pytest.approx(1.0)
    assert result.observed == {"A": 50, "B": 50}


def test_srm_broken_fails():
    users = pd.DataFrame(
        {
            "user_id": range(100),
            "variant": ["A"] * 99 + ["B"],
        }
    )

    result = check_srm(users)

    assert not result.ok
    assert result.p_value < 0.01


def test_user_in_two_variants_raises():
    df = _events(
        [
            [1, "A", "2024-01-01", "signup", 0.0],
            [1, "B", "2024-01-02", "pay", 10.0],
        ]
    )

    with pytest.raises(ValueError, match="users appear in more than one variant"):
        build_user_table(df)


def test_ztest_empty_group_safe():
    users = pd.DataFrame(
        {
            "user_id": [1, 2],
            "variant": ["A", "A"],
            "is_payer": [True, False],
            "revenue": [10.0, 0.0],
        }
    )

    result = conversion_ztest_with_ci(users)

    assert result.p_value == 1.0
    assert result.conv_a == 0.5
    assert result.conv_b == 0.0
    assert math.isnan(result.ci95_abs[0])
    assert math.isnan(result.ci95_abs[1])


def test_bootstrap_returns_ci():
    users = pd.DataFrame(
        {
            "user_id": range(6),
            "variant": ["A", "A", "A", "B", "B", "B"],
            "revenue": [0.0, 1.0, 2.0, 2.0, 4.0, 6.0],
        }
    )

    result = arpu_bootstrap(users, n_boot=100, seed=1)

    assert result.metric == "ARPU"
    assert result.n_boot == 100
    assert result.ci95[0] <= result.diff_mean <= result.ci95[1]
    assert 0.0 <= result.p_value_two_sided <= 1.0


def test_ordered_funnel_rejects_wrong_order():
    df = _events(
        [
            [1, "A", "2024-01-02", "pay", 10.0],
            [1, "A", "2024-01-03", "signup", 0.0],
            [2, "A", "2024-01-01", "signup", 0.0],
            [2, "A", "2024-01-02", "pay", 10.0],
        ]
    )

    result = compute_ordered_funnel(df, ["signup", "pay"]).by_variant
    a_steps = result[result["variant"] == "A"].set_index("step")

    assert a_steps.loc["signup", "users_reached"] == 2
    assert a_steps.loc["pay", "users_reached"] == 1
