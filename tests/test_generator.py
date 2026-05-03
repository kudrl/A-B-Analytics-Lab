import pandas as pd

from src.analysis import compute_funnel
from src.generator import generate_funnel_drop


def test_generate_funnel_drop_structure_and_monotonicity():
    steps = ["signup", "view", "pay"]
    df = generate_funnel_drop(
        n_users=20000,
        steps=steps,
        step_conv_a=[1.0, 1.0, 1.0],
        step_lift_rel_b=[0.0, 0.0, 0.0],
        seed=1,
        max_days=7,
    )

    assert isinstance(df, pd.DataFrame)
    assert set(["user_id", "variant", "ts", "event", "amount"]).issubset(df.columns)
    assert set(df["variant"].unique()) <= {"A", "B"}

    funnel = compute_funnel(df, steps=steps).by_variant
    for v in ["A", "B"]:
        reached = funnel.loc[funnel["variant"] == v, "users_reached"].tolist()
        assert reached == sorted(reached, reverse=True)


def test_generate_funnel_drop_applies_drop_to_selected_variant_and_step():
    steps = ["signup", "view", "pay"]
    df = generate_funnel_drop(
        n_users=20000,
        steps=steps,
        step_conv_a=[1.0, 1.0, 1.0],
        step_lift_rel_b=[0.0, 0.0, 0.0],
        drop_step="view",
        drop_multiplier=0.5,
        drop_variant="B",
        seed=2,
        max_days=7,
    )

    funnel = compute_funnel(df, steps=steps).by_variant
    a_view = int(funnel[(funnel["variant"] == "A") & (funnel["step"] == "view")]["users_reached"].iloc[0])
    b_view = int(funnel[(funnel["variant"] == "B") & (funnel["step"] == "view")]["users_reached"].iloc[0])

    # У A все доходят до view, у B примерно половина (из-за drop_multiplier=0.5 на шаге view).
    assert a_view > b_view
    assert b_view / a_view < 0.7
