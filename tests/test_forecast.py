"""Forecast panel projection + 7 selection-rule helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.forecast import (
    RULES,
    _project_to_forecast_panel,
    apply_rule,
    rule_ewma_smoothed,
    rule_hybrid,
    rule_last_day_regime,
    rule_ma200,
    rule_prob_threshold,
    rule_regime_and_prob,
    rule_trend,
)


def _make_signal(p_bear: list[float], state: list[int]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=len(p_bear), freq="B")
    sig = pd.DataFrame(
        {
            "Raw_Prob": p_bear,
            "State_Prob": p_bear,
            "Forecast_State": state,
        },
        index=idx,
    )
    sig.attrs["asset_name"] = "LargeCap"
    return sig


def test_projection_translates_to_bull_relative():
    sig = _make_signal([0.2, 0.7], [0, 1])
    panel = _project_to_forecast_panel({"IVV": sig})
    assert list(panel.columns) == [
        "symbol", "asset_name", "date", "p_bear", "p_bull", "p_bull_smoothed", "regime_forecast"
    ]
    # p_bull = 1 - p_bear
    assert panel["p_bull"].tolist() == pytest.approx([0.8, 0.3])
    assert panel["p_bull_smoothed"].tolist() == pytest.approx([0.8, 0.3])
    assert panel["regime_forecast"].tolist() == ["Bull", "Bear"]


def test_all_7_rules_registered():
    assert set(RULES) == {
        "prob_threshold", "regime_and_prob", "ewma_smoothed", "trend", "last_day_regime",
        "ma200", "hybrid",
    }


def _per_symbol(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.set_index("date").drop(columns=["symbol", "asset_name"])


def test_prob_threshold_rule_strict_geq():
    sig = _make_signal([0.5, 0.39, 0.41], [0, 0, 0])  # p_bull = 0.5, 0.61, 0.59
    panel = _project_to_forecast_panel({"IVV": sig})
    sel = rule_prob_threshold(_per_symbol(panel), theta=0.60)
    assert sel.tolist() == [False, True, False]


def test_regime_and_prob_requires_both():
    sig = _make_signal([0.2, 0.2, 0.8], [0, 1, 0])
    panel = _project_to_forecast_panel({"IVV": sig})
    sel = rule_regime_and_prob(_per_symbol(panel), theta=0.55)
    # p_bull = 0.8 / 0.8 / 0.2 ; regime = Bull / Bear / Bull
    assert sel.tolist() == [True, False, False]


def test_ewma_smoothed_uses_smoothed_prob():
    sig = _make_signal([0.6, 0.6], [1, 1])  # p_bull_smoothed = 0.4
    panel = _project_to_forecast_panel({"IVV": sig})
    sel = rule_ewma_smoothed(_per_symbol(panel), theta=0.5)
    assert sel.tolist() == [False, False]


def test_trend_detects_rising_p_bull():
    # p_bear monotonically decreasing → p_bull monotonically rising
    sig = _make_signal([0.9, 0.7, 0.5, 0.3, 0.1], [1, 1, 0, 0, 0])
    panel = _project_to_forecast_panel({"IVV": sig})
    sel = rule_trend(_per_symbol(panel), window=5)
    # First 4 are NaN (rolling window), last has positive slope ⇒ True
    assert sel.tolist()[:4] == [False, False, False, False]
    assert sel.tolist()[4] is True or sel.iloc[4] == True  # noqa: E712


def test_trend_detects_falling_p_bull():
    sig = _make_signal([0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 0, 1, 1])
    panel = _project_to_forecast_panel({"IVV": sig})
    sel = rule_trend(_per_symbol(panel), window=5)
    assert bool(sel.iloc[4]) is False


def test_last_day_regime_just_reads_forecast():
    sig = _make_signal([0.4, 0.6], [0, 1])
    panel = _project_to_forecast_panel({"IVV": sig})
    sel = rule_last_day_regime(_per_symbol(panel))
    assert sel.tolist() == [True, False]


def _price_series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2023-01-02", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_ma200_all_bull_after_warmup_on_uptrend():
    sig = _make_signal([0.5] * 6, [0] * 6)  # forecast content is irrelevant to rule_ma200
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    sel = rule_ma200(df, price=price, ma_window=3)
    assert sel.tolist() == [True, True, True, True, True, True]


def test_ma200_downtrend_below_sma_is_bear():
    sig = _make_signal([0.5] * 6, [0] * 6)
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    sel = rule_ma200(df, price=price, ma_window=3)
    # First 2 days are SMA warmup (default bull); the rest trail below the SMA.
    assert sel.tolist() == [True, True, False, False, False, False]


def test_ma200_warmup_defaults_to_bull():
    sig = _make_signal([0.5] * 4, [0] * 4)
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series([100.0, 90.0, 110.0, 80.0])  # choppy, but window never fills
    sel = rule_ma200(df, price=price, ma_window=10)
    assert sel.tolist() == [True, True, True, True]


def test_ma200_nan_price_not_selected():
    sig = _make_signal([0.5] * 6, [0] * 6)
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series([100.0, 101.0, float("nan"), 103.0, 104.0, 105.0])
    sel = rule_ma200(df, price=price, ma_window=3)
    assert bool(sel.iloc[2]) is False


def test_rule_ma200_interior_gap_defaults_bull():
    """A NaN gap after warm-up: any rolling(window) that spans the gap has
    fewer than ``ma_window`` valid observations, so pandas yields NaN for it —
    those dates fall under the ``sma.isna() & price.notna()`` warm-up clause
    and default bull, while the NaN date itself is not selected. Documents
    why the executor feeds the price-aware rules a forward-filled panel."""
    sig = _make_signal([0.5] * 8, [0] * 8)
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series(
        [100.0, 101.0, 102.0, float("nan"), 104.0, 105.0, 106.0, 107.0]
    )
    sel = rule_ma200(df, price=price, ma_window=3)
    # 0-1: SMA warmup. 2: SMA primed, price above it. 3: the NaN date itself.
    # 4-5: rolling windows still touching the gap -> SMA NaN -> default bull.
    # 6-7: gap has rolled out of the window -> SMA primed again.
    assert sel.tolist() == [True, True, True, False, True, True, True, True]


def test_ma200_uses_full_price_history_not_truncated_to_df_index():
    """The SMA must be computed on the full ``price`` series before reindexing
    to ``df.index`` — truncating first would leave the SMA in perpetual
    warmup (default bull) for a panel window shorter than ``ma_window``."""
    sig = _make_signal([0.5] * 3, [0] * 3)
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)  # 3-day index
    full_price = _price_series([100.0] * 7 + [90.0, 80.0, 70.0])  # 10-day history
    df.index = full_price.index[-3:]  # panel only covers the last 3 days
    sel = rule_ma200(df, price=full_price, ma_window=5)
    # Full-history SMA is already primed by day 8 — price has since dropped below it.
    assert sel.tolist() == [False, False, False]
    # Confirms the two approaches would actually diverge: truncating first
    # leaves too few points for the rolling window to ever fill.
    truncated = full_price.loc[df.index]
    assert truncated.rolling(5).mean().isna().all()


def test_hybrid_deselects_on_p_bear_spike_even_when_price_above_sma():
    sig = _make_signal(
        p_bear=[0.10, 0.10, 0.10, 0.90, 0.90, 0.10],
        state=[0, 0, 0, 1, 1, 0],
    )
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])  # steady uptrend
    sel = rule_hybrid(df, price=price, ma_window=2, hybrid_bear_threshold=0.80)
    # Price stays above its SMA throughout; only the smoothed p_bear spike
    # (days 3-4, > 0.80) should deselect.
    assert sel.tolist() == [True, True, True, False, False, True]


def test_hybrid_follows_ma200_when_p_bear_below_threshold():
    sig = _make_signal(p_bear=[0.10] * 6, state=[0] * 6)  # p_bear_smoothed = 0.10 always
    panel = _project_to_forecast_panel({"IVV": sig})
    df = _per_symbol(panel)
    price = _price_series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])  # crosses below its SMA
    sel = rule_hybrid(df, price=price, ma_window=3, hybrid_bear_threshold=0.80)
    expected = rule_ma200(df, price=price, ma_window=3)
    assert sel.tolist() == expected.tolist() == [True, True, False, False, False, False]


def test_apply_rule_groups_by_symbol():
    ivv = _make_signal([0.3, 0.7], [0, 1])
    agg = _make_signal([0.7, 0.3], [1, 0])
    panel = _project_to_forecast_panel({"IVV": ivv, "AGG": agg})
    flags = apply_rule(panel, "prob_threshold", theta=0.55, trend_window=5)
    assert set(flags.columns) == {"symbol", "date", "selected"}
    # IVV p_bull = 0.7/0.3 → True/False ; AGG p_bull = 0.3/0.7 → False/True
    by_symbol = {s: g.sort_values("date")["selected"].tolist() for s, g in flags.groupby("symbol")}
    assert by_symbol["IVV"] == [True, False]
    assert by_symbol["AGG"] == [False, True]


def test_apply_rule_rejects_unknown_rule():
    panel = _project_to_forecast_panel({"IVV": _make_signal([0.5], [0])})
    with pytest.raises(ValueError, match="Unknown rule"):
        apply_rule(panel, "garbage", theta=0.5, trend_window=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("rule", ["ma200", "hybrid"])
def test_apply_rule_price_rules_require_prices(rule):
    panel = _project_to_forecast_panel({"IVV": _make_signal([0.2, 0.2], [0, 0])})
    with pytest.raises(ValueError, match="requires prices"):
        apply_rule(panel, rule, theta=0.5, trend_window=5, prices=None)


def test_apply_rule_missing_symbol_in_prices_raises():
    ivv = _make_signal([0.2, 0.2], [0, 0])
    agg = _make_signal([0.2, 0.2], [0, 0])
    panel = _project_to_forecast_panel({"IVV": ivv, "AGG": agg})
    prices = pd.DataFrame(
        {"IVV": [100.0, 101.0]}, index=pd.date_range("2023-01-02", periods=2, freq="B")
    )
    with pytest.raises(ValueError, match="AGG"):
        apply_rule(panel, "ma200", theta=0.5, trend_window=5, prices=prices, ma_window=2)


def test_apply_rule_ma200_happy_path_shape():
    ivv = _make_signal([0.2, 0.2, 0.2], [0, 0, 0])
    panel = _project_to_forecast_panel({"IVV": ivv})
    idx = pd.date_range("2023-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"IVV": [100.0, 101.0, 102.0]}, index=idx)
    flags = apply_rule(panel, "ma200", theta=0.5, trend_window=5, prices=prices, ma_window=2)
    assert set(flags.columns) == {"symbol", "date", "selected"}
    assert len(flags) == 3
    assert flags["selected"].dtype == bool


def test_apply_rule_hybrid_happy_path_shape():
    ivv = _make_signal([0.2, 0.2, 0.2], [0, 0, 0])
    panel = _project_to_forecast_panel({"IVV": ivv})
    idx = pd.date_range("2023-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"IVV": [100.0, 101.0, 102.0]}, index=idx)
    flags = apply_rule(
        panel, "hybrid", theta=0.5, trend_window=5, prices=prices,
        ma_window=2, hybrid_bear_threshold=0.80,
    )
    assert set(flags.columns) == {"symbol", "date", "selected"}
    assert len(flags) == 3
    assert flags["selected"].dtype == bool
