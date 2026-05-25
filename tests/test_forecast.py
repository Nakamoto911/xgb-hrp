"""Forecast panel projection + 5 selection-rule helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.forecast import (
    RULES,
    _project_to_forecast_panel,
    apply_rule,
    rule_ewma_smoothed,
    rule_last_day_regime,
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


def test_all_5_rules_registered():
    assert set(RULES) == {
        "prob_threshold", "regime_and_prob", "ewma_smoothed", "trend", "last_day_regime"
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
