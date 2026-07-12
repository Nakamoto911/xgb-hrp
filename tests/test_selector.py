"""Asset selector — edge cases + diff bookkeeping."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.selector import select


def _panel(
    rows: list[tuple[str, str, float, float, str]],
) -> pd.DataFrame:
    """Build a tiny long-form forecast panel from (symbol, date, p_bull, p_bull_smoothed, regime) tuples."""
    return pd.DataFrame(
        rows, columns=["symbol", "date", "p_bull", "p_bull_smoothed", "regime_forecast"],
    ).assign(
        date=lambda d: pd.to_datetime(d["date"]),
        asset_name=lambda d: d["symbol"],
        p_bear=lambda d: 1.0 - d["p_bull"],
    )


def test_empty_selection_when_no_symbol_passes():
    panel = _panel([
        ("IVV", "2024-01-02", 0.50, 0.50, "Bear"),
        ("AGG", "2024-01-02", 0.40, 0.40, "Bear"),
    ])
    cfg = PipelineConfig(forecast_method="prob_threshold", bull_prob_threshold=0.99)
    out = select(panel, cfg)
    assert out.by_date[pd.Timestamp("2024-01-02")] == []


def test_singleton_selection_returned_as_is():
    panel = _panel([
        ("IVV", "2024-01-02", 0.95, 0.95, "Bull"),
        ("AGG", "2024-01-02", 0.40, 0.40, "Bear"),
    ])
    cfg = PipelineConfig(forecast_method="prob_threshold", bull_prob_threshold=0.60)
    out = select(panel, cfg)
    assert out.by_date[pd.Timestamp("2024-01-02")] == ["IVV"]


def test_diffs_track_added_removed_kept():
    panel = _panel([
        ("IVV", "2024-01-02", 0.95, 0.95, "Bull"),
        ("AGG", "2024-01-02", 0.95, 0.95, "Bull"),
        ("IVV", "2024-01-03", 0.30, 0.30, "Bear"),  # dropped
        ("AGG", "2024-01-03", 0.95, 0.95, "Bull"),  # kept
        ("GLD", "2024-01-03", 0.95, 0.95, "Bull"),  # added
    ])
    cfg = PipelineConfig(forecast_method="prob_threshold", bull_prob_threshold=0.60)
    out = select(panel, cfg)
    diffs = out.diffs.set_index("date")
    assert diffs.loc[pd.Timestamp("2024-01-02"), "added"] == ["AGG", "IVV"]
    assert diffs.loc[pd.Timestamp("2024-01-02"), "removed"] == []
    assert diffs.loc[pd.Timestamp("2024-01-03"), "added"] == ["GLD"]
    assert diffs.loc[pd.Timestamp("2024-01-03"), "removed"] == ["IVV"]
    assert diffs.loc[pd.Timestamp("2024-01-03"), "kept"] == ["AGG"]


def test_restricts_to_on_dates():
    panel = _panel([
        ("IVV", "2024-01-02", 0.95, 0.95, "Bull"),
        ("IVV", "2024-01-03", 0.95, 0.95, "Bull"),
        ("IVV", "2024-01-04", 0.95, 0.95, "Bull"),
    ])
    cfg = PipelineConfig(forecast_method="prob_threshold", bull_prob_threshold=0.60)
    out = select(panel, cfg, on_dates=pd.DatetimeIndex(["2024-01-03"]))
    assert list(out.by_date) == [pd.Timestamp("2024-01-03")]


def test_select_passes_prices_through_to_ma200_rule():
    panel = _panel([
        ("IVV", "2024-01-02", 0.50, 0.50, "Bull"),
        ("IVV", "2024-01-03", 0.50, 0.50, "Bull"),
        ("IVV", "2024-01-04", 0.50, 0.50, "Bull"),
    ])
    prices = pd.DataFrame(
        {"IVV": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    cfg = PipelineConfig(forecast_method="ma200", ma_window=2)
    out = select(panel, cfg, prices=prices)
    # Steady uptrend on a 2-day SMA → selected every day (warmup + above SMA).
    assert out.by_date[pd.Timestamp("2024-01-04")] == ["IVV"]


def test_select_ma200_without_prices_raises():
    panel = _panel([
        ("IVV", "2024-01-02", 0.50, 0.50, "Bull"),
    ])
    cfg = PipelineConfig(forecast_method="ma200")
    with pytest.raises(ValueError, match="requires prices"):
        select(panel, cfg)
