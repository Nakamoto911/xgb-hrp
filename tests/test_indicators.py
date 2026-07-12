"""Price-derived regime indicators: sma_bear_signal, adx, ma_adx_bear_score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.indicators import adx, ma_adx_bear_score, sma_bear_signal


def _prices(values: list[float], symbol: str = "A") -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=len(values), freq="B")
    return pd.DataFrame({symbol: values}, index=idx, dtype=float)


# -----------------------------------------------------------------------------
# sma_bear_signal
# -----------------------------------------------------------------------------
def test_sma_bear_signal_rising_series_is_bull_after_warmup():
    prices = _prices([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    sig = sma_bear_signal(prices, window=5)
    # First 4 rows are SMA warmup (min_periods=5) -> NaN; rest -> bull (0.0).
    assert sig["A"].iloc[:4].isna().all()
    assert sig["A"].iloc[4:].tolist() == [0.0, 0.0, 0.0]


def test_sma_bear_signal_falling_series_is_bear_after_warmup():
    prices = _prices([106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
    sig = sma_bear_signal(prices, window=5)
    assert sig["A"].iloc[:4].isna().all()
    assert sig["A"].iloc[4:].tolist() == [1.0, 1.0, 1.0]


def test_sma_bear_signal_warmup_rows_are_nan():
    prices = _prices([100.0, 99.0, 98.0, 97.0])
    sig = sma_bear_signal(prices, window=5)
    assert sig["A"].isna().all()


# -----------------------------------------------------------------------------
# adx
# -----------------------------------------------------------------------------
def test_adx_trend_scores_higher_than_chop():
    n = 30
    trend = _prices([100.0 + i for i in range(n)])
    chop_vals = [100.0]
    for i in range(n - 1):
        chop_vals.append(chop_vals[-1] + (1.0 if i % 2 == 0 else -1.0))
    chop = _prices(chop_vals)

    adx_trend = adx(trend, window=5)["A"].dropna()
    adx_chop = adx(chop, window=5)["A"].dropna()
    assert adx_trend.iloc[-1] > adx_chop.iloc[-1]


def test_adx_values_within_0_100():
    n = 30
    prices = _prices([100.0 + np.sin(i / 3.0) * 5 for i in range(n)])
    vals = adx(prices, window=5)["A"].dropna()
    assert (vals >= 0).all()
    assert (vals <= 100).all()


def test_adx_constant_series_no_inf_no_exception():
    prices = _prices([100.0] * 20)
    vals = adx(prices, window=5)["A"]
    assert not np.isinf(vals).any()
    non_nan = vals.dropna()
    assert (non_nan == 0.0).all()


# -----------------------------------------------------------------------------
# ma_adx_bear_score
# -----------------------------------------------------------------------------
def test_ma_adx_bear_score_falling_trend_above_half():
    prices = _prices([120.0 - i for i in range(20)])
    score = ma_adx_bear_score(prices, ma_window=5, adx_window=5)["A"].dropna()
    assert (score > 0.5).all()
    assert score.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_ma_adx_bear_score_rising_trend_below_half():
    prices = _prices([100.0 + i for i in range(20)])
    score = ma_adx_bear_score(prices, ma_window=5, adx_window=5)["A"].dropna()
    assert (score < 0.5).all()
    assert score.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_ma_adx_bear_score_within_0_1():
    n = 30
    prices = _prices([100.0 + np.sin(i / 3.0) * 5 for i in range(n)])
    score = ma_adx_bear_score(prices, ma_window=5, adx_window=5)["A"].dropna()
    assert (score >= 0).all()
    assert (score <= 1).all()
