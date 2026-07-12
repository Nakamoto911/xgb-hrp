"""Price-derived regime indicators — alternative signal sources for the
Asset Regime heatmap (:func:`pipeline.charts.asset_regime_chart`).

These are lightweight, dependency-free alternatives to the XGB model's
P(bear) forecast: pure functions of the adjusted-close price panel loaded
by :func:`pipeline.data.load_prices`, with no fit / walk-forward step. They
let the dashboard cross-check the model's regime calls against a classic
trend-following baseline (MA200) and a trend-strength-weighted variant
(MA200 + ADX).

Caveat — ADX here is a CLOSE-ONLY approximation. The pipeline's price panel
only ever carries Adjusted Close (see ``pipeline.data._fetch_one_ticker`` —
no High/Low columns are fetched), so the textbook Wilder ADX (which needs
the high/low range to build directional movement and true range) cannot be
reproduced exactly. Day-over-day close deltas stand in for both legs here.
Treat the output as a rough trend-strength proxy, not a certified ADX value.

All functions take/return wide date x symbol DataFrames (DatetimeIndex,
one column per symbol), are NaN-safe (NaN in -> NaN out, no inf), and are
fully vectorized (no Python loops over rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["sma_bear_signal", "adx", "ma_adx_bear_score"]


def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """Elementwise ``num / den`` with exact-zero denominators -> 0.0.

    Prevents +-inf when a denominator (e.g. Wilder-smoothed true range on a
    flat price series) is exactly zero. NaN denominators/numerators (still
    in warmup) propagate as NaN, since ``NaN == 0`` is False so ``.mask``
    leaves the (already-NaN) division result untouched.
    """
    return (num / den).mask(den == 0, 0.0)


def sma_bear_signal(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Binary bear signal: 1.0 when price < its trailing SMA, else 0.0.

    ``window`` is a FIXED canonical SMA length (default 200) — the same
    convention as the cockpit's ``ma200`` reference baseline
    (``pipeline.cockpit._baseline_labels``: ``px < px.rolling(200).mean()``),
    not the configurable production ``ma_window`` used by
    ``pipeline.forecast.rule_ma200``.

    NaN wherever the SMA is undefined (the ``window``-row warmup) or the
    price itself is missing that day — ``rolling(window, min_periods=window)``
    already yields NaN for any window that spans a missing price, including
    the row of the missing price itself.
    """
    sma = prices.rolling(window, min_periods=window).mean()
    bear = pd.DataFrame(
        np.where(prices < sma, 1.0, 0.0), index=prices.index, columns=prices.columns
    )
    return bear.where(sma.notna())


def adx(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Close-only Average Directional Index (ADX) approximation, in [0, 100].

    CAVEAT: true ADX needs High/Low to build directional movement (+DM/-DM)
    and true range (TR); this pipeline only ever carries Adjusted Close
    (see ``pipeline.data._fetch_one_ticker``). Here the day-over-day close
    delta substitutes for both legs — a rough trend-strength proxy, not a
    textbook ADX.

    Wilder smoothing is ``ewm(alpha=1/window, adjust=False,
    min_periods=window)``. Divisions are guarded (:func:`_safe_div`) so a
    flat/constant price series (zero true range throughout) yields 0.0
    DI/DX rather than inf or a raised exception. NaN during warmup.
    """
    delta = prices.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    plus_dm = up.where(up > down, 0.0)
    minus_dm = down.where(down > up, 0.0)
    tr = delta.abs()  # close-only true range (no High/Low available)

    def _wilder(x: pd.DataFrame) -> pd.DataFrame:
        return x.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    tr_w = _wilder(tr)
    plus_di = 100.0 * _safe_div(_wilder(plus_dm), tr_w)
    minus_di = 100.0 * _safe_div(_wilder(minus_dm), tr_w)

    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return _wilder(dx)


def ma_adx_bear_score(
    prices: pd.DataFrame,
    ma_window: int = 200,
    adx_window: int = 14,
    adx_scale: float = 50.0,
) -> pd.DataFrame:
    """MA200-direction x ADX-strength bear score in [0, 1].

    ``score = 0.5 + 0.5 * direction * strength`` where ``direction`` is +1
    when price is below its SMA (bearish) and -1 when at/above it
    (bullish), and ``strength = clip(adx / adx_scale, 0, 1)``. A strong
    downtrend -> ~1.0 (red on the heatmap), a strong uptrend -> ~0.0
    (green), and weak/no trend collapses toward 0.5 (gold) regardless of
    direction. NaN wherever either the SMA or the ADX is NaN (warmup).
    """
    sma = prices.rolling(ma_window, min_periods=ma_window).mean()
    direction = pd.DataFrame(
        np.where(prices < sma, 1.0, -1.0), index=prices.index, columns=prices.columns
    )
    adx_val = adx(prices, window=adx_window)
    strength = (adx_val / adx_scale).clip(lower=0.0, upper=1.0)

    score = 0.5 + 0.5 * direction * strength
    return score.where(sma.notna() & adx_val.notna())
