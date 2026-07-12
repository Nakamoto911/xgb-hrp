"""Per-asset XGBoost forecast panel + selection-rule helpers — Module 4 / §6 of SPEC.md.

Output: long-form parquet
``[symbol, asset_name, date, p_bull, p_bear, p_bull_smoothed, regime_forecast]``.

Vendor convention reminder (vendor/xgboost/main.py):
    * State 0 = Bull, State 1 = Bear.
    * ``Raw_Prob`` is P(state == 1) = P(bear), unshifted.
    * ``State_Prob`` is the EWMA-smoothed P(bear) at the paper-tuned halflife.
    * ``Forecast_State`` is the binary (State_Prob > config.prob_threshold) flag,
      also unshifted; the +1-day shift is applied by the executor when trading on it.

We translate to bull-relative quantities so downstream rules read intuitively.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from pipeline._walk_forward import POOL_TO_XGB_UNIVERSE, compute_signals
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


RuleName = Literal[
    "prob_threshold",
    "regime_and_prob",
    "ewma_smoothed",
    "trend",
    "last_day_regime",
    "ma200",
    "hybrid",
]


@dataclass(frozen=True)
class ForecastOutput:
    panel: pd.DataFrame
    path: Path | None = None


# -----------------------------------------------------------------------------
# Build the forecast panel
# -----------------------------------------------------------------------------
def _project_to_forecast_panel(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    required = {"Raw_Prob", "State_Prob", "Forecast_State"}
    for symbol, df in signals.items():
        missing = required - set(df.columns)
        if missing:
            logger.warning("%s missing forecast columns %s — skipping", symbol, missing)
            continue
        asset_name = df.attrs.get("asset_name", symbol)
        long = pd.DataFrame(
            {
                "symbol": symbol,
                "asset_name": asset_name,
                "date": df.index,
                "p_bear": df["Raw_Prob"].to_numpy(),
                "p_bull": (1.0 - df["Raw_Prob"]).to_numpy(),
                "p_bull_smoothed": (1.0 - df["State_Prob"]).to_numpy(),
                "regime_forecast": df["Forecast_State"]
                .map({0: "Bull", 1: "Bear"})
                .to_numpy(),
            }
        )
        rows.append(long)
    if not rows:
        raise RuntimeError("No forecast outputs from any signal frame.")
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])


def forecast_panel_cache_path(config: PipelineConfig) -> Path:
    """Cache path for a pool/window/feature-version forecast panel."""
    end_str = config.end_date or date.today().isoformat()
    return (
        config.cache_dir
        / f"forecasts_{config.pool}_{config.start_date}_{end_str}_{config.feature_version}.parquet"
    )


def build_forecasts(
    config: PipelineConfig,
    *,
    save: bool = True,
    force_refresh: bool = False,
) -> ForecastOutput:
    if config.pool not in POOL_TO_XGB_UNIVERSE:
        raise NotImplementedError(
            f"Forecast build for pool={config.pool!r} not wired. "
            f"Supported pools: {list(POOL_TO_XGB_UNIVERSE)}."
        )
    out_path = forecast_panel_cache_path(config)
    if save and out_path.exists() and not force_refresh:
        logger.info("Loading cached forecast panel from %s", out_path)
        return ForecastOutput(panel=pd.read_parquet(out_path), path=out_path)

    signals = compute_signals(config, force_refresh=force_refresh)
    panel = _project_to_forecast_panel(signals)

    out_path_resolved: Path | None = None
    if save:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)
        out_path_resolved = out_path
        logger.info("Saved forecast panel to %s — shape %s", out_path, panel.shape)
    return ForecastOutput(panel=panel, path=out_path_resolved)


# -----------------------------------------------------------------------------
# Selection rules (§6.3)
# Each returns a boolean Series indexed on date — True means "select symbol".
# Input ``df`` is a single-symbol view of the forecast panel, date-indexed,
# with at least p_bull / p_bull_smoothed / regime_forecast columns.
# -----------------------------------------------------------------------------
def rule_prob_threshold(df: pd.DataFrame, *, theta: float) -> pd.Series:
    return df["p_bull"] >= theta


def rule_regime_and_prob(df: pd.DataFrame, *, theta: float) -> pd.Series:
    return (df["regime_forecast"] == "Bull") & (df["p_bull"] >= theta)


def rule_ewma_smoothed(df: pd.DataFrame, *, theta: float) -> pd.Series:
    return df["p_bull_smoothed"] >= theta


def rule_trend(df: pd.DataFrame, *, window: int) -> pd.Series:
    """True when the OLS slope of p_bull over the last ``window`` days is positive.

    Slope sign is invariant to centering, so we use a closed-form
    ``corr × (sigma_y / sigma_x)`` only via the numerator, which suffices
    for sign and is vectorizable as a rolling sum.
    """
    if window < 2:
        raise ValueError(f"trend rule requires window >= 2, got {window}.")
    y = df["p_bull"].to_numpy(dtype=float)
    n = len(y)
    # Centered x [-(w-1)/2 ... +(w-1)/2]; slope ∝ Σ x_i * y_i over the window.
    x_centered = np.arange(window, dtype=float) - (window - 1) / 2.0
    slope_sign = pd.Series(np.nan, index=df.index)
    if n >= window:
        # Convolution gives Σ x_i * y_{t-w+1+i} for each t ≥ window-1.
        conv = np.convolve(y, x_centered[::-1], mode="valid")  # length n-w+1
        # Pad leading positions with NaN to align with df.index.
        sign_vals = np.full(n, np.nan)
        sign_vals[window - 1:] = conv
        slope_sign.iloc[:] = sign_vals
    return slope_sign > 0


def rule_last_day_regime(df: pd.DataFrame, *, theta: float = 0.0) -> pd.Series:
    """Select when the model's binary regime forecast is Bull. ``theta`` ignored."""
    del theta  # unused — declared for uniform call signature
    return df["regime_forecast"] == "Bull"


def rule_ma200(df: pd.DataFrame, *, price: pd.Series, ma_window: int = 200) -> pd.Series:
    """True when price is at/above its ``ma_window``-day SMA.

    Warmup convention (mirrors cockpit.py's ``_baseline_labels``): while the
    SMA is NaN but price exists, default to bull; where price itself is NaN,
    default to not-selected. ``price`` is the full-history series — the SMA
    is computed before restricting to ``df.index``, so the warmup window is
    anchored to price history, not to the panel's date range.
    """
    sma = price.rolling(ma_window).mean()
    bull = (price >= sma) | (sma.isna() & price.notna())
    return bull.reindex(df.index)


def rule_hybrid(
    df: pd.DataFrame,
    *,
    price: pd.Series,
    ma_window: int = 200,
    hybrid_bear_threshold: float = 0.80,
) -> pd.Series:
    """Bear when price < MA200 OR smoothed p_bear > threshold.

    Combines the price-trend rule with a volatility-regime backstop: an asset
    is only held while it is both above its SMA and the model's smoothed
    bear probability stays at/below ``hybrid_bear_threshold``.
    """
    above_ma = rule_ma200(df, price=price, ma_window=ma_window)
    p_bear_smoothed = 1.0 - df["p_bull_smoothed"]
    return above_ma & (p_bear_smoothed <= hybrid_bear_threshold)


RULES: dict[RuleName, Callable[..., pd.Series]] = {
    "prob_threshold": rule_prob_threshold,
    "regime_and_prob": rule_regime_and_prob,
    "ewma_smoothed": rule_ewma_smoothed,
    "trend": rule_trend,
    "last_day_regime": rule_last_day_regime,
    "ma200": rule_ma200,
    "hybrid": rule_hybrid,
}

# Rules that need the price panel (as opposed to just the forecast panel).
PRICE_RULES = {"ma200", "hybrid"}


def apply_rule(
    panel: pd.DataFrame,
    rule: RuleName,
    *,
    theta: float,
    trend_window: int,
    prices: pd.DataFrame | None = None,
    ma_window: int = 200,
    hybrid_bear_threshold: float = 0.80,
) -> pd.DataFrame:
    """Apply a selection rule across the full long-form forecast panel.

    ``prices`` (date x symbol) is required for the price-aware rules
    (``ma200``, ``hybrid``); every symbol in ``panel`` must have a column.

    Returns a DataFrame with columns ``[symbol, date, selected]``.
    """
    if rule not in RULES:
        raise ValueError(f"Unknown rule {rule!r}. Known: {list(RULES)}.")
    fn = RULES[rule]
    if rule in PRICE_RULES:
        if prices is None:
            raise ValueError(f"rule {rule!r} requires prices (date x symbol close panel).")
        missing = sorted(set(panel["symbol"].unique()) - set(prices.columns))
        if missing:
            raise ValueError(f"prices panel missing symbols required by rule {rule!r}: {missing}.")
    out: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol"):
        sub = group.set_index("date").sort_index()
        if rule == "trend":
            sel = fn(sub, window=trend_window)
        elif rule in PRICE_RULES:
            kwargs = {"price": prices[symbol], "ma_window": ma_window}
            if rule == "hybrid":
                kwargs["hybrid_bear_threshold"] = hybrid_bear_threshold
            sel = fn(sub, **kwargs)
        else:
            sel = fn(sub, theta=theta)
        out.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": sel.index,
                    "selected": sel.fillna(False).to_numpy(dtype=bool),
                }
            )
        )
    return pd.concat(out, ignore_index=True).sort_values(["date", "symbol"])


__all__ = [
    "ForecastOutput",
    "PRICE_RULES",
    "RULES",
    "RuleName",
    "apply_rule",
    "build_forecasts",
    "forecast_panel_cache_path",
    "rule_ewma_smoothed",
    "rule_hybrid",
    "rule_last_day_regime",
    "rule_ma200",
    "rule_prob_threshold",
    "rule_regime_and_prob",
    "rule_trend",
]
