"""Portfolio allocator — Module 6 of SPEC.md.

Public entry point ``allocate(returns, selected, config)`` dispatches to
one of four allocators:

    hrp           — vendor/hrp/hrp_engine/hrp.optimize_hrp with RMT-denoised cov
    ew            — equal weight across the selected set
    momentum_30d  — clipped trailing 30-day return, EW fallback if all clipped
    min_vol_30d   — inverse trailing 30-day vol

All return a ``pd.Series`` of weights summing to 1.0 (or an empty Series
when ``selected`` is empty — the caller routes those rebalances to the
risk-free asset).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline import _vendor
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

_MIN_VOL_FLOOR = 1e-6


# -----------------------------------------------------------------------------
# Individual allocators
# -----------------------------------------------------------------------------
def _ew(selected: list[str]) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    w = 1.0 / len(selected)
    return pd.Series({s: w for s in selected})


def _momentum_30d(returns: pd.DataFrame, selected: list[str]) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    window = returns.iloc[-30:][selected]
    if window.shape[0] < 30:
        logger.warning(
            "momentum_30d: only %d obs available, need 30 — falling back to EW.",
            window.shape[0],
        )
        return _ew(selected)
    # Total return over the trailing 30 trading days, per asset.
    total_ret = (1.0 + window).prod() - 1.0
    clipped = total_ret.clip(lower=0.0)
    denom = clipped.sum()
    if denom <= 0:
        return _ew(selected)
    w = clipped / denom
    return w.reindex(selected).fillna(0.0)


def _min_vol_30d(returns: pd.DataFrame, selected: list[str]) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    window = returns.iloc[-30:][selected]
    if window.shape[0] < 30:
        logger.warning(
            "min_vol_30d: only %d obs available, need 30 — falling back to EW.",
            window.shape[0],
        )
        return _ew(selected)
    sigma = window.std(ddof=1).clip(lower=_MIN_VOL_FLOOR)
    inv = 1.0 / sigma
    w = inv / inv.sum()
    return w.reindex(selected).fillna(0.0)


def _hrp(
    returns: pd.DataFrame,
    selected: list[str],
    config: PipelineConfig,
) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    if len(selected) == 1:
        return pd.Series({selected[0]: 1.0})

    lookback_days = config.hrp_lookback_years * 252
    window = returns.iloc[-lookback_days:][selected].dropna()
    if window.shape[0] < max(60, len(selected) + 1):
        logger.warning(
            "hrp: lookback only %d rows for %d assets — falling back to EW.",
            window.shape[0], len(selected),
        )
        return _ew(selected)

    cov = window.cov()

    if config.hrp_denoise and len(selected) >= 2:
        from hrp_engine import denoiser as hrp_denoiser  # type: ignore[import-not-found]

        cov_np = hrp_denoiser.denoise_covariance(cov.to_numpy(), n_obs=window.shape[0])
        cov = pd.DataFrame(cov_np, index=cov.index, columns=cov.columns)

    hrp = _vendor.vendor_hrp_hrp()
    weights = hrp.optimize_hrp(
        cov,
        linkage_method=config.hrp_linkage,
        bisection_method=config.hrp_bisection,
    )
    # optimize_hrp returns a Series indexed by the selected symbols.
    return weights.reindex(selected).fillna(0.0)


# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------
def allocate(
    returns: pd.DataFrame,
    selected: list[str],
    config: PipelineConfig,
) -> pd.Series:
    """Allocate capital across ``selected`` symbols using the configured allocator.

    Parameters
    ----------
    returns : pd.DataFrame
        Date-indexed daily returns panel. Columns must be a superset of
        ``selected``.
    selected : list[str]
        Symbols passing the asset selector for this rebalance.
    config : PipelineConfig
        Driver of allocator choice + HRP params.
    """
    missing = [s for s in selected if s not in returns.columns]
    if missing:
        raise KeyError(f"returns panel is missing columns for selected symbols: {missing}")

    alloc = config.allocator
    if alloc == "ew":
        w = _ew(selected)
    elif alloc == "momentum_30d":
        w = _momentum_30d(returns, selected)
    elif alloc == "min_vol_30d":
        w = _min_vol_30d(returns, selected)
    elif alloc == "hrp":
        w = _hrp(returns, selected, config)
    else:
        raise ValueError(f"Unknown allocator {alloc!r}.")

    # Re-index to selected order and assert weights are valid (long-only, sum=1).
    if not w.empty:
        if (w < -1e-9).any():
            raise ValueError(f"Allocator {alloc!r} produced negative weights: {w[w < 0]}")
        total = w.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            logger.warning(
                "Allocator %s weights sum to %.9f (≠ 1) — normalizing.", alloc, total
            )
            w = w / total
    return w.reindex(selected).fillna(0.0)


__all__ = ["allocate"]
