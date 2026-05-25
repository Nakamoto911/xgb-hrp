"""Data-layer eval checks (spec §12.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.eval._base import EvalCheck, EvalContext, EvalResult, failed, passed


def _check_missing_data(ctx: EvalContext) -> EvalResult:
    """NaN share per asset, computed on each asset's valid range only.

    Pre-inception NaN (e.g. SPBO before 2011 in a 2007 OOS) is structural
    and shouldn't be flagged — the spec budgets 0.5% NaN inside the live
    history, not across the union of all assets' calendars.
    """
    if ctx.prices is None or ctx.prices.empty:
        return failed("missing_data", message="no prices in context")
    per_asset_nan: list[float] = []
    offenders: list[str] = []
    for col in ctx.prices.columns:
        series = ctx.prices[col]
        first_valid = series.first_valid_index()
        if first_valid is None:
            offenders.append(f"{col}(all-NaN)")
            per_asset_nan.append(1.0)
            continue
        live = series.loc[first_valid:]
        nan_share = float(live.isna().mean())
        per_asset_nan.append(nan_share)
        if nan_share >= 0.005:
            offenders.append(f"{col}({nan_share:.2%})")
    worst = max(per_asset_nan) if per_asset_nan else 0.0
    if offenders:
        return failed("missing_data", metric=worst,
                      message=f"in-life NaN > 0.5% for: {offenders[:5]}")
    return passed("missing_data", metric=worst,
                  message=f"max in-life NaN share {worst:.3%}")


def _check_calendar_alignment(ctx: EvalContext) -> EvalResult:
    if ctx.prices is None or ctx.prices.empty:
        return failed("calendar_alignment", message="no prices")
    idx = ctx.prices.index
    if not idx.is_monotonic_increasing:
        return failed("calendar_alignment", message="prices index not monotonic")
    if idx.has_duplicates:
        return failed("calendar_alignment", message="prices index has duplicates")
    return passed("calendar_alignment", metric=len(idx),
                  message=f"{len(idx)} unique sorted business days")


def _check_outliers(ctx: EvalContext) -> EvalResult:
    """Flag any |daily log-return| > 7σ (per-asset)."""
    if ctx.prices is None or ctx.prices.empty:
        return failed("outliers", message="no prices")
    log_ret = np.log(ctx.prices / ctx.prices.shift(1)).dropna(how="all")
    if log_ret.empty:
        return passed("outliers", message="not enough rows to test")
    z = (log_ret - log_ret.mean()) / log_ret.std(ddof=1).replace(0.0, np.nan)
    n_outliers = int((z.abs() > 7).sum().sum())
    if n_outliers > 0:
        # Non-critical: outliers can be real (e.g. ETF splits before cleanup).
        return failed("outliers", metric=n_outliers,
                      message=f"{n_outliers} bars with |z| > 7 detected")
    return passed("outliers", metric=0.0, message="no >7σ bars")


def _check_stale_streak(ctx: EvalContext) -> EvalResult:
    """Flag any flat-price run ≥ 5 days in any asset."""
    if ctx.prices is None or ctx.prices.empty:
        return failed("stale_streak", message="no prices")
    max_run = 0
    worst = ""
    for col in ctx.prices.columns:
        s = ctx.prices[col].dropna()
        if s.empty:
            continue
        runs = (s.diff().fillna(1.0) == 0.0).astype(int)
        # Run lengths via cumulative reset on zeros.
        run_lengths = runs.groupby((runs == 0).cumsum()).cumsum()
        m = int(run_lengths.max())
        if m > max_run:
            max_run, worst = m, col
    if max_run >= 5:
        return failed("stale_streak", metric=max_run,
                      message=f"{worst} flat for {max_run} consecutive days")
    return passed("stale_streak", metric=max_run,
                  message=f"max flat-run = {max_run} days")


CHECKS: list[EvalCheck] = [
    EvalCheck("missing_data", "data", _check_missing_data, critical=True),
    EvalCheck("calendar_alignment", "data", _check_calendar_alignment, critical=True),
    EvalCheck("outliers", "data", _check_outliers, critical=False),
    EvalCheck("stale_streak", "data", _check_stale_streak, critical=False),
]
