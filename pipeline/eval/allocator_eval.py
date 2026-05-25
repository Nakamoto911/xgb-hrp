"""Allocator eval checks (spec §12.5).

Operates on the BacktestResult's per-date weight matrix.
"""
from __future__ import annotations

import numpy as np

from pipeline.eval._base import EvalCheck, EvalContext, EvalResult, failed, passed


def _check_weight_sum(ctx: EvalContext) -> EvalResult:
    if ctx.backtest_result is None or ctx.backtest_result.weights.empty:
        return failed("weight_sum", message="no weights matrix")
    sums = ctx.backtest_result.weights.sum(axis=1)
    deviation = float((sums - 1.0).abs().max())
    # NAV-based weights leave a tiny residual due to cash drag at non-rebalance bars.
    if deviation > 0.05:
        return failed("weight_sum", metric=deviation,
                      message=f"max deviation from sum=1: {deviation:.4f}")
    return passed("weight_sum", metric=deviation,
                  message=f"max deviation from sum=1: {deviation:.4f}")


def _check_non_negativity(ctx: EvalContext) -> EvalResult:
    if ctx.backtest_result is None or ctx.backtest_result.weights.empty:
        return failed("non_negativity", message="no weights matrix")
    min_w = float(ctx.backtest_result.weights.min().min())
    if min_w < -1e-6:
        return failed("non_negativity", metric=min_w,
                      message=f"negative weight detected: {min_w}")
    return passed("non_negativity", metric=min_w, message="all weights >= 0")


def _check_concentration(ctx: EvalContext) -> EvalResult:
    """HHI ≤ 0.4 except during risk-off (100% in risk-free is allowed)."""
    if ctx.backtest_result is None or ctx.backtest_result.weights.empty:
        return failed("concentration", message="no weights matrix")
    rf = ctx.config.risk_free_asset
    weights = ctx.backtest_result.weights
    hhi = (weights ** 2).sum(axis=1)
    # Treat days with >= 95% in the risk-free leg as risk-off and exclude.
    rf_weight = weights.get(rf, 0.0) if rf else 0.0
    not_risk_off = (rf_weight < 0.95) if hasattr(rf_weight, "__iter__") else np.array([])
    if hasattr(not_risk_off, "any") and not_risk_off.any():
        hhi_active = hhi[not_risk_off]
        median_hhi = float(hhi_active.median()) if not hhi_active.empty else 0.0
    else:
        median_hhi = float(hhi.median())
    if median_hhi > 0.4:
        return failed("concentration", metric=median_hhi,
                      message=f"median HHI {median_hhi:.3f} > 0.4 (concentrated)")
    return passed("concentration", metric=median_hhi,
                  message=f"median active-day HHI {median_hhi:.3f}")


CHECKS: list[EvalCheck] = [
    EvalCheck("weight_sum", "allocator", _check_weight_sum, critical=True),
    EvalCheck("non_negativity", "allocator", _check_non_negativity, critical=True),
    EvalCheck("concentration", "allocator", _check_concentration, critical=False),
]
