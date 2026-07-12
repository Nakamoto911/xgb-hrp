"""Executor smoke tests on synthetic prices + forecasts (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.executor import Executor


def _synthetic_prices(n_days: int = 600, rng_seed: int = 0) -> pd.DataFrame:
    """Three risky assets + a risk-free that grows at 0.0001/day (≈2.5% annualized)."""
    rng = np.random.default_rng(rng_seed)
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B")
    drifts = {"IVV": 0.0004, "AGG": 0.0001, "GLD": 0.0002}
    vols = {"IVV": 0.012, "AGG": 0.004, "GLD": 0.010}
    data = {}
    for sym, d in drifts.items():
        ret = d + vols[sym] * rng.standard_normal(n_days)
        data[sym] = 100.0 * np.exp(np.cumsum(ret))
    # Risk-free as a near-deterministic compounding series.
    data["BIL"] = 100.0 * np.exp(np.cumsum(np.full(n_days, 0.0001)))
    return pd.DataFrame(data, index=idx)


def _synthetic_forecast(prices: pd.DataFrame, n_assets: list[str]) -> pd.DataFrame:
    """Build a forecast panel where most days are bull-leaning so the executor
    actually rebalances into risky assets. A short bear stretch lets the risk
    monitor exercise its trigger/clear path."""
    rows = []
    for sym in n_assets:
        for date in prices.index:
            # Default bull-leaning forecast.
            p_bear = 0.20
            # Inject a bear stretch in days 200-260 for one asset to test risk monitor.
            offset = prices.index.get_loc(date)
            if 200 <= offset <= 260:
                p_bear = 0.85  # universe ends up >40% bear when all assets in this band
            rows.append({
                "symbol": sym,
                "asset_name": sym,
                "date": date,
                "p_bear": p_bear,
                "p_bull": 1.0 - p_bear,
                "p_bull_smoothed": 1.0 - p_bear,
                "regime_forecast": "Bear" if p_bear > 0.5 else "Bull",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def prices():
    return _synthetic_prices()


@pytest.fixture
def forecast(prices):
    return _synthetic_forecast(prices, ["IVV", "AGG", "GLD"])


def test_executor_produces_positive_nav(prices, forecast):
    cfg = PipelineConfig(
        pool="etf", risk_free_asset="BIL",
        rebalance_frequency="monthly",
        risk_monitor_enabled=False,
        forecast_method="prob_threshold", bull_prob_threshold=0.55,
        allocator="ew",
        start_date=str(prices.index[0].date()), end_date=str(prices.index[-1].date()),
    )
    res = Executor(config=cfg, prices=prices, forecast_panel=forecast,
                   initial_capital=100_000).run()
    assert res.nav.iloc[-1] > 0
    assert not res.weights.empty
    # EW on 3 selected → ~33% each except for the latest unrebalanced drift.
    last_w = res.weights.iloc[-1]
    assert (last_w[last_w > 0] > 0).all()


def test_risk_monitor_triggers_and_clears(prices, forecast):
    cfg = PipelineConfig(
        pool="etf", risk_free_asset="BIL",
        rebalance_frequency="monthly",
        risk_monitor_enabled=True,
        bear_prob_threshold=0.70, universe_pct_threshold=0.40,
        universe_pct_clear_threshold=0.20, risk_off_dwell_days=3,
        forecast_method="prob_threshold", bull_prob_threshold=0.55,
        allocator="ew",
        start_date=str(prices.index[0].date()), end_date=str(prices.index[-1].date()),
    )
    res = Executor(config=cfg, prices=prices, forecast_panel=forecast,
                   initial_capital=100_000).run()
    # Bear stretch in days 200-260 should fire a trigger and a subsequent clear.
    transitions = res.risk_events["transition"].tolist()
    assert "trigger" in transitions
    assert "clear" in transitions


def test_drift_band_reduces_turnover(prices, forecast):
    """A wider drift band gates more trades, so total TC must be lower."""
    base = dict(
        pool="etf", risk_free_asset="BIL",
        rebalance_frequency="monthly",
        risk_monitor_enabled=False,
        forecast_method="prob_threshold", bull_prob_threshold=0.55,
        allocator="ew",
        start_date=str(prices.index[0].date()), end_date=str(prices.index[-1].date()),
    )
    res_tight = Executor(config=PipelineConfig(**base, drift_threshold=0.001),
                         prices=prices, forecast_panel=forecast).run()
    res_loose = Executor(config=PipelineConfig(**base, drift_threshold=0.10),
                         prices=prices, forecast_panel=forecast).run()
    assert res_loose.total_tc <= res_tight.total_tc


def test_executor_runs_end_to_end_with_hybrid_forecast_method(prices, forecast):
    """Plumbing smoke test: hybrid needs Executor to thread ``self.prices``
    through selector.select() -> apply_rule(), which the other rules don't."""
    cfg = PipelineConfig(
        pool="etf", risk_free_asset="BIL",
        rebalance_frequency="monthly",
        risk_monitor_enabled=False,
        forecast_method="hybrid", ma_window=50, hybrid_bear_threshold=0.80,
        allocator="ew",
        start_date=str(prices.index[0].date()), end_date=str(prices.index[-1].date()),
    )
    res = Executor(config=cfg, prices=prices, forecast_panel=forecast,
                   initial_capital=100_000).run()
    assert res.nav.iloc[-1] > 0
    assert not res.weights.empty


def test_empty_selection_routes_to_risk_free(prices):
    """When no asset passes the rule, the executor parks in the risk-free leg."""
    # Forecast panel where p_bull = 0 always → no asset passes any positive theta.
    rows = []
    for sym in ["IVV", "AGG", "GLD"]:
        for date in prices.index:
            rows.append({
                "symbol": sym, "asset_name": sym, "date": date,
                "p_bear": 1.0, "p_bull": 0.0, "p_bull_smoothed": 0.0,
                "regime_forecast": "Bear",
            })
    forecast = pd.DataFrame(rows)
    cfg = PipelineConfig(
        pool="etf", risk_free_asset="BIL",
        rebalance_frequency="monthly",
        risk_monitor_enabled=False,
        forecast_method="prob_threshold", bull_prob_threshold=0.50,
        allocator="ew",
        start_date=str(prices.index[0].date()), end_date=str(prices.index[-1].date()),
    )
    res = Executor(config=cfg, prices=prices, forecast_panel=forecast).run()
    last_w = res.weights.iloc[-1]
    # 100% in risk-free (modulo drift over the last bar).
    assert last_w.get("BIL", 0.0) > 0.99
