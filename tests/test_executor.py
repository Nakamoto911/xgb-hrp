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


# -----------------------------------------------------------------------------
# flip_only execution policy (SPEC §9.2.1 — tax-aware)
# -----------------------------------------------------------------------------
def _deterministic_prices(n_days: int = 400) -> pd.DataFrame:
    """Noise-free exponential paths so weight-drift arithmetic is exact.
    HOT rallies hard, FLAT and COLD stay put, BIL compounds slowly."""
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B")
    t = np.arange(n_days)
    return pd.DataFrame(
        {
            "COLD": 100.0 * np.exp(0.0000 * t),
            "FLAT": 100.0 * np.exp(0.0000 * t),
            "HOT": 100.0 * np.exp(0.0040 * t),
            "BIL": 100.0 * np.exp(0.0001 * t),
        },
        index=idx,
    )


def _bull_panel(
    prices: pd.DataFrame, symbols: list[str], bear_from: dict[str, int] | None = None
) -> pd.DataFrame:
    """All-bull forecast panel; ``bear_from[sym] = day_offset`` flips that
    symbol to bear (drops it from selection) from that offset onward."""
    bear_from = bear_from or {}
    rows = []
    for sym in symbols:
        for offset, date in enumerate(prices.index):
            bear = offset >= bear_from.get(sym, len(prices) + 1)
            p_bear = 0.9 if bear else 0.1
            rows.append({
                "symbol": sym, "asset_name": sym, "date": date,
                "p_bear": p_bear, "p_bull": 1.0 - p_bear,
                "p_bull_smoothed": 1.0 - p_bear,
                "regime_forecast": "Bear" if bear else "Bull",
            })
    return pd.DataFrame(rows)


def _flip_only_cfg(prices: pd.DataFrame, **overrides) -> PipelineConfig:
    base = dict(
        pool="etf", risk_free_asset="BIL",
        rebalance_frequency="monthly",
        risk_monitor_enabled=False,
        forecast_method="prob_threshold", bull_prob_threshold=0.55,
        allocator="ew",
        execution_policy="flip_only",
        start_date=str(prices.index[0].date()), end_date=str(prices.index[-1].date()),
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_flip_only_enters_below_band():
    """Flip-in entries are never band-gated: with EW targets (1/3 ≈ 0.33) below
    a 0.40 band, drift_band never invests at all while flip_only deploys the
    full book on the first rebalance."""
    prices = _deterministic_prices()
    panel = _bull_panel(prices, ["COLD", "FLAT", "HOT"])
    res_flip = Executor(
        config=_flip_only_cfg(prices, drift_threshold=0.40),
        prices=prices, forecast_panel=panel,
    ).run()
    res_band = Executor(
        config=_flip_only_cfg(prices, drift_threshold=0.40, execution_policy="drift_band"),
        prices=prices, forecast_panel=panel,
    ).run()
    assert res_band.trades.empty  # 0.33 target < 0.40 gate: drift_band stuck in cash
    last_w = res_flip.weights.iloc[-1]
    assert last_w.sum() > 0.95  # flip_only fully invested


def test_flip_only_exits_dropped_symbol_below_band():
    """A selection flip-out always sells in full, even when the position's
    weight is far below the band; drift_band would keep holding it."""
    prices = _deterministic_prices()
    panel = _bull_panel(prices, ["COLD", "FLAT", "HOT"], bear_from={"COLD": 150})
    band = dict(drift_threshold=0.60)
    res_flip = Executor(
        config=_flip_only_cfg(prices, **band), prices=prices, forecast_panel=panel,
    ).run()
    res_band = Executor(
        config=_flip_only_cfg(prices, **band, execution_policy="drift_band"),
        prices=prices, forecast_panel=panel,
    ).run()
    # flip_only: COLD is gone after the first post-flip rebalance…
    assert res_flip.weights.iloc[-1].get("COLD", 0.0) == 0.0
    cold_sells = res_flip.trades[
        (res_flip.trades["symbol"] == "COLD") & (res_flip.trades["side"] == "sell")
    ]
    assert not cold_sells.empty
    # …while drift_band never even entered (weights below the 0.60 gate).
    assert res_band.trades.empty


def test_flip_only_does_not_sell_winners_within_cap():
    """A continuing winner drifting above target but within target+band is
    left alone — no sell, no realized gain (the tax-aware core guarantee)."""
    prices = _deterministic_prices(n_days=120)  # HOT drifts up but stays < 0.5+0.30
    panel = _bull_panel(prices, ["FLAT", "HOT"])
    res = Executor(
        config=_flip_only_cfg(prices, drift_threshold=0.30),
        prices=prices, forecast_panel=panel,
    ).run()
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells.empty  # nothing was ever sold — HOT rode from 0.50 to <0.80 untouched
    assert res.total_tax == 0.0


def test_flip_only_trims_to_cap_not_to_target():
    """Above target+band the winner is trimmed to the cap, not re-banded to
    target: at the first trim event (same date, same trigger threshold for
    both policies) flip_only sells only the overshoot above target+band while
    drift_band sells all the way down to target — a strictly larger
    realization. (Cumulative tax over a long monotone rally is NOT smaller
    for flip_only: the fatter winner position compounds more dollars of gain;
    the policy's guarantee is minimal realization per event plus deferral.)"""
    prices = _deterministic_prices()  # HOT: +0.4%/day → breaches 0.5+0.10 fast
    panel = _bull_panel(prices, ["FLAT", "HOT"])
    band = dict(drift_threshold=0.10)
    res_flip = Executor(
        config=_flip_only_cfg(prices, **band), prices=prices, forecast_panel=panel,
    ).run()
    res_band = Executor(
        config=_flip_only_cfg(prices, **band, execution_policy="drift_band"),
        prices=prices, forecast_panel=panel,
    ).run()

    def hot_sells(res):
        return res.trades[
            (res.trades["symbol"] == "HOT") & (res.trades["side"] == "sell")
        ]

    flip_sells, band_sells = hot_sells(res_flip), hot_sells(res_band)
    assert not flip_sells.empty  # the cap does bind on a hard rally
    # Both policies trigger at the same threshold → same first-event date.
    assert flip_sells["date"].iloc[0] == band_sells["date"].iloc[0]
    # Post-trim weight sits at the cap (0.60) for flip_only, back at target
    # (0.50) for drift_band.
    trim_date = flip_sells["date"].iloc[0]
    assert res_flip.weights.loc[trim_date, "HOT"] == pytest.approx(0.60, abs=0.02)
    assert res_band.weights.loc[trim_date, "HOT"] == pytest.approx(0.50, abs=0.02)
    # Minimal realization per event, and taxes actually accrue.
    assert 0.0 < flip_sells["realized_gain"].iloc[0] < band_sells["realized_gain"].iloc[0]
    assert res_flip.total_tax > 0.0


def test_flip_only_empty_selection_full_exit_to_risk_free():
    """When every symbol flips bear, flip_only parks 100% in the risk-free leg
    even with a band far wider than any single position weight."""
    prices = _deterministic_prices()
    panel = _bull_panel(
        prices, ["COLD", "FLAT", "HOT"],
        bear_from={"COLD": 150, "FLAT": 150, "HOT": 150},
    )
    res = Executor(
        config=_flip_only_cfg(prices, drift_threshold=0.60),
        prices=prices, forecast_panel=panel,
    ).run()
    assert res.weights.iloc[-1].get("BIL", 0.0) > 0.99


def test_flip_only_is_deterministic():
    """Two identical runs must be byte-identical (guards the 73e70af fix)."""
    prices = _deterministic_prices()
    panel = _bull_panel(prices, ["COLD", "FLAT", "HOT"], bear_from={"COLD": 150})
    cfg = _flip_only_cfg(prices, drift_threshold=0.20)
    res1 = Executor(config=cfg, prices=prices, forecast_panel=panel).run()
    res2 = Executor(config=cfg, prices=prices, forecast_panel=panel).run()
    pd.testing.assert_series_equal(res1.nav, res2.nav)
    pd.testing.assert_frame_equal(res1.trades, res2.trades)


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
