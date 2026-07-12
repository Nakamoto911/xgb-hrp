"""scripts/compare_rules.py harness test — synthetic warm caches, no network.

Builds a synthetic price/forecast/risk-free/benchmark cache set on disk
(matching the exact filenames pipeline.data / pipeline.forecast expect),
then drives the real `compare()` entry point end-to-end. The synthetic
price panel injects a sustained downtrend on one symbol (IVV) so the
price-aware rules (ma200, hybrid) actually diverge from the production
(ewma_smoothed) rule, which is blind to price and stays fully invested
throughout — mirroring the MEMORY.md 2026-07-07 "production loses to
MA200" finding at the per-asset level, now exercised through the full
portfolio backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.data import _cache_path
from pipeline.forecast import apply_rule
from scripts.compare_rules import VARIANTS, compare

N_DAYS = 600
SYMBOLS = ["IVV", "AGG", "GLD", "QQQ"]
RF_TICKER = "BIL"
BENCH_BH = "^GSPC"
BENCH_6040 = "VSMGX"
FEATURE_VERSION = "v1"
POOL = "etf"
# Sustained downtrend window on IVV — starts well after the 200-day SMA
# warmup so ma200/hybrid have a primed baseline to cross below.
DOWN_START, DOWN_END = 260, 420


def _synthetic_prices(seed: int = 7) -> pd.DataFrame:
    """4 risky assets. IVV gets a sustained crash + partial recovery so the
    price-aware rules (ma200, hybrid) have something to deselect; the other
    three trend gently so the divergence is attributable to IVV."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=N_DAYS, freq="B")

    ivv_drift = np.full(N_DAYS, 0.0006)
    ivv_drift[DOWN_START:DOWN_END] = -0.0045  # sustained crash (~-45% cumulative)
    ivv_drift[DOWN_END:] = 0.0010  # partial recovery afterward
    ivv = 100.0 * np.exp(np.cumsum(ivv_drift + 0.010 * rng.standard_normal(N_DAYS)))

    agg = 100.0 * np.exp(np.cumsum(0.0002 + 0.004 * rng.standard_normal(N_DAYS)))
    gld = 100.0 * np.exp(np.cumsum(0.0003 + 0.008 * rng.standard_normal(N_DAYS)))
    qqq = 100.0 * np.exp(np.cumsum(0.0005 + 0.011 * rng.standard_normal(N_DAYS)))

    return pd.DataFrame({"IVV": ivv, "AGG": agg, "GLD": gld, "QQQ": qqq}, index=idx)


def _synthetic_rf(idx: pd.DatetimeIndex) -> pd.DataFrame:
    bil = 100.0 * np.exp(np.cumsum(np.full(len(idx), 0.0001)))
    return pd.DataFrame({RF_TICKER: bil}, index=idx)


def _synthetic_benchmarks(idx: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sp = 100.0 * np.exp(np.cumsum(0.0004 + 0.010 * rng.standard_normal(len(idx))))
    balanced = 100.0 * np.exp(np.cumsum(0.0003 + 0.006 * rng.standard_normal(len(idx))))
    return pd.DataFrame({BENCH_BH: sp, BENCH_6040: balanced}, index=idx)


def _synthetic_forecast(idx: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Constant bull-leaning forecast for every symbol/date.

    Keeps the production (`ewma_smoothed`) rule fully invested throughout —
    including through IVV's synthetic crash, since it never looks at price —
    so any divergence from ma200/hybrid in the backtest is attributable to
    the selection rule, not to the forecast panel. p_bear never nears the
    risk-monitor / hybrid thresholds, so those stay inert by design; this
    isolates the effect the harness exists to measure.
    """
    rows = [
        {
            "symbol": sym, "asset_name": sym, "date": d,
            "p_bear": 0.20, "p_bull": 0.80, "p_bull_smoothed": 0.80,
            "regime_forecast": "Bull",
        }
        for sym in symbols
        for d in idx
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def warm_cache(tmp_path):
    """Write every parquet cache file compare_rules.compare() needs, using
    the exact naming conventions of pipeline.data / pipeline.forecast."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    prices = _synthetic_prices()
    idx = prices.index
    rf = _synthetic_rf(idx)
    bench = _synthetic_benchmarks(idx)
    forecast = _synthetic_forecast(idx, SYMBOLS)

    start, end = str(idx[0].date()), str(idx[-1].date())

    prices.to_parquet(_cache_path(cache_dir, POOL, start, end, FEATURE_VERSION))
    rf.to_parquet(_cache_path(cache_dir, f"{POOL}_rf", start, end, FEATURE_VERSION))
    bench.to_parquet(_cache_path(cache_dir, f"{POOL}_bench", start, end, FEATURE_VERSION))
    forecast.to_parquet(
        cache_dir / f"forecasts_{POOL}_{start}_{end}_{FEATURE_VERSION}.parquet"
    )

    return cache_dir, start, end, prices, forecast


# -----------------------------------------------------------------------------
# Sanity check on the synthetic fixture itself: the crash actually flips the
# price-aware rule but leaves the (price-blind) production rule untouched.
# -----------------------------------------------------------------------------
def test_synthetic_downtrend_deselects_ivv_under_ma200_not_production(warm_cache):
    _cache_dir, _start, _end, prices, forecast = warm_cache
    ma_flags = apply_rule(forecast, "ma200", theta=0.5, trend_window=5, prices=prices, ma_window=200)
    ivv_ma = ma_flags[ma_flags["symbol"] == "IVV"].set_index("date")["selected"]

    prod_flags = apply_rule(forecast, "ewma_smoothed", theta=0.60, trend_window=5)
    ivv_prod = prod_flags[prod_flags["symbol"] == "IVV"].set_index("date")["selected"]

    # Deep inside the crash (well past the SMA-crossing lag) ma200 must have
    # deselected IVV on at least some days, while production — blind to
    # price — never wavers.
    crash_dates = prices.index[300:400]
    assert not ivv_ma.loc[crash_dates].all()
    assert ivv_prod.loc[crash_dates].all()


# -----------------------------------------------------------------------------
# Fail-fast on missing caches.
# -----------------------------------------------------------------------------
def test_missing_cache_fails_fast_naming_exact_path(tmp_path):
    cfg = PipelineConfig(
        pool="etf", start_date="2015-01-02", end_date="2015-06-01",
        cache_dir=tmp_path / "empty_cache",
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        compare(cfg, write=False, print_table=False)
    msg = str(exc_info.value)
    assert "prices_etf_2015-01-02_2015-06-01_v1.parquet" in msg
    assert "run_pipeline.py --phase all" in msg
    assert "--pool etf" in msg
    assert "--start-date 2015-01-02" in msg


# -----------------------------------------------------------------------------
# End-to-end: the real comparison table.
# -----------------------------------------------------------------------------
def test_compare_rules_on_synthetic_caches(warm_cache):
    cache_dir, start, end, _prices, _forecast = warm_cache
    cfg = PipelineConfig(
        pool="etf", start_date=start, end_date=end, cache_dir=cache_dir,
        # EW (not HRP) keeps IVV's weight share deterministic and undiminished
        # by covariance-based downweighting, so the MDD divergence assertion
        # below is robust rather than depending on HRP's denoiser output.
        allocator="ew",
    )

    table = compare(cfg, write=True, print_table=False)

    variant_names = [name for name, _ in VARIANTS]
    assert list(table.index)[: len(variant_names)] == variant_names
    assert "S&P_BH" in table.index
    assert "60_40" in table.index
    assert len(table) == len(variant_names) + 2

    strategy_cols = [
        "cagr", "sharpe", "sortino", "mdd", "calmar", "ann_vol",
        "turnover_annual", "tax_drag", "tc_drag", "trades_per_year", "n_risk_events",
    ]
    for name in variant_names:
        row = table.loc[name, strategy_cols].astype(float)
        assert np.isfinite(row).all(), f"{name}: {row.to_dict()}"
        assert row["trades_per_year"] >= 0
        assert row["n_risk_events"] >= 0

    bench_cols = ["cagr", "sharpe", "sortino", "mdd", "calmar", "ann_vol", "tax_drag", "tc_drag"]
    for name in ("S&P_BH", "60_40"):
        row = table.loc[name, bench_cols].astype(float)
        assert np.isfinite(row).all(), f"{name}: {row.to_dict()}"
        # Benchmarks are gross — no tax, no transaction costs.
        assert row["tax_drag"] == pytest.approx(0.0)
        assert row["tc_drag"] == pytest.approx(0.0)

    # Decisive check: ma200/hybrid must diverge from BOTH production variants
    # — that's the entire reason this harness exists (MEMORY.md 2026-07-07).
    prod_names = ["production_theta060", "production_theta040"]
    trend_names = ["ma200", "hybrid"]
    for pn in prod_names:
        for tn in trend_names:
            assert table.loc[pn, "cagr"] != pytest.approx(table.loc[tn, "cagr"], abs=1e-6)
            assert table.loc[pn, "mdd"] != pytest.approx(table.loc[tn, "mdd"], abs=1e-6)

    # ma200/hybrid sidestep part of IVV's synthetic crash -> shallower (less
    # negative) max drawdown than the always-fully-invested production rows.
    for tn in trend_names:
        for pn in prod_names:
            assert table.loc[tn, "mdd"] > table.loc[pn, "mdd"], (
                f"{tn} mdd={table.loc[tn, 'mdd']} should be shallower than "
                f"{pn} mdd={table.loc[pn, 'mdd']}"
            )

    # Output file written with the same content.
    out_path = cache_dir / f"compare_rules_etf_{start}_{end}.md"
    assert out_path.exists()
    md = out_path.read_text()
    for name in variant_names:
        assert name in md
    assert "S&P_BH" in md and "60_40" in md
    assert "NET" in md and "GROSS" in md
    assert "etf" in md
