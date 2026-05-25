"""Data-layer tests — no network, using hrp's synthetic price generator."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.data import _cache_path, get_universe_tickers, load_prices


def test_etf_pool_tickers_match_hrp_constant():
    from pipeline import _vendor

    hrp_data = _vendor.vendor_hrp_data()
    assert get_universe_tickers("etf") == list(hrp_data.ETF_POOL)


def test_mutual_fund_pool_tickers_present():
    tickers = get_universe_tickers("mutual_fund")
    assert "VBMFX" in tickers  # the hrp risk-free leg lives here
    assert len(tickers) >= 10


def test_european_pool_tickers_present():
    tickers = get_universe_tickers("european")
    assert any(t.endswith(".DE") for t in tickers)


def test_unknown_pool_raises():
    with pytest.raises(ValueError, match="Unknown pool"):
        get_universe_tickers("does_not_exist")


def test_cache_path_includes_feature_version(tmp_cache_dir):
    p = _cache_path(tmp_cache_dir, "etf", "2007-01-01", "2026-05-24", "v1")
    assert p.name == "prices_etf_2007-01-01_2026-05-24_v1.parquet"
    assert p.parent == tmp_cache_dir


def test_load_prices_uses_parquet_cache_when_present(tmp_cache_dir, synthetic_prices):
    """If a parquet cache exists, load_prices reads it without hitting the network."""
    cfg = PipelineConfig(
        pool="etf",
        start_date="2024-01-02",
        end_date="2024-06-30",
        cache_dir=tmp_cache_dir,
    )
    # Pre-seed the cache with synthetic data
    cache_file = _cache_path(
        tmp_cache_dir, cfg.pool, cfg.start_date, cfg.end_date, cfg.feature_version
    )
    synthetic_prices.to_parquet(cache_file)

    df = load_prices(cfg)  # must NOT touch network
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(df.columns) == set(synthetic_prices.columns)
    assert len(df) == len(synthetic_prices)
