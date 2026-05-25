"""Portfolio allocator — dispatcher + each allocator on synthetic returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.allocator import allocate
from pipeline.config import PipelineConfig


@pytest.fixture
def returns_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_days, symbols = 1200, ["IVV", "AGG", "GLD", "DBC"]  # 4.5 years × 4 assets
    # Different vols + slight drifts so allocators distinguish them.
    vols = [0.012, 0.004, 0.010, 0.018]
    drifts = [0.0006, 0.0001, 0.0003, 0.0]
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = {
        s: drifts[i] + vols[i] * rng.standard_normal(n_days) for i, s in enumerate(symbols)
    }
    return pd.DataFrame(data, index=idx)


def test_ew_equal_weights(returns_panel):
    cfg = PipelineConfig(allocator="ew")
    w = allocate(returns_panel, ["IVV", "AGG", "GLD"], cfg)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w.values, [1/3, 1/3, 1/3])


def test_empty_selected_returns_empty_series(returns_panel):
    cfg = PipelineConfig(allocator="ew")
    w = allocate(returns_panel, [], cfg)
    assert w.empty


def test_singleton_collapses_to_100pct(returns_panel):
    for allocator in ("ew", "momentum_30d", "min_vol_30d", "hrp"):
        cfg = PipelineConfig(allocator=allocator)
        w = allocate(returns_panel, ["IVV"], cfg)
        assert np.isclose(w.sum(), 1.0)
        assert w["IVV"] == pytest.approx(1.0)


def test_min_vol_weights_inversely_to_vol(returns_panel):
    cfg = PipelineConfig(allocator="min_vol_30d")
    # AGG has the lowest vol (0.004) so it should get the largest weight.
    w = allocate(returns_panel, ["IVV", "AGG", "GLD", "DBC"], cfg)
    assert np.isclose(w.sum(), 1.0)
    assert w["AGG"] > w["IVV"] > w["DBC"]


def test_momentum_falls_back_to_ew_when_all_negative(returns_panel):
    # Force all negative momentum by injecting a synthetic crash window.
    crash = returns_panel.copy()
    crash.iloc[-30:] = -0.02  # everyone losing
    cfg = PipelineConfig(allocator="momentum_30d")
    w = allocate(crash, ["IVV", "AGG", "GLD", "DBC"], cfg)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w.values, 0.25)


def test_hrp_produces_positive_weights_summing_to_one(returns_panel):
    cfg = PipelineConfig(allocator="hrp", hrp_lookback_years=4)
    w = allocate(returns_panel, ["IVV", "AGG", "GLD", "DBC"], cfg)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= 0).all()
    assert len(w) == 4


def test_unknown_selected_symbol_raises(returns_panel):
    cfg = PipelineConfig(allocator="ew")
    with pytest.raises(KeyError, match="missing columns"):
        allocate(returns_panel, ["IVV", "ZZZZ"], cfg)
