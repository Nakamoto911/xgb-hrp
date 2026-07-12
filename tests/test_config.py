"""PipelineConfig (SPEC.md §14) — schema, defaults, validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.config import POOL_DEFAULTS, PipelineConfig


def test_defaults_match_etf_pool():
    cfg = PipelineConfig()
    assert cfg.pool == "etf"
    assert cfg.risk_free_asset == POOL_DEFAULTS["etf"]["risk_free_asset"]
    assert cfg.benchmark_bh == POOL_DEFAULTS["etf"]["benchmark_bh"]
    assert cfg.benchmark_6040 == POOL_DEFAULTS["etf"]["benchmark_6040"]
    assert cfg.allocator == "hrp"
    assert cfg.rebalance_frequency == "quarterly"
    assert cfg.reenter_mode == "immediate_fresh"  # v1.1 default


@pytest.mark.parametrize("pool", ["etf", "mutual_fund", "european"])
def test_pool_default_resolution(pool):
    cfg = PipelineConfig(pool=pool)
    assert cfg.risk_free_asset == POOL_DEFAULTS[pool]["risk_free_asset"]
    assert cfg.benchmark_bh == POOL_DEFAULTS[pool]["benchmark_bh"]


def test_user_overrides_win_over_pool_defaults():
    cfg = PipelineConfig(pool="etf", risk_free_asset="SHY")
    assert cfg.risk_free_asset == "SHY"


def test_frozen_model_cannot_mutate():
    cfg = PipelineConfig()
    with pytest.raises(ValidationError):
        cfg.pool = "european"  # type: ignore[misc]


def test_hysteresis_validator_blocks_inverted_thresholds():
    with pytest.raises(ValidationError, match="Hysteresis"):
        PipelineConfig(universe_pct_threshold=0.30, universe_pct_clear_threshold=0.40)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig(nonexistent_field=True)  # type: ignore[call-arg]


def test_bull_prob_threshold_bounds():
    PipelineConfig(bull_prob_threshold=0.01)
    PipelineConfig(bull_prob_threshold=0.99)
    with pytest.raises(ValidationError):
        PipelineConfig(bull_prob_threshold=0.0)
    with pytest.raises(ValidationError):
        PipelineConfig(bull_prob_threshold=1.0)


def test_bull_prob_threshold_default_is_040():
    assert PipelineConfig().bull_prob_threshold == 0.40


@pytest.mark.parametrize("method", ["ma200", "hybrid"])
def test_price_aware_forecast_methods_accepted(method):
    cfg = PipelineConfig(forecast_method=method)
    assert cfg.forecast_method == method


def test_ma_window_default_and_bounds():
    assert PipelineConfig().ma_window == 200
    PipelineConfig(ma_window=2)
    with pytest.raises(ValidationError):
        PipelineConfig(ma_window=1)


def test_hybrid_bear_threshold_default_and_bounds():
    assert PipelineConfig().hybrid_bear_threshold == 0.80
    PipelineConfig(hybrid_bear_threshold=0.01)
    PipelineConfig(hybrid_bear_threshold=0.99)
    with pytest.raises(ValidationError):
        PipelineConfig(hybrid_bear_threshold=0.0)
    with pytest.raises(ValidationError):
        PipelineConfig(hybrid_bear_threshold=1.0)


def test_lambda_grid_paper_default():
    cfg = PipelineConfig()
    assert cfg.jm_lambda_grid == (4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0)


def test_round_trip_via_dict():
    cfg = PipelineConfig(pool="mutual_fund", allocator="ew")
    d = cfg.to_dict()
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.pool == "mutual_fund"
    assert cfg2.allocator == "ew"
