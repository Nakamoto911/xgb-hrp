"""Regime-builder tests — schema contract + NotImplementedError gate.

End-to-end JM walk-forward is exercised via the CLI smoke test, not pytest
(it needs network access and ~30s of compute).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from pipeline._walk_forward import POOL_TO_XGB_UNIVERSE
from pipeline.config import PipelineConfig
from pipeline.regime import RegimeOutput, _project_to_regime_panel, build_regimes


def test_pool_to_xgb_universe_table_present():
    assert POOL_TO_XGB_UNIVERSE["etf"] == "yahoo"
    assert POOL_TO_XGB_UNIVERSE["mutual_fund"] == "yahoo_mutual"
    assert POOL_TO_XGB_UNIVERSE["european"] == "european"  # synthesized in pipeline._pools


def test_unsupported_pool_raises_not_implemented(tmp_path):
    # Validators forbid arbitrary pool strings, so we craft via direct construction.
    from pipeline._walk_forward import _resolve_universe
    with pytest.raises(NotImplementedError):
        _resolve_universe("bogus_pool")


def test_regime_output_is_frozen():
    out = RegimeOutput(panel=pd.DataFrame())
    with pytest.raises(FrozenInstanceError):
        out.panel = pd.DataFrame()  # type: ignore[misc]


def test_project_to_regime_panel_schema():
    idx = pd.date_range("2023-01-02", periods=4, freq="B")
    sig = pd.DataFrame(
        {"JM_Target_State": [0, 0, 1, 1]}, index=idx
    )
    sig.attrs["asset_name"] = "LargeCap"
    panel = _project_to_regime_panel({"IVV": sig})
    assert list(panel.columns) == ["symbol", "asset_name", "date", "regime_label"]
    assert (panel["symbol"] == "IVV").all()
    assert (panel["asset_name"] == "LargeCap").all()
    assert panel["regime_label"].tolist() == ["Bull", "Bull", "Bear", "Bear"]


def test_build_regimes_force_project_bypasses_panel_cache(tmp_cache_dir, monkeypatch):
    """force_project re-projects from walk-forward caches without forcing them."""
    import pipeline.regime as regime_mod

    idx = pd.date_range("2023-01-02", periods=4, freq="B")
    calls: list[bool] = []

    def fake_signals(states):
        def _fake(config, force_refresh=False):
            calls.append(force_refresh)
            sig = pd.DataFrame({"JM_Target_State": states}, index=idx)
            sig.attrs["asset_name"] = "LargeCap"
            return {"IVV": sig}
        return _fake

    cfg = PipelineConfig(
        cache_dir=tmp_cache_dir, start_date="2023-01-01", end_date="2023-01-10"
    )

    monkeypatch.setattr(regime_mod, "compute_signals", fake_signals([0, 0, 1, 1]))
    first = build_regimes(cfg)
    assert first.panel["regime_label"].tolist() == ["Bull", "Bull", "Bear", "Bear"]

    # New signals, no force flags → panel parquet cache wins.
    monkeypatch.setattr(regime_mod, "compute_signals", fake_signals([1, 1, 1, 1]))
    cached = build_regimes(cfg)
    assert cached.panel["regime_label"].tolist() == ["Bull", "Bull", "Bear", "Bear"]

    # force_project bypasses the panel cache but must NOT force the walk-forward.
    projected = build_regimes(cfg, force_project=True)
    assert projected.panel["regime_label"].tolist() == ["Bear"] * 4
    assert calls == [False, False]  # cached call never reached compute_signals
