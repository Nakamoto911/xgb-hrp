"""Regime-builder tests — schema contract + NotImplementedError gate.

End-to-end JM walk-forward is exercised via the CLI smoke test, not pytest
(it needs network access and ~30s of compute).
"""
from __future__ import annotations

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
    with pytest.raises(Exception):  # FrozenInstanceError
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
