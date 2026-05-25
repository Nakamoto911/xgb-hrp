"""Per-asset regime panel — Module 3 of SPEC.md.

Output: long-form parquet ``[symbol, asset_name, date, regime_label]``
where ``regime_label`` comes from the JM's ``JM_Target_State`` column
(0 = Bull, 1 = Bear) without the +1-day shift applied to the XGB forecast.

Internally we share the JM+XGB walk-forward with :mod:`pipeline.forecast`
via :mod:`pipeline._walk_forward`, so running both phases on the same
config triggers a single fit per asset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline._walk_forward import POOL_TO_XGB_UNIVERSE, compute_signals
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeOutput:
    panel: pd.DataFrame  # [symbol, asset_name, date, regime_label]
    path: Optional[Path] = None


def _project_to_regime_panel(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, df in signals.items():
        if "JM_Target_State" not in df.columns:
            logger.warning("Missing JM_Target_State for %s — skipping", symbol)
            continue
        asset_name = df.attrs.get("asset_name", symbol)
        long = pd.DataFrame(
            {
                "symbol": symbol,
                "asset_name": asset_name,
                "date": df.index,
                "regime_label": df["JM_Target_State"]
                .map({0: "Bull", 1: "Bear"})
                .to_numpy(),
            }
        )
        rows.append(long)
    if not rows:
        raise RuntimeError("No JM regime labels available in any signal frame.")
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])


def build_regimes(
    config: PipelineConfig,
    *,
    save: bool = True,
    force_refresh: bool = False,
) -> RegimeOutput:
    """Build the per-asset regime panel.

    Reuses the cached walk-forward output when present (from
    :func:`pipeline._walk_forward.compute_signals`), so calling
    :func:`build_regimes` and :func:`pipeline.forecast.build_forecasts`
    consecutively triggers a single JM+XGB fit per asset.
    """
    if config.pool not in POOL_TO_XGB_UNIVERSE:
        raise NotImplementedError(
            f"Regime build for pool={config.pool!r} not wired. "
            f"Supported pools: {list(POOL_TO_XGB_UNIVERSE)}."
        )

    end_str = config.end_date or date.today().isoformat()
    out_path = (
        config.cache_dir
        / f"regimes_{config.pool}_{config.start_date}_{end_str}_{config.feature_version}.parquet"
    )

    if save and out_path.exists() and not force_refresh:
        logger.info("Loading cached regime panel from %s", out_path)
        return RegimeOutput(panel=pd.read_parquet(out_path), path=out_path)

    signals = compute_signals(config, force_refresh=force_refresh)
    panel = _project_to_regime_panel(signals)

    out_path_resolved: Optional[Path] = None
    if save:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)
        out_path_resolved = out_path
        logger.info("Saved regime panel to %s — shape %s", out_path, panel.shape)
    return RegimeOutput(panel=panel, path=out_path_resolved)


__all__ = ["RegimeOutput", "build_regimes"]
