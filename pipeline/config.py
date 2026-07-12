"""PipelineConfig — single source of truth for run-time options.

Mirrors SPEC.md Section 14. The Streamlit UI binds 1:1 to this model.

All numerical defaults match the source repos (vendor/hrp, vendor/xgboost) or
the paper they reproduce; deviations are flagged in the field description.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -----------------------------------------------------------------------------
# Per-pool defaults (Section 3.1 / 3.2 / 3.3 of SPEC.md)
# -----------------------------------------------------------------------------
# Risk-free leg and benchmarks per pool. The European-pool defaults are
# placeholders — they need user validation before running that pool live, since
# the spec gives a "EUR T-bill ETF / UCITS 60-40" description without a
# canonical ticker. Override via config when running the european pool.
POOL_DEFAULTS: dict[str, dict[str, str]] = {
    "etf": {
        "risk_free_asset": "BIL",
        "benchmark_bh": "^GSPC",
        "benchmark_6040": "VSMGX",
    },
    "mutual_fund": {
        "risk_free_asset": "VBMFX",
        "benchmark_bh": "^SP500TR",
        "benchmark_6040": "VSMGX",
    },
    "european": {
        "risk_free_asset": "IBTE.L",   # placeholder — iShares EUR Govt 0-1y UCITS
        "benchmark_bh": "^STOXX",
        "benchmark_6040": "V60A.L",    # placeholder — Vanguard LifeStrategy 60 UCITS
    },
}


class PipelineConfig(BaseModel):
    """End-to-end run configuration. Frozen and validated; mutate via model_copy."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    # ---- Pool ---------------------------------------------------------------
    pool: Literal["etf", "mutual_fund", "european"] = "etf"
    risk_free_asset: str | None = Field(
        default=None,
        description="Resolved from POOL_DEFAULTS if omitted.",
    )
    benchmark_bh: str | None = Field(
        default=None,
        description="Buy-and-hold benchmark ticker. Resolved from POOL_DEFAULTS if omitted.",
    )
    benchmark_6040: str | None = Field(
        default=None,
        description="60/40 lifestyle benchmark ticker. Resolved from POOL_DEFAULTS if omitted.",
    )

    # ---- Regime (Module 3) --------------------------------------------------
    jm_lambda_grid: tuple[float, ...] = (4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0)
    jm_lookback_years: int = Field(default=11, ge=1, le=30)
    jm_n_states: Literal[2] = 2
    jm_max_iter: int = 1000
    jm_tol: float = 1e-8
    jm_n_init: int = 10

    # ---- Forecast (Module 4) ------------------------------------------------
    forecast_method: Literal[
        "prob_threshold",
        "regime_and_prob",
        "ewma_smoothed",
        "trend",
        "last_day_regime",
        "ma200",
        "hybrid",
    ] = "ewma_smoothed"
    bull_prob_threshold: float = Field(default=0.40, gt=0.0, lt=1.0)
    trend_window: int = Field(default=5, ge=2)
    ma_window: int = Field(default=200, ge=2)  # SMA window (days) for the ma200 / hybrid rules
    # Per-asset hybrid-rule gate on smoothed p_bear — distinct from bear_prob_threshold
    # below (portfolio-level, Module 8 risk monitor).
    hybrid_bear_threshold: float = Field(default=0.80, gt=0.0, lt=1.0)
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.3
    xgb_n_estimators: int = 100
    xgb_smoothing_halflife: int = 8  # EWMA half-life on P_bull, paper default

    # ---- Allocator (Module 6) -----------------------------------------------
    allocator: Literal["hrp", "ew", "momentum_30d", "min_vol_30d"] = "hrp"
    hrp_lookback_years: int = Field(default=4, ge=1)
    hrp_linkage: Literal["single", "complete", "ward"] = "single"
    hrp_bisection: Literal["tree", "index"] = "tree"
    hrp_denoise: bool = True

    # ---- Execution (Module 7) ----------------------------------------------
    rebalance_frequency: Literal[
        "daily", "weekly", "monthly", "quarterly", "semi-annually", "yearly"
    ] = "quarterly"
    drift_threshold: float = Field(default=0.015, ge=0.0, lt=1.0)
    transaction_cost_bps: float = Field(default=5.0, ge=0.0)
    pfu_rate: float = Field(default=0.314, ge=0.0, lt=1.0)
    loss_carryforward_years: int = Field(default=10, ge=0)

    # ---- Risk Monitor (Module 8) -------------------------------------------
    risk_monitor_enabled: bool = True
    # raw = unsmoothed daily P(bear) (`Raw_Prob`); smoothed = EWMA-smoothed
    # P(bear) (`State_Prob`, what the selection rules use). Default "raw"
    # preserves current behavior (real-data runs show raw whipsaws — 51
    # triggers in 19.5y — because it reacts to every daily XGB jitter).
    risk_monitor_signal: Literal["raw", "smoothed"] = "raw"
    bear_prob_threshold: float = Field(default=0.70, gt=0.0, lt=1.0)
    universe_pct_threshold: float = Field(default=0.40, gt=0.0, le=1.0)
    universe_pct_clear_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    risk_off_dwell_days: int = Field(default=5, ge=1)
    reenter_mode: Literal[
        "immediate_fresh", "immediate_last_targets", "next_rebalance"
    ] = "immediate_fresh"

    # ---- Backtest window ---------------------------------------------------
    start_date: str = "2007-01-01"
    end_date: str | None = None  # None → today at run time

    # ---- Infrastructure (not in spec §14, needed by §4/§13.2) --------------
    feature_version: str = Field(
        default="v1",
        description="Cache-key salt. Bump to invalidate parquet caches after a feature change.",
    )
    cache_dir: Path = Path("cache")
    n_jobs: int = -1  # joblib outer-level parallelism
    seed: int = 42  # reproducibility (Section 12.8 e2e check)

    # ------------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------------
    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_dates(cls, v: Any) -> str | None:
        """Normalize any parseable date to zero-padded ISO (YYYY-MM-DD).

        Keeps cache keys stable (``2026-06-9`` and ``2026-06-09`` must not
        produce two different parquet files) and lets the data layer compute an
        exclusive end with ``date.fromisoformat``. Blank end_date → None (the
        loaders resolve None to *today* at run time, so the window always rolls
        forward to the latest session).
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            return pd.Timestamp(s).date().isoformat()
        except Exception as exc:  # noqa: BLE001 — surface a clear config error
            raise ValueError(f"Unparseable date {v!r}: {exc}") from exc

    @model_validator(mode="before")
    @classmethod
    def _resolve_pool_defaults(cls, data: Any) -> Any:
        """Fill risk-free + benchmarks from POOL_DEFAULTS when the user omits them."""
        if not isinstance(data, dict):
            return data
        pool = data.get("pool", "etf")
        defaults = POOL_DEFAULTS.get(pool, {})
        for key in ("risk_free_asset", "benchmark_bh", "benchmark_6040"):
            if data.get(key) is None and key in defaults:
                data[key] = defaults[key]
        return data

    @model_validator(mode="after")
    def _check_hysteresis(self) -> PipelineConfig:
        if self.universe_pct_clear_threshold >= self.universe_pct_threshold:
            raise ValueError(
                f"Hysteresis broken: universe_pct_clear_threshold "
                f"({self.universe_pct_clear_threshold}) must be < "
                f"universe_pct_threshold ({self.universe_pct_threshold}). "
                f"Otherwise risk-off flip-flops."
            )
        return self

    # ------------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        import yaml  # local import: yaml is optional at runtime

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
