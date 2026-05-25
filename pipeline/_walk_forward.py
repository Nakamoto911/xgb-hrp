"""Shared per-asset JM+XGBoost walk-forward worker.

Runs ``walk_forward_backtest`` from vendor/xgboost with
``include_xgboost=True`` so the output contains both Module-3 outputs
(``JM_Target_State``) and Module-4 outputs (``Raw_Prob``, ``State_Prob``,
``Forecast_State``). Caches one parquet per (asset, window, feature_version)
so that regime.py and forecast.py can each derive their views without
re-running the JM/XGB fit.

Heavy work runs in joblib loky subprocesses; the vendor's per-process
``_main`` globals are set inside the worker for safety.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed

from pipeline import _vendor
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# Map our pool keys → xgboost universe keys. 'european' is special-cased:
# vendor has no european universe, so we synthesize asset specs from
# pipeline._pools.EUROPEAN_ASSETS and reuse vendor's yahoo feature builder
# (yfinance supports .DE/.MI tickers natively).
POOL_TO_XGB_UNIVERSE: dict[str, str] = {
    "etf": "yahoo",
    "mutual_fund": "yahoo_mutual",
    "european": "european",
}


def _resolve_universe(pool: str) -> str:
    if pool not in POOL_TO_XGB_UNIVERSE:
        raise NotImplementedError(
            f"Pool {pool!r} has no xgboost universe wired in. "
            f"Supported: {list(POOL_TO_XGB_UNIVERSE)}."
        )
    return POOL_TO_XGB_UNIVERSE[pool]


def _cache_path_one_asset(
    cache_dir: Path,
    pool: str,
    symbol: str,
    start: str,
    end: str,
    feature_version: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Some tickers contain '.' (BTCE.DE) or '^' (^SP500TR); sanitize.
    safe = symbol.replace(".", "_").replace("^", "_").replace("=", "_")
    return cache_dir / f"wf_{pool}_{safe}_{start}_{end}_{feature_version}.parquet"


def _spec_symbol(spec: tuple, universe: str) -> str:
    """Return the yfinance-style symbol from an xgboost asset spec."""
    # YAHOO_ASSETS / MUTUAL_FUNDS_ASSETS / EUROPEAN_ASSETS layout:
    # (asset_name, ticker, hl_proxy, include_dd) — ticker at index 1.
    if universe in ("yahoo", "yahoo_mutual", "european"):
        return spec[1]
    raise NotImplementedError(f"Universe {universe!r} symbol extraction not wired.")


def _run_one_asset(
    spec: tuple,
    universe: str,
    start_date: str,
    oos_start: str,
    oos_end: str,
    lambda_grid: tuple[float, ...],
    transaction_cost: float,
    cache_path: Path,
    force_refresh: bool,
) -> tuple[str, str, Optional[pd.DataFrame]]:
    """Walk-forward for one asset. Returns (asset_name, symbol, df_or_none).

    Reads from ``cache_path`` if it exists and ``force_refresh=False``.
    """
    asset_name = spec[0]
    symbol = _spec_symbol(spec, universe)

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return asset_name, symbol, df

    # Re-inject vendor sys.path inside the subprocess and clamp BLAS/OMP threads.
    from pipeline import _vendor as _v  # noqa: F401

    portfolio = _v.vendor_xgboost_portfolio()
    main = _v.vendor_xgboost_main()
    xgb_config = _v.vendor_xgboost_config()
    portfolio._limit_inner_threads()

    if universe in ("yahoo", "yahoo_mutual"):
        _, ticker, hl_proxy, include_dd = spec
        df_feat = portfolio._build_yahoo_features(
            ticker, start_date, oos_end, include_dd=include_dd
        )
    elif universe == "european":
        # Reuse vendor's yahoo feature pipeline (yfinance handles .DE/.MI),
        # but swap the risk-free leg before the call. Macro features (US yields,
        # ^VIX) stay US-based as a global-macro proxy — see SPEC.md §16 future
        # work for a EUR-native macro feature set.
        _, ticker, hl_proxy, include_dd = spec
        original_rf = main.RISK_FREE_TICKER
        try:
            main.RISK_FREE_TICKER = "^IRX"  # keep US rf for vendor's feature math
            df_feat = portfolio._build_yahoo_features(
                ticker, start_date, oos_end, include_dd=include_dd
            )
        finally:
            main.RISK_FREE_TICKER = original_rf
    else:
        raise NotImplementedError(f"Feature build for universe={universe!r}.")

    main.TARGET_TICKER = hl_proxy
    main._forecast_cache.clear()
    main.OOS_START_DATE = oos_start
    main.END_DATE = oos_end
    main.START_DATE_DATA = start_date
    main.LAMBDA_GRID = list(lambda_grid)
    main.TRANSACTION_COST = transaction_cost

    cfg = xgb_config.StrategyConfig(
        name=f"WF_{universe}_{asset_name}",
        ewma_mode="paper",
        include_xgboost=True,
    )
    try:
        result = main.walk_forward_backtest(df_feat, cfg)
    except Exception as e:
        logger.warning("walk_forward_backtest failed for %s: %s", asset_name, e)
        return asset_name, symbol, None
    if result is None or result.empty:
        return asset_name, symbol, None

    # Persist as parquet (drop .attrs since parquet doesn't preserve them).
    # The attrs we need (lambda_history, lambda_dates, ewma_halflife) get
    # stashed as scalar/list columns or returned via a sidecar JSON if a
    # future eval module needs them. For Phase 2 the result columns suffice.
    out = result.copy()
    out.attrs.clear()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    return asset_name, symbol, result


def compute_signals(
    config: PipelineConfig,
    *,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run per-asset walk-forward for every ticker in the pool.

    Returns a dict ``{symbol: result_df}`` where each result_df has at minimum
    ``Target_Return``, ``RF_Rate``, ``JM_Target_State``, ``Raw_Prob``,
    ``State_Prob``, ``Forecast_State``. Heavy work is parallelized via
    joblib(loky); per-asset parquet caches make subsequent calls instant.
    """
    universe = _resolve_universe(config.pool)
    portfolio = _vendor.vendor_xgboost_portfolio()
    if universe == "european":
        from pipeline._pools import EUROPEAN_ASSETS, european_start_date
        specs = EUROPEAN_ASSETS
        start_date = european_start_date()
    else:
        specs = portfolio._asset_specs_for(universe)
        start_date = portfolio._start_date_for(universe)
    end_str = config.end_date or date.today().isoformat()

    from pipeline.data import get_universe_tickers
    pool_tickers = set(get_universe_tickers(config.pool))
    specs_to_run = [s for s in specs if _spec_symbol(s, universe) in pool_tickers]
    skipped = [_spec_symbol(s, universe) for s in specs if _spec_symbol(s, universe) not in pool_tickers]
    if skipped:
        logger.warning("Skipping symbols not in hrp %s pool: %s", config.pool, skipped)
    if not specs_to_run:
        raise RuntimeError(
            f"No asset specs match the {config.pool!r} pool. "
            f"Universe={universe!r}, pool tickers={sorted(pool_tickers)}."
        )

    cache_paths = {
        spec[0]: _cache_path_one_asset(
            config.cache_dir, config.pool, _spec_symbol(spec, universe),
            config.start_date, end_str, config.feature_version,
        )
        for spec in specs_to_run
    }

    logger.info(
        "Walk-forward on %d assets (pool=%s, OOS=[%s, %s], n_jobs=%d, force_refresh=%s)",
        len(specs_to_run), config.pool, config.start_date, end_str,
        config.n_jobs, force_refresh,
    )

    results = Parallel(
        n_jobs=config.n_jobs,
        backend="loky",
        return_as="generator_unordered",
    )(
        delayed(_run_one_asset)(
            spec,
            universe,
            start_date,
            config.start_date,
            end_str,
            tuple(config.jm_lambda_grid),
            config.transaction_cost_bps / 1e4,
            cache_paths[spec[0]],
            force_refresh,
        )
        for spec in specs_to_run
    )

    out: dict[str, pd.DataFrame] = {}
    for asset_name, symbol, df in results:
        if df is None or df.empty:
            logger.warning("No walk-forward output for %s (%s)", asset_name, symbol)
            continue
        df = df.copy()
        df.attrs["asset_name"] = asset_name
        df.attrs["symbol"] = symbol
        out[symbol] = df

    if not out:
        raise RuntimeError("All per-asset walk-forward runs failed.")
    return out


__all__ = ["compute_signals", "POOL_TO_XGB_UNIVERSE"]
