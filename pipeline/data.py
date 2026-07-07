"""Data layer — Module 2 of SPEC.md.

Responsibilities:
1. Resolve a pool key → ticker list (uses vendor/hrp pool constants).
2. Load daily prices for the pool with parallel per-ticker yfinance fetch +
   the hrp cleaning pipeline (flat-trim + 3-day median).
3. Cache as Parquet under cache/ keyed by {pool}_{start}_{end}_{feature_version}.

The vendored hrp.fetch_data does a single batch yfinance call and caches as
CSV with a hard-coded end date. We replace it with a parallel per-ticker
fetch and a parquet cache while reusing the cleaning function.

Per-pool feature engineering (return + macro features) lives in
``pipeline.regime`` for now — it's only needed there.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pipeline import _vendor  # noqa: F401 — populates sys.path for vendor imports
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pool → tickers
# -----------------------------------------------------------------------------
def get_universe_tickers(pool: str) -> list[str]:
    """Resolve a pool key to its ticker list. Source: vendor/hrp/hrp_engine/data.py."""
    hrp_data = _vendor.vendor_hrp_data()
    table = {
        "etf": hrp_data.ETF_POOL,
        "mutual_fund": hrp_data.MUTUAL_FUND_POOL,
        "european": hrp_data.EUROPEAN_POOL,
    }
    if pool not in table:
        raise ValueError(f"Unknown pool {pool!r}. Known: {list(table)}.")
    return list(table[pool])


# -----------------------------------------------------------------------------
# Parquet cache key
# -----------------------------------------------------------------------------
def _cache_path(cache_dir: Path, pool: str, start: str, end: str, feature_version: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"prices_{pool}_{start}_{end}_{feature_version}.parquet"


def _resolve_end(config: PipelineConfig) -> str:
    """Resolved end of the window: the configured end, else *today*.

    Blank/None always rolls forward to today's date, so the dashboard shows the
    latest session no matter when the user connects.
    """
    return config.end_date or date.today().isoformat()


def _fetch_end(end: str) -> str:
    """yfinance ``end`` is *exclusive*; add a day so the latest available bar
    (including today's, once the session opens) is returned regardless of the
    hour of day. The cache filename still keys on the inclusive ``end``."""
    return (date.fromisoformat(end) + timedelta(days=1)).isoformat()


# -----------------------------------------------------------------------------
# Parallel per-ticker yfinance fetch
# -----------------------------------------------------------------------------
def _fetch_one_ticker(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch Adj Close for one ticker. Empty Series on error (logged)."""
    import yfinance as yf

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,  # we parallelize externally
        )
    except Exception as e:  # network/auth/etc.
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return pd.Series(dtype=float, name=ticker)

    if df is None or df.empty:
        logger.warning("yfinance returned empty frame for %s", ticker)
        return pd.Series(dtype=float, name=ticker)

    if isinstance(df.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in df.columns.levels[0] else "Close"
        series = df[col].squeeze()
    else:
        series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    series.name = ticker
    return series


def _fetch_all_parallel(
    tickers: list[str], start: str, end: str, max_workers: int = 12
) -> pd.DataFrame:
    """ThreadPoolExecutor across tickers (yfinance is HTTP-I/O bound)."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one_ticker, t, start, end): t for t in tickers}
        series_list: list[pd.Series] = []
        for fut in as_completed(futs):
            s = fut.result()
            if not s.empty:
                series_list.append(s)
    if not series_list:
        raise RuntimeError(
            f"All ticker fetches failed for tickers={tickers}. "
            "Check network and yfinance version."
        )
    return pd.concat(series_list, axis=1).sort_index()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_prices(
    config: PipelineConfig,
    *,
    force_refresh: bool = False,
    max_workers: int = 12,
) -> pd.DataFrame:
    """Load cleaned price panel for the configured pool.

    Reads from a Parquet cache if present; otherwise fetches in parallel via
    yfinance, applies the hrp cleaning pipeline, and persists.
    """
    pool = config.pool
    start = config.start_date
    end = _resolve_end(config)
    cache_path = _cache_path(config.cache_dir, pool, start, end, config.feature_version)

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached prices from %s", cache_path)
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df

    tickers = get_universe_tickers(pool)
    logger.info(
        "Fetching %d tickers for pool=%s window=[%s, %s]", len(tickers), pool, start, end
    )
    df_raw = _fetch_all_parallel(tickers, start, _fetch_end(end), max_workers=max_workers)

    # Reorder columns to match the pool list, dropping any that failed.
    df_raw = df_raw.reindex(columns=[t for t in tickers if t in df_raw.columns])

    hrp_data = _vendor.vendor_hrp_data()
    df_clean = hrp_data.clean_prices(df_raw)

    df_clean.to_parquet(cache_path)
    logger.info("Saved %s — shape %s", cache_path, df_clean.shape)
    return df_clean


def load_benchmarks(config: PipelineConfig, *, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the two benchmark series (B&H + 60/40) per Section 3.3."""
    tickers = [config.benchmark_bh, config.benchmark_6040]
    tickers = [t for t in tickers if t]  # drop None
    if not tickers:
        return pd.DataFrame()

    start = config.start_date
    end = _resolve_end(config)
    cache_path = _cache_path(
        config.cache_dir, f"{config.pool}_bench", start, end, config.feature_version
    )
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = _fetch_all_parallel(tickers, start, _fetch_end(end), max_workers=len(tickers))
    df.to_parquet(cache_path)
    return df


def load_risk_free(config: PipelineConfig, *, force_refresh: bool = False) -> pd.Series:
    """Fetch the risk-free series for the pool (e.g. BIL for the ETF pool)."""
    ticker = config.risk_free_asset
    if not ticker:
        raise ValueError(f"No risk_free_asset resolved for pool={config.pool!r}.")
    start = config.start_date
    end = _resolve_end(config)
    cache_path = _cache_path(
        config.cache_dir, f"{config.pool}_rf", start, end, config.feature_version
    )
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)[ticker]

    df = _fetch_all_parallel([ticker], start, _fetch_end(end), max_workers=1)
    df.to_parquet(cache_path)
    return df[ticker]


__all__ = [
    "get_universe_tickers",
    "load_prices",
    "load_benchmarks",
    "load_risk_free",
]
