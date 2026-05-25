"""Shared pytest fixtures. Network-free."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Tiny synthetic price panel reusing hrp's generator.

    3 tickers, ~6 months of business days. Enough to exercise schema
    and cleaning but too small for a real JM fit.
    """
    from pipeline import _vendor

    hrp_data = _vendor.vendor_hrp_data()
    return hrp_data.generate_synthetic_prices(
        tickers=["IVV", "AGG", "GLD"],
        start_date="2024-01-02",
        end_date="2024-06-30",
    )
