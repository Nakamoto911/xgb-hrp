"""Vendor import shim.

Both source repos are checked out under vendor/ as git submodules and neither
ships an __init__.py. They were designed to be run as scripts from inside
their own directories (e.g. `import main as _main` in vendor/xgboost/portfolio.py).

This module is the single place that injects them into sys.path. Importing
anything from `pipeline._vendor` is enough to make `import main`,
`import portfolio`, and `import hrp_engine.<x>` resolve to the vendored code.

Call `vendor_xgboost()` / `vendor_hrp()` to get the loaded modules without
worrying about path order.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _REPO_ROOT / "vendor"
_XGB_DIR = _VENDOR / "xgboost"
_HRP_DIR = _VENDOR / "hrp"


def _ensure_on_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# Inject both vendor roots on import. xgboost's portfolio.py does
# `import main as _main`, which only resolves if vendor/xgboost is on sys.path.
_ensure_on_path(_XGB_DIR)
_ensure_on_path(_HRP_DIR)


def vendor_xgboost_main() -> ModuleType:
    """vendor/xgboost/main.py — StatisticalJumpModel, walk_forward_backtest, features."""
    import main  # type: ignore[import-not-found]
    return main


def vendor_xgboost_portfolio() -> ModuleType:
    """vendor/xgboost/portfolio.py — compute_asset_signals, AssetPanel, MVO helpers."""
    import portfolio  # type: ignore[import-not-found]
    return portfolio


def vendor_xgboost_config() -> ModuleType:
    """vendor/xgboost/config.py — StrategyConfig dataclass."""
    import config as xgb_config  # type: ignore[import-not-found]
    return xgb_config


def vendor_hrp_data() -> ModuleType:
    """vendor/hrp/hrp_engine/data.py — pool ticker lists + fetch_data + clean_prices."""
    from hrp_engine import data  # type: ignore[import-not-found]
    return data


def vendor_hrp_hrp() -> ModuleType:
    """vendor/hrp/hrp_engine/hrp.py — optimize_hrp + bisection helpers."""
    from hrp_engine import hrp  # type: ignore[import-not-found]
    return hrp


def vendor_hrp_backtest() -> ModuleType:
    """vendor/hrp/hrp_engine/backtest.py — drift-band executor + PFU tax accounting."""
    from hrp_engine import backtest  # type: ignore[import-not-found]
    return backtest


__all__ = [
    "vendor_xgboost_main",
    "vendor_xgboost_portfolio",
    "vendor_xgboost_config",
    "vendor_hrp_data",
    "vendor_hrp_hrp",
    "vendor_hrp_backtest",
]
