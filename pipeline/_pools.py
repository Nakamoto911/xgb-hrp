"""Pool → asset-spec tables.

For the ETF/MF pools we delegate to vendor/xgboost's YAHOO_ASSETS and
MUTUAL_FUNDS_ASSETS. For the European pool we maintain our own table
since vendor has no European universe — class-based half-life mappings
mirror the vendor's convention:

    hl=8  broad equity, REIT, gov bond, dev. eq smart-beta
    hl=4  commodity (oil/copper/agg) + gold
    hl=2  investment-grade corporate
    hl=0  emerging markets, high yield, crypto

DD-exclusion follows the same logic: bonds and gold drop the drawdown
features (their drawdowns are too shallow / regime-uninformative).
"""
from __future__ import annotations

# (asset_name, ticker, hl_proxy, include_dd)
EUROPEAN_ASSETS: list[tuple[str, str, str, bool]] = [
    # Core developed equity
    ("EU_LargeCap_US",   "SXR8.DE",  "SXR8.DE", True),   # iShares Core S&P 500 UCITS
    ("EU_DevWorld",      "SXRW.DE",  "SXRW.DE", True),   # iShares MSCI World
    ("EU_ACWI",          "SXRT.DE",  "SXRT.DE", True),   # iShares MSCI ACWI
    ("EU_USA",           "IUSM.DE",  "IUSM.DE", True),   # iShares MSCI USA
    # Emerging markets
    ("EU_EM",            "SXRZ.DE",  "SXRZ.DE", False),  # iShares MSCI EM
    ("EU_EM_Asia",       "IS3N.DE",  "IS3N.DE", False),  # iShares MSCI EM Asia
    # Regional equity
    ("EU_Japan",         "XJSE.DE",  "XJSE.DE", True),   # Xtrackers MSCI Japan
    # Bonds
    ("EU_T_Bill",        "IS0L.DE",  "IS0L.DE", False),  # iShares overnight rate
    ("EU_UK_Gilt",       "IGLT.MI",  "IGLT.MI", False),  # iShares UK Gilts
    ("EU_Corporate",     "IBCQ.DE",  "IBCQ.DE", False),  # iShares Euro Corp
    ("EU_HighYield",     "IHYG.MI",  "IHYG.MI", False),  # iShares Euro HY
    # Commodities / gold
    ("EU_Gold",          "4GLD.DE",  "4GLD.DE", False),  # Xetra-Gold
    ("EU_Oil",           "CRUD.MI",  "CRUD.MI", True),   # WisdomTree Brent
    ("EU_Copper",        "COPA.MI",  "COPA.MI", True),   # WisdomTree Copper
    ("EU_Commodity",     "AIGP.MI",  "AIGP.MI", True),   # WisdomTree All-Commodity
    # Smart-beta
    ("EU_SmallCap",      "IS3S.DE",  "IS3S.DE", True),
    ("EU_Quality",       "IS3R.DE",  "IS3R.DE", True),
    ("EU_Momentum",      "IS3Q.DE",  "IS3Q.DE", True),
    ("EU_MinVol",        "IQQ0.DE",  "IQQ0.DE", True),
    # Crypto
    ("EU_Bitcoin",       "BTCE.DE",  "BTCE.DE", False),
]


def european_start_date() -> str:
    """Most European UCITS ETFs above have inception ~2010-2012; BTCE 2020.

    Wide enough to give the 11-year JM lookback room for assets that
    inceptioned around 2010; later-inception assets are handled by the
    vendor walk-forward's partial-window logic.
    """
    return "1999-01-01"
