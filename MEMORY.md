# Project research log — xgb-hrp

Durable findings and decisions that aren't recoverable from the code or git
history. Newest first. Keep entries short; link to the scripts/commits that
produced them.

---

## 2026-07-07 — Production regime rule loses to MA200 (per-asset, net)

**Setup.** ETF pool, 2007-01-01 → 2026-07-06, warm caches. Each detector
simulated as a *tradable long/flat strategy* per asset: hold the asset when the
detector says bull, earn the daily risk-free rate when it says bear, charge
`transaction_cost_bps` (5 bps) per flip and `pfu_rate` (31.4%) tax on realized
gains at each exit. This is the "Detector shootout" table in the Regime Cockpit
(commit `fa897d6`). Analysis scripts (session scratchpad, not in repo):
`prod_vs_ma200.py`, `hybrid_test.py`.

**Result — equal-weight net CAGR across the 12 assets:**

| Detector | Net CAGR (EW) | Sortino | Avg MDD | Beats MA200 |
|---|---|---|---|---|
| Production (θ=0.40 on smoothed p_bear) | 3.7% | 0.36 | −33% | 0/12 |
| Best-tuned production family (θ=0.60, min-len 10) | 4.3% | 0.45 | −34% | **0/12** |
| Buy & hold | 5.2% | 0.48 | −48% | 0/12 |
| Hybrid: `price < MA200 OR p_bear > 0.80` | 7.6% | 0.92 | **−16.4%** | 1/12 |
| **MA200 alone** | **8.8%** | **1.06** | −17.8% | — |

Production loses to MA200 on **every one of the 12 assets**. A 324-config grid
over the whole production rule family (θ × θ_clear × dwell × min-length) never
beats MA200 on a single asset — so the gap is **structural, not a tuning
problem**.

**Why.** The p_bear signal classifies *volatility regimes*; a long/flat
strategy is paid for *price direction*. Double-sided losses vs MA200 on IVV:
- 338 days out while MA200 was in → +124% cumulative upside missed. Recurs
  almost every year (sits out vol-heavy rebounds: 41 days into the 2012
  recovery, 32 into the 2020 rebound, big misses again in 2023).
- 181 days in while MA200 was out → −49% absorbed (Dec-2018, Aug-2011,
  Jan-2016, Dec-2022 selloffs the trend rule had already left).
- Only **14%** of IVV bear calls saw price actually fall.

Exit *speed* is fine (production 15d median crash lag ≈ MA200's 13d) — the
problem is being defensive during up-markets, not being slow to crashes.

**Next steps (agreed, in order):**
1. Quick win (does **not** close the gap): sidebar bull-prob threshold
   0.60 → 0.40 (θ_bear 0.40 → 0.60). +0.6pp EW net CAGR, stable in both
   history halves.
2. **Decisive test:** add `ma200` and the hybrid rule (`price < MA200 OR
   p_bear > 0.80`) as `forecast_method` options in `pipeline/forecast.py`
   (`apply_rule`), then run the **full portfolio backtest** (allocator +
   executor + taxes). The per-asset verdict may change at the portfolio level,
   where p_bear also drives the risk monitor and allocation.
3. **Root-cause fix:** the training target. JM labels reward vol-regime
   separation, not tradable timing. Add price-location features (distance to
   200d MA, 12-month momentum) to the XGB, or move to a directional label.

**Caveats.** Single historical sample; MA200 window is the canonical default,
not tuned; a per-asset long/flat arena inherently favors price-level rules; the
cockpit's MA200 "bear win rate" (97%) is partly a convention artifact (the
switch-day return is assigned to the new regime, which flatters price-triggered
rules).
