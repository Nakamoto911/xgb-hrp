"""Regime-quality metric tests — pure layer only, network-free.

The render layer is exercised manually via the dashboard; everything here
runs on hand-built series with known answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.regime_quality import (
    append_vintage,
    detect_drawdown_events,
    event_aligned_matrix,
    extract_spells,
    next_day_spread,
    regime_feature_frame,
    regime_w2,
    revision_stats,
    signal_lags,
    spell_stats,
    survival_curve,
    w2_gaussian_2d,
)


def _labels(seq: str, start: str = "2020-01-01") -> pd.Series:
    """'BBbb' → Bear,Bear,Bull,Bull on consecutive business days."""
    idx = pd.bdate_range(start, periods=len(seq))
    return pd.Series(["Bear" if c == "B" else "Bull" for c in seq], index=idx)


# -----------------------------------------------------------------------------
# Spell extraction
# -----------------------------------------------------------------------------
def test_extract_spells_lengths_and_censoring():
    spells = extract_spells(_labels("bbbBBbbbbb"))
    assert spells["label"].tolist() == ["Bull", "Bear", "Bull"]
    assert spells["length"].tolist() == [3, 2, 5]
    # First and last spells touch the sample edges.
    assert spells["censored"].tolist() == [True, False, True]


def test_extract_spells_leading_nan():
    labels = _labels("bbBB")
    labels.iloc[0] = np.nan
    spells = extract_spells(labels)
    assert spells["length"].tolist() == [1, 2]


def test_extract_spells_single_spell_is_censored():
    spells = extract_spells(_labels("bbbb"))
    assert len(spells) == 1
    assert bool(spells["censored"].iloc[0])


def test_spell_stats_basic():
    spells = extract_spells(_labels("bbbbbbbbBBbbbbbbbbbb"))
    stats = spell_stats(spells, n_days=20)
    assert stats["n_spells"] == 3
    assert stats["n_censored"] == 2
    # 2 switches over 20 days → 2 * 252 / 20.
    assert stats["switches_per_year"] == pytest.approx(2 * 252 / 20)
    # The 2-day Bear spell is a flicker.
    assert stats["flicker_rate"] == pytest.approx(1 / 3)


# -----------------------------------------------------------------------------
# Survival curve
# -----------------------------------------------------------------------------
def test_survival_uncensored_matches_empirical():
    lengths = np.array([1, 2, 3, 4])
    censored = np.zeros(4, dtype=bool)
    surv = survival_curve(lengths, censored)
    expected = [(lengths > t).mean() for t in surv.index]
    assert np.allclose(surv.to_numpy(), expected)


def test_survival_censored_kaplan_meier():
    # Censored 2-day spell stays at risk through day 2, drops without event.
    surv = survival_curve(np.array([2, 3, 5]), np.array([True, False, False]))
    assert surv[1] == pytest.approx(1.0)
    assert surv[2] == pytest.approx(1.0)
    assert surv[3] == pytest.approx(0.5)   # at-risk {3,5}, one event
    assert surv[4] == pytest.approx(0.5)
    assert surv[5] == pytest.approx(0.0)


def test_survival_monotone_nonincreasing():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 30, size=50)
    censored = rng.random(50) < 0.3
    surv = survival_curve(lengths, censored)
    assert (np.diff(surv.to_numpy()) <= 1e-12).all()


# -----------------------------------------------------------------------------
# Wasserstein-2
# -----------------------------------------------------------------------------
def test_w2_identical_is_zero():
    mu = np.array([0.3, -0.2])
    cov = np.array([[1.0, 0.4], [0.4, 2.0]])
    assert w2_gaussian_2d(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-9)


def test_w2_pure_mean_shift_equals_distance():
    cov = np.eye(2)
    mu1, mu2 = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert w2_gaussian_2d(mu1, cov, mu2, cov) == pytest.approx(5.0)


def test_w2_matches_scipy_sqrtm_oracle():
    from scipy.linalg import sqrtm

    rng = np.random.default_rng(7)
    for _ in range(10):
        a = rng.normal(size=(2, 2))
        b = rng.normal(size=(2, 2))
        c1 = a @ a.T + 0.1 * np.eye(2)
        c2 = b @ b.T + 0.1 * np.eye(2)
        mu1, mu2 = rng.normal(size=2), rng.normal(size=2)
        s1 = np.real(sqrtm(c1))
        cross = np.real(sqrtm(s1 @ c2 @ s1))
        expected = np.sqrt(
            np.sum((mu1 - mu2) ** 2)
            + np.trace(c1) + np.trace(c2) - 2 * np.trace(cross)
        )
        assert w2_gaussian_2d(mu1, c1, mu2, c2) == pytest.approx(expected, rel=1e-9)


def test_regime_w2_needs_min_obs():
    idx = pd.bdate_range("2020-01-01", periods=30)
    feats = pd.DataFrame(
        {"ret": np.zeros(30), "vol": np.zeros(30), "label": ["Bull"] * 25 + ["Bear"] * 5},
        index=idx,
    )
    assert np.isnan(regime_w2(feats, min_obs=10))


def test_regime_feature_frame_zscored():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(1)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=idx)
    labels = pd.Series(np.where(np.arange(200) < 100, "Bull", "Bear"), index=idx)
    feats = regime_feature_frame(px, labels)
    assert list(feats.columns) == ["ret", "vol", "label"]
    assert feats["ret"].mean() == pytest.approx(0.0, abs=1e-9)
    assert feats["ret"].std() == pytest.approx(1.0, rel=1e-6)
    assert not feats.isna().any().any()


# -----------------------------------------------------------------------------
# Next-day spread (no look-ahead)
# -----------------------------------------------------------------------------
def test_next_day_spread_no_lookahead():
    # r_{t+1} = +1% iff signal_t is Bull, −1% iff Bear → spread strongly
    # positive only when conditioning is day-t signal → day-t+1 return.
    rng = np.random.default_rng(2)
    n = 500
    idx = pd.bdate_range("2015-01-01", periods=n)
    bear = pd.Series(rng.random(n) < 0.4, index=idx)
    rets = np.where(bear.to_numpy()[:-1], -0.01, 0.01)
    px = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + rets)]), index=idx)
    spread, bull_ann, bear_ann = next_day_spread(px, bear)
    assert bull_ann == pytest.approx(0.01 * 252, rel=1e-6)
    assert bear_ann == pytest.approx(-0.01 * 252, rel=1e-6)
    assert spread == pytest.approx(0.02 * 252, rel=1e-6)


def test_next_day_spread_same_day_correlation_is_flat():
    # Signal that mirrors the SAME day's return carries no next-day info →
    # spread ≈ 0. Catches an accidental off-by-one in the alignment.
    rng = np.random.default_rng(3)
    n = 5000
    idx = pd.bdate_range("2005-01-01", periods=n)
    rets = rng.choice([-0.01, 0.01], size=n - 1)
    px = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + rets)]), index=idx)
    bear = pd.Series(np.concatenate([[False], rets < 0]), index=idx)
    spread, _, _ = next_day_spread(px, bear)
    assert abs(spread) < 0.25  # ≈0 vs the 5.04 a one-day leak would produce


def test_next_day_spread_degenerate_regime_is_nan():
    idx = pd.bdate_range("2020-01-01", periods=50)
    px = pd.Series(np.linspace(100, 110, 50), index=idx)
    always_bull = pd.Series(False, index=idx)
    spread, _, bear_ann = next_day_spread(px, always_bull)
    assert np.isnan(spread) and np.isnan(bear_ann)


# -----------------------------------------------------------------------------
# Drawdown events
# -----------------------------------------------------------------------------
def _crash_series() -> pd.Series:
    """100 → 110 (day 9), down to 77 (day 19), recovered ≥110 at day 29."""
    up1 = np.linspace(100, 110, 10)
    down = np.linspace(110, 77, 11)[1:]
    up2 = np.linspace(77, 115, 11)[1:]
    px = np.concatenate([up1, down, up2])
    return pd.Series(px, index=pd.bdate_range("2020-01-01", periods=len(px)))


def test_detect_drawdown_events_synthetic():
    px = _crash_series()
    events = detect_drawdown_events(px, threshold=0.15)
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["peak_date"] == px.index[9]
    assert ev["trough_date"] == px.index[19]
    assert ev["depth"] == pytest.approx(77 / 110 - 1)
    # Recovery = first new high after the trough.
    assert px[ev["recovery_date"]] >= 110


def test_detect_drawdown_events_unrecovered_tail():
    px = _crash_series().iloc[:20]  # ends at the trough
    events = detect_drawdown_events(px, threshold=0.15)
    assert len(events) == 1
    assert pd.isna(events.iloc[0]["recovery_date"])


def test_detect_drawdown_events_below_threshold():
    px = _crash_series()
    assert detect_drawdown_events(px, threshold=0.40).empty


# -----------------------------------------------------------------------------
# Signal lags
# -----------------------------------------------------------------------------
def test_signal_lags_known_offsets():
    px = _crash_series()
    events = detect_drawdown_events(px, threshold=0.15)
    # Bear flag turns on 4 trading days after the peak, off 6 after the trough.
    flag = pd.Series(False, index=px.index)
    flag.iloc[13:26] = True
    out = signal_lags(events, flag, px)
    assert out["crash_lag"].iloc[0] == pytest.approx(4)
    assert out["recovery_lag"].iloc[0] == pytest.approx(7)  # first False at 26
    assert out["missed_exit_ret"].iloc[0] == pytest.approx(px.iloc[13] / px.iloc[9] - 1)
    assert out["missed_rebound_ret"].iloc[0] == pytest.approx(px.iloc[26] / px.iloc[19] - 1)


def test_signal_lags_already_bear_at_peak_is_zero():
    px = _crash_series()
    events = detect_drawdown_events(px, threshold=0.15)
    flag = pd.Series(True, index=px.index)
    flag.iloc[25:] = False
    out = signal_lags(events, flag, px)
    assert out["crash_lag"].iloc[0] == pytest.approx(0)


def test_signal_lags_never_fires_is_nan():
    px = _crash_series()
    events = detect_drawdown_events(px, threshold=0.15)
    flag = pd.Series(False, index=px.index)
    out = signal_lags(events, flag, px)
    assert np.isnan(out["crash_lag"].iloc[0])
    # A missed crash can't score an instant re-entry either.
    assert np.isnan(out["recovery_lag"].iloc[0])


# -----------------------------------------------------------------------------
# Event alignment
# -----------------------------------------------------------------------------
def test_event_aligned_matrix_padding_and_shape():
    idx = pd.bdate_range("2020-01-01", periods=100)
    series = pd.Series(np.arange(100, dtype=float), index=idx)
    mat = event_aligned_matrix([idx[10], idx[95]], series, window=(-30, 60))
    assert mat.shape == (2, 91)
    # Anchor at position 10: offsets −30..−11 fall before the sample → NaN.
    assert mat.iloc[0].isna().sum() == 20
    assert mat.iloc[0][0] == pytest.approx(10.0)
    assert mat.iloc[0][-10] == pytest.approx(0.0)
    # Anchor at position 95: offsets +5..+60 fall after the sample → NaN.
    assert mat.iloc[1].isna().sum() == 56
    assert mat.iloc[1][0] == pytest.approx(95.0)


def test_event_aligned_matrix_anchor_past_sample_dropped():
    idx = pd.bdate_range("2020-01-01", periods=10)
    series = pd.Series(np.arange(10, dtype=float), index=idx)
    mat = event_aligned_matrix([idx[-1] + pd.Timedelta(days=30)], series)
    assert mat.empty


# -----------------------------------------------------------------------------
# Vintages
# -----------------------------------------------------------------------------
def _panel(labels: dict[str, str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for symbol, seq in labels.items():
        for d, c in zip(dates, seq, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "asset_name": symbol,
                    "date": d,
                    "regime_label": "Bear" if c == "B" else "Bull",
                }
            )
    return pd.DataFrame(rows)


def test_append_vintage_dedupes_by_run_date(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=4)
    panel = _panel({"IVV": "bbBB"}, dates)
    path = tmp_path / "vintages.parquet"
    v1 = append_vintage(panel, path, "2026-06-10")
    assert len(v1) == 4
    # Same run_date again → no-op, even with different labels.
    v2 = append_vintage(_panel({"IVV": "BBBB"}, dates), path, "2026-06-10")
    assert len(v2) == 4
    assert (v2["regime_label"] == v1["regime_label"]).all()
    v3 = append_vintage(panel, path, "2026-06-11")
    assert sorted(v3["run_date"].unique()) == ["2026-06-10", "2026-06-11"]


def test_revision_stats_counts_changed_labels(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=4)
    path = tmp_path / "vintages.parquet"
    append_vintage(_panel({"IVV": "bbBB", "AGG": "bbbb"}, dates), path, "2026-06-10")
    # Next run flips one of IVV's four labels; AGG unchanged.
    vintages = append_vintage(
        _panel({"IVV": "bBBB", "AGG": "bbbb"}, dates), path, "2026-06-11"
    )
    stats = revision_stats(vintages)
    by_symbol = stats.set_index("symbol")
    assert by_symbol.loc["IVV", "pct_revised"] == pytest.approx(0.25)
    assert by_symbol.loc["AGG", "pct_revised"] == pytest.approx(0.0)
    assert (stats["n_overlap"] == 4).all()
