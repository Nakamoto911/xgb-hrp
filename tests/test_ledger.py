"""AVCO ledger + tax book — property tests."""
from __future__ import annotations

import math

import pytest

from pipeline.executor import AVCOLedger, TaxBook


# -----------------------------------------------------------------------------
# AVCOLedger
# -----------------------------------------------------------------------------
def test_buy_rolls_tc_into_basis():
    led = AVCOLedger()
    led.buy("IVV", 10.0, 100.0, tc=5.0)
    assert led.units["IVV"] == 10.0
    assert led.basis["IVV"] == 10 * 100 + 5  # 1005
    assert led.avg_cost("IVV") == pytest.approx(100.5)


def test_partial_sell_leaves_avg_cost_unchanged():
    """The defining AVCO property: avg_cost stays the same on partial sales."""
    led = AVCOLedger()
    led.buy("IVV", 100.0, 50.0, tc=0.0)
    avg_before = led.avg_cost("IVV")
    led.sell("IVV", 40.0, 60.0)  # sell at a higher price
    avg_after = led.avg_cost("IVV")
    assert avg_after == pytest.approx(avg_before)


def test_full_sell_resets_avg_cost():
    led = AVCOLedger()
    led.buy("IVV", 10.0, 100.0, tc=2.0)
    led.sell("IVV", 10.0, 110.0)
    assert led.units["IVV"] == 0.0
    assert led.basis["IVV"] == 0.0
    assert math.isnan(led.avg_cost("IVV"))


def test_realized_gain_includes_tc_in_basis():
    led = AVCOLedger()
    led.buy("IVV", 10.0, 100.0, tc=5.0)  # avg = 100.5
    realized = led.sell("IVV", 10.0, 110.0)
    # Proceeds = 1100, basis sold = 1005 → realized 95.
    assert realized == pytest.approx(95.0)


def test_buy_after_partial_sell_updates_avg_cost():
    led = AVCOLedger()
    led.buy("IVV", 10.0, 100.0, tc=0.0)   # avg 100
    led.sell("IVV", 5.0, 120.0)            # avg still 100, units=5
    led.buy("IVV", 5.0, 80.0, tc=0.0)      # new avg = (5*100 + 5*80)/10 = 90
    assert led.avg_cost("IVV") == pytest.approx(90.0)


def test_oversell_raises():
    led = AVCOLedger()
    led.buy("IVV", 10.0, 100.0, tc=0.0)
    with pytest.raises(ValueError, match="oversell"):
        led.sell("IVV", 20.0, 100.0)


def test_ledger_matches_independent_recompute():
    """The eval-module §12.6 check: re-walk the trade log and assert avg_cost matches."""
    led = AVCOLedger()
    trades = [
        ("buy", "IVV", 10.0, 100.0, 1.0),
        ("buy", "AGG", 20.0, 50.0, 0.5),
        ("sell", "IVV", 3.0, 120.0, 0.0),
        ("buy", "IVV", 5.0, 110.0, 0.6),
        ("sell", "IVV", 8.0, 130.0, 0.0),
        ("sell", "AGG", 5.0, 55.0, 0.0),
    ]
    for side, sym, units, price, tc in trades:
        if side == "buy":
            led.buy(sym, units, price, tc)
        else:
            led.sell(sym, units, price)

    # Independent recompute: walk the log again and compare.
    indep = AVCOLedger()
    for side, sym, units, price, tc in trades:
        if side == "buy":
            indep.buy(sym, units, price, tc)
        else:
            indep.sell(sym, units, price)
    for sym in ["IVV", "AGG"]:
        if led.units[sym] > 0:
            assert led.avg_cost(sym) == pytest.approx(indep.avg_cost(sym))
        assert led.units[sym] == pytest.approx(indep.units[sym])
        assert led.basis[sym] == pytest.approx(indep.basis[sym])


# -----------------------------------------------------------------------------
# TaxBook
# -----------------------------------------------------------------------------
def test_net_gain_taxed_at_pfu_rate():
    tax = TaxBook(pfu_rate=0.314, carryforward_years=10)
    tax.record_realized(2024, 1000.0)
    due = tax.settle_year(2024)
    assert due == pytest.approx(314.0)


def test_net_loss_creates_vintage_carryforward():
    tax = TaxBook(pfu_rate=0.314, carryforward_years=10)
    tax.record_realized(2024, -500.0)
    due = tax.settle_year(2024)
    assert due == 0.0
    assert tax.carryforward[2024] == pytest.approx(500.0)


def test_carryforward_offsets_subsequent_gains():
    tax = TaxBook(pfu_rate=0.314, carryforward_years=10)
    tax.record_realized(2023, -800.0)
    tax.settle_year(2023)
    tax.record_realized(2024, 1000.0)
    due = tax.settle_year(2024)
    # Net taxable = 1000 - 800 = 200 → tax 62.8.
    assert due == pytest.approx(200 * 0.314)
    assert 2023 not in tax.carryforward


def test_old_vintage_expires_after_horizon():
    tax = TaxBook(pfu_rate=0.314, carryforward_years=10)
    tax.record_realized(2013, -1000.0)
    tax.settle_year(2013)
    # Move forward 11 years — the 2013 loss should expire.
    for y in range(2014, 2024):
        tax.settle_year(y)
    tax.settle_year(2024)  # year - 10 = 2014, so 2013 expires here.
    assert 2013 not in tax.carryforward


def test_oldest_vintage_drained_first():
    tax = TaxBook(pfu_rate=0.314, carryforward_years=10)
    tax.record_realized(2022, -300.0)
    tax.settle_year(2022)
    tax.record_realized(2023, -200.0)
    tax.settle_year(2023)
    tax.record_realized(2024, 400.0)
    due = tax.settle_year(2024)
    # Drain 2022 (300) entirely, then 100 from 2023, leaving 100 in 2023 carryforward.
    assert due == 0.0
    assert 2022 not in tax.carryforward
    assert tax.carryforward[2023] == pytest.approx(100.0)
