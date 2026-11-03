from decimal import Decimal

import pytest

from legacy.errors import PricingError
from legacy.plans import MeterPlan, Tier, get_plan, upgrade_path
from legacy.pricing.engine import price_meter
from legacy.pricing.flat import billable_units, price_flat
from legacy.pricing.promos import apply_promos, lookup
from legacy.pricing.tiers import price_graduated, split_into_tiers, units_until_next_tier
from legacy.pricing.volume import price_volume, select_tier

from datetime import date

T = (Tier(100, Decimal("1.00")), Tier(1000, Decimal("0.50")), Tier(None, Decimal("0.25")))


def test_graduated_split():
    parts = split_into_tiers(250, T)
    assert [(p.index, p.units) for p in parts] == [(0, 100), (1, 150)]
    assert price_graduated(250, T) == Decimal("175.00")
    assert price_graduated(0, T) == 0
    assert price_graduated(100, T) == Decimal("100.00")
    assert price_graduated(101, T) == Decimal("100.50")


def test_graduated_open_ended_tier():
    assert price_graduated(1500, T) == Decimal("100") + Decimal("450") + Decimal("125")


def test_units_until_next_tier():
    assert units_until_next_tier(40, T) == 60
    assert units_until_next_tier(5000, T) is None


def test_volume_prices_everything_at_one_tier():
    assert select_tier(50, T).unit_price == Decimal("1.00")
    assert select_tier(500, T).unit_price == Decimal("0.50")
    assert select_tier(5000, T).unit_price == Decimal("0.25")
    assert price_volume(500, T) == Decimal("250.00")
    assert price_volume(0, T) == 0


def test_flat_and_allowance():
    assert price_flat(10, (Tier(None, Decimal("0.30")),)) == Decimal("3.00")
    assert billable_units(7, 5) == 2
    assert billable_units(3, 5) == 0


def test_price_meter_uses_allowance_then_mode():
    mp = get_plan("growth").meters["api_calls"]  # graduated, 5000 included
    c = price_meter(mp, 15_000)
    assert c.billed_units == 10_000
    assert c.amount == Decimal("15.00")
    assert price_meter(mp, 100).amount == 0
    vol = get_plan("growth").meters["storage_gb"]  # volume, 10 GB included
    assert price_meter(vol, 60).amount == Decimal("12.50")  # 50 billable at 0.25


def test_tier_table_validation():
    with pytest.raises(PricingError):
        MeterPlan("x", "graduated", (Tier(10, Decimal("1")), Tier(5, Decimal("1")), Tier(None, Decimal("1"))))
    with pytest.raises(PricingError):
        MeterPlan("x", "graduated", (Tier(10, Decimal("1")),))
    with pytest.raises(PricingError):
        MeterPlan("x", "mystery", (Tier(None, Decimal("1")),))


def test_promos():
    assert lookup("welcome10").value == Decimal("10")
    d, applied = apply_promos(["WELCOME10"], "growth", date(2024, 1, 1), Decimal("200.00"))
    assert (d, applied) == (Decimal("20.00"), ["WELCOME10"])
    d, applied = apply_promos(["WELCOME10"], "scale", date(2024, 1, 1), Decimal("200.00"))
    assert applied == []
    d, applied = apply_promos(["SPRING25", "LOYAL5"], "scale", date(2024, 4, 1), Decimal("1000.00"))
    assert applied == ["SPRING25", "LOYAL5"]
    assert d == Decimal("200.00") + Decimal("40.00")  # 25% capped at 200, then 5% of the remaining 800


def test_upgrade_path():
    assert upgrade_path("growth") == ["flat99", "scale", "enterprise"]
