"""Dispatch from a plan's meter definition to the right price function."""

from dataclasses import dataclass
from decimal import Decimal

from legacy.errors import PricingError
from legacy.plans import MeterPlan
from legacy.pricing.flat import billable_units, price_flat
from legacy.pricing.tiers import price_graduated
from legacy.pricing.volume import price_volume
from legacy.util.money import quantize_cents


@dataclass(frozen=True)
class MeterCharge:
    meter: str
    used: int
    included: int
    billed_units: int
    mode: str
    amount: Decimal


def price_meter(plan_meter: MeterPlan, used: int) -> MeterCharge:
    """Charge for `used` units of one meter after the included allowance.

    Graduated and volume pricing run on the billable units (usage minus allowance), so a tier limit counts
    billable units.
    """
    if used < 0:
        raise ValueError("used must not be negative")
    billed = billable_units(used, plan_meter.included_units)
    if plan_meter.mode == "flat":
        raw = price_flat(billed, plan_meter.tiers)
    elif plan_meter.mode == "graduated":
        raw = price_graduated(billed, plan_meter.tiers)
    elif plan_meter.mode == "volume":
        raw = price_volume(billed, plan_meter.tiers)
    else:
        raise PricingError(f"{plan_meter.meter}: unknown pricing mode {plan_meter.mode!r}")
    return MeterCharge(plan_meter.meter, used, plan_meter.included_units, billed, plan_meter.mode, quantize_cents(raw))


def price_all(plan_meters: dict[str, MeterPlan], usage: dict[str, int]) -> list[MeterCharge]:
    """Charges for every meter of the plan, in alphabetical order; meters with no usage cost nothing."""
    return [price_meter(plan_meters[name], usage.get(name, 0)) for name in sorted(plan_meters)]


def unbilled_meters(plan_meters: dict[str, MeterPlan], usage: dict[str, int]) -> list[str]:
    """Meters that have usage but are not part of the plan (reported, never charged)."""
    return sorted(name for name, units in usage.items() if units and name not in plan_meters)
