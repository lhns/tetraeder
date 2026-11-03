"""Flat pricing: every unit costs the same."""

from decimal import Decimal

from legacy.plans import Tier


def price_flat(units: int, tiers: tuple[Tier, ...]) -> Decimal:
    if units < 0:
        raise ValueError("units must not be negative")
    return tiers[0].unit_price * units


def billable_units(used: int, included: int) -> int:
    """Units left after the plan's included allowance."""
    return max(0, used - included)
