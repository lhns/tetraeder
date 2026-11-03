"""Graduated pricing: usage is split over the tiers and each part pays its own tier's price."""

from dataclasses import dataclass
from decimal import Decimal

from legacy.plans import Tier
from legacy.util.money import ZERO


@dataclass(frozen=True)
class TierSlice:
    """The part of the usage that falls into one tier."""

    index: int
    units: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return self.unit_price * self.units


def split_into_tiers(units: int, tiers: tuple[Tier, ...]) -> list[TierSlice]:
    """Split `units` over `tiers`; a tier with `up_to=N` covers units previous_limit+1 ..= N."""
    if units < 0:
        raise ValueError("units must not be negative")
    slices: list[TierSlice] = []
    previous = 0
    for index, tier in enumerate(tiers):
        if units <= previous:
            break
        upper = units if tier.up_to is None else min(units, tier.up_to)
        in_tier = upper - previous
        if in_tier > 0:
            slices.append(TierSlice(index, in_tier, tier.unit_price))
        if tier.up_to is not None:
            previous = tier.up_to
    return slices


def price_graduated(units: int, tiers: tuple[Tier, ...]) -> Decimal:
    """Unrounded charge for `units` under graduated pricing."""
    total = ZERO
    for part in split_into_tiers(units, tiers):
        total += part.amount
    return total


def effective_unit_price(units: int, tiers: tuple[Tier, ...]) -> Decimal:
    """Average price per unit after tiering; the first tier's price for zero usage."""
    if units <= 0:
        return tiers[0].unit_price
    return price_graduated(units, tiers) / units


def units_until_next_tier(units: int, tiers: tuple[Tier, ...]) -> int | None:
    """Units that can still be used before the price changes; None in the last tier."""
    for tier in tiers:
        if tier.up_to is None:
            return None
        if units < tier.up_to:
            return tier.up_to - units
    return None


def breakdown(units: int, tiers: tuple[Tier, ...]) -> list[str]:
    """Human-readable lines, one per tier slice."""
    lines = []
    for part in split_into_tiers(units, tiers):
        lines.append(f"tier {part.index + 1}: {part.units} x {part.unit_price} = {part.amount}")
    return lines
