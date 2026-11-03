"""Volume pricing: the whole usage is priced at the rate of the single tier it falls into."""

from decimal import Decimal

from legacy.errors import PricingError
from legacy.plans import Tier
from legacy.util.money import ZERO


def select_tier(units: int, tiers: tuple[Tier, ...]) -> Tier:
    """The tier that prices `units`. A tier with `up_to=N` includes usage of exactly N units."""
    if units < 0:
        raise ValueError("units must not be negative")
    for tier in tiers:
        if tier.up_to is None or units <= tier.up_to:
            return tier
    raise PricingError("tier table has no open-ended last tier")


def price_volume(units: int, tiers: tuple[Tier, ...]) -> Decimal:
    """Unrounded charge: all units at the selected tier's price."""
    if units == 0:
        return ZERO
    return select_tier(units, tiers).unit_price * units


def tier_index(units: int, tiers: tuple[Tier, ...]) -> int:
    chosen = select_tier(units, tiers)
    return tiers.index(chosen)


def savings_vs_first_tier(units: int, tiers: tuple[Tier, ...]) -> Decimal:
    """How much cheaper the volume price is than pricing everything at the first tier."""
    first = tiers[0].unit_price * units
    return first - price_volume(units, tiers)


def next_discount_threshold(units: int, tiers: tuple[Tier, ...]) -> int | None:
    """Usage level at which the unit price drops again (one above the current tier's limit)."""
    chosen = select_tier(units, tiers)
    if chosen.up_to is None:
        return None
    return chosen.up_to + 1


def price_table(tiers: tuple[Tier, ...], samples: list[int]) -> list[tuple[int, Decimal]]:
    """Charge for each sample usage, for the quote sheets."""
    return [(n, price_volume(n, tiers)) for n in samples]
