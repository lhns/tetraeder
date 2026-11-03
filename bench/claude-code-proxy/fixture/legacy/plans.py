"""Plan catalog: which meters a plan bills and how each meter is priced.

Tier tables
-----------
A tier table is an ordered list of `Tier(up_to, unit_price)`. `up_to` is the inclusive upper bound of the tier in
units; the last tier has `up_to=None` (open ended). A meter is priced in one of three ways:

* `flat`:      every unit costs the first tier's price.
* `graduated`: units are split over the tiers and each part is priced at its tier.
* `volume`:    all units are priced at the price of the one tier the total falls into.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from legacy.errors import PricingError, UnknownPlan

PRICING_MODES = ("flat", "graduated", "volume")


@dataclass(frozen=True)
class Tier:
    up_to: int | None
    unit_price: Decimal


@dataclass(frozen=True)
class MeterPlan:
    meter: str
    mode: str
    tiers: tuple[Tier, ...]
    included_units: int = 0
    unit_label: str = "unit"

    def __post_init__(self):
        validate_tiers(self.meter, self.mode, self.tiers)


@dataclass(frozen=True)
class Plan:
    plan_id: str
    name: str
    base_fee: Decimal
    meters: dict[str, MeterPlan] = field(default_factory=dict)
    currency: str = "USD"
    min_commit: Decimal = Decimal("0")
    trial_days: int = 0

    def meter_plan(self, meter: str) -> MeterPlan | None:
        return self.meters.get(meter)


def validate_tiers(meter: str, mode: str, tiers: tuple[Tier, ...]) -> None:
    if mode not in PRICING_MODES:
        raise PricingError(f"{meter}: unknown pricing mode {mode!r}")
    if not tiers:
        raise PricingError(f"{meter}: no tiers")
    last_limit = 0
    for i, tier in enumerate(tiers):
        is_last = i == len(tiers) - 1
        if tier.unit_price < 0:
            raise PricingError(f"{meter}: negative price in tier {i}")
        if tier.up_to is None:
            if not is_last:
                raise PricingError(f"{meter}: open-ended tier must be last")
        else:
            if is_last:
                raise PricingError(f"{meter}: last tier must be open ended")
            if tier.up_to <= last_limit:
                raise PricingError(f"{meter}: tier limits must increase ({tier.up_to} after {last_limit})")
            last_limit = tier.up_to


def _tiers(*pairs: tuple[int | None, str]) -> tuple[Tier, ...]:
    return tuple(Tier(limit, Decimal(price)) for limit, price in pairs)


def _build_catalog() -> dict[str, Plan]:
    starter = Plan(
        "starter",
        "Starter",
        Decimal("0"),
        {
            "api_calls": MeterPlan("api_calls", "flat", _tiers((None, "0.0020")), included_units=1_000, unit_label="call"),
            "storage_gb": MeterPlan("storage_gb", "flat", _tiers((None, "0.30")), included_units=5, unit_label="GB"),
        },
        trial_days=14,
    )
    growth = Plan(
        "growth",
        "Growth",
        Decimal("49.00"),
        {
            "api_calls": MeterPlan(
                "api_calls",
                "graduated",
                _tiers((10_000, "0.0015"), (100_000, "0.0010"), (None, "0.0006")),
                included_units=5_000,
                unit_label="call",
            ),
            "storage_gb": MeterPlan(
                "storage_gb",
                "volume",
                _tiers((100, "0.25"), (1_000, "0.20"), (None, "0.15")),
                included_units=10,
                unit_label="GB",
            ),
            "seats": MeterPlan("seats", "flat", _tiers((None, "9.00")), included_units=3, unit_label="seat"),
        },
        min_commit=Decimal("49.00"),
    )
    scale = Plan(
        "scale",
        "Scale",
        Decimal("299.00"),
        {
            "api_calls": MeterPlan(
                "api_calls",
                "graduated",
                _tiers((100_000, "0.0010"), (1_000_000, "0.0007"), (None, "0.0004")),
                included_units=50_000,
                unit_label="call",
            ),
            "storage_gb": MeterPlan(
                "storage_gb",
                "volume",
                _tiers((500, "0.18"), (5_000, "0.14"), (None, "0.10")),
                included_units=100,
                unit_label="GB",
            ),
            "seats": MeterPlan("seats", "graduated", _tiers((20, "8.00"), (None, "6.50")), included_units=10, unit_label="seat"),
            "egress_gb": MeterPlan("egress_gb", "volume", _tiers((1_000, "0.09"), (None, "0.07")), unit_label="GB"),
        },
        min_commit=Decimal("299.00"),
    )
    enterprise = Plan(
        "enterprise",
        "Enterprise",
        Decimal("1500.00"),
        {
            "api_calls": MeterPlan(
                "api_calls",
                "graduated",
                _tiers((1_000_000, "0.0006"), (10_000_000, "0.0004"), (None, "0.0002")),
                included_units=500_000,
                unit_label="call",
            ),
            "storage_gb": MeterPlan(
                "storage_gb",
                "volume",
                _tiers((2_000, "0.12"), (20_000, "0.09"), (None, "0.07")),
                included_units=1_000,
                unit_label="GB",
            ),
            "seats": MeterPlan("seats", "flat", _tiers((None, "5.00")), included_units=50, unit_label="seat"),
            "egress_gb": MeterPlan("egress_gb", "volume", _tiers((5_000, "0.06"), (None, "0.04")), unit_label="GB"),
        },
        min_commit=Decimal("1500.00"),
    )
    legacy_flat = Plan(
        "flat99",
        "Flat 99",
        Decimal("99.00"),
        {"seats": MeterPlan("seats", "flat", _tiers((None, "0")), included_units=25, unit_label="seat")},
        currency="EUR",
    )
    return {p.plan_id: p for p in (starter, growth, scale, enterprise, legacy_flat)}


CATALOG: dict[str, Plan] = _build_catalog()


def get_plan(plan_id: str) -> Plan:
    try:
        return CATALOG[plan_id]
    except KeyError:
        raise UnknownPlan(plan_id) from None


def plan_ids() -> list[str]:
    return sorted(CATALOG)


def upgrade_path(plan_id: str) -> list[str]:
    """Plans ordered by base fee that cost more than `plan_id`."""
    base = get_plan(plan_id).base_fee
    higher = [p for p in CATALOG.values() if p.base_fee > base]
    return [p.plan_id for p in sorted(higher, key=lambda p: p.base_fee)]


def metered_names(plan: Plan) -> list[str]:
    return sorted(plan.meters)
