"""Promo codes: percentage or fixed discounts with validity windows and plan restrictions."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from legacy.errors import PromoError
from legacy.util.money import ZERO, clamp, percent_of, quantize_cents


@dataclass(frozen=True)
class Promo:
    code: str
    kind: str  # "percent" or "fixed"
    value: Decimal
    valid_from: date
    valid_until: date | None = None
    plans: frozenset[str] = field(default_factory=frozenset)  # empty means every plan
    max_discount: Decimal | None = None
    stackable: bool = False

    def active_on(self, day: date) -> bool:
        if day < self.valid_from:
            return False
        return self.valid_until is None or day <= self.valid_until

    def applies_to_plan(self, plan_id: str) -> bool:
        return not self.plans or plan_id in self.plans


PROMOS: dict[str, Promo] = {
    p.code: p
    for p in (
        Promo("WELCOME10", "percent", Decimal("10"), date(2023, 1, 1), None, frozenset({"starter", "growth"})),
        Promo("SPRING25", "percent", Decimal("25"), date(2024, 3, 1), date(2024, 5, 31), max_discount=Decimal("200.00")),
        Promo("FIXED50", "fixed", Decimal("50.00"), date(2023, 6, 1), date(2025, 12, 31), frozenset({"scale", "enterprise"})),
        Promo("LOYAL5", "percent", Decimal("5"), date(2022, 1, 1), None, stackable=True),
    )
}


def lookup(code: str) -> Promo:
    try:
        return PROMOS[code.strip().upper()]
    except KeyError:
        raise PromoError(f"unknown promo code: {code}") from None


def check_applicable(promo: Promo, plan_id: str, day: date) -> None:
    if not promo.active_on(day):
        raise PromoError(f"{promo.code} is not valid on {day.isoformat()}")
    if not promo.applies_to_plan(plan_id):
        raise PromoError(f"{promo.code} does not apply to plan {plan_id}")


def discount_for(promo: Promo, amount: Decimal) -> Decimal:
    """Discount in dollars on `amount`, never above the amount or the promo's cap."""
    if amount <= 0:
        return ZERO
    if promo.kind == "percent":
        raw = percent_of(amount, promo.value)
    elif promo.kind == "fixed":
        raw = promo.value
    else:
        raise PromoError(f"{promo.code}: unknown kind {promo.kind!r}")
    return quantize_cents(clamp(raw, ZERO, min(amount, promo.max_discount) if promo.max_discount else amount))


def apply_promos(codes: list[str], plan_id: str, day: date, amount: Decimal) -> tuple[Decimal, list[str]]:
    """Apply the first non-stackable code that fits, then every stackable one. Returns (discount, applied codes).

    Codes that do not apply are skipped silently; stacked percentages work on the amount left after earlier ones.
    """
    remaining = amount
    total = ZERO
    applied: list[str] = []
    used_main = False
    for code in codes:
        try:
            promo = lookup(code)
            check_applicable(promo, plan_id, day)
        except PromoError:
            continue
        if not promo.stackable:
            if used_main:
                continue
            used_main = True
        d = discount_for(promo, remaining)
        remaining -= d
        total += d
        applied.append(promo.code)
    return total, applied
