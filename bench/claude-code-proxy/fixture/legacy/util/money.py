"""Decimal money helpers."""

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable

from legacy.config import currency_decimals, currency_symbol

CENT = Decimal("0.01")
ZERO = Decimal("0")


def D(value) -> Decimal:
    """Decimal from int, str or Decimal; floats go through str so 0.1 stays 0.1."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        value = repr(value)
    try:
        return Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"not a number: {value!r}") from None


def quantize(amount: Decimal, currency: str = "USD") -> Decimal:
    """Round half up to the currency's decimals."""
    places = currency_decimals(currency)
    exp = Decimal(1).scaleb(-places)
    return D(amount).quantize(exp, rounding=ROUND_HALF_UP)


def quantize_cents(amount: Decimal) -> Decimal:
    return D(amount).quantize(CENT, rounding=ROUND_HALF_UP)


def truncate_cents(amount: Decimal) -> Decimal:
    return D(amount).quantize(CENT, rounding=ROUND_DOWN)


def to_cents(amount: Decimal) -> int:
    return int(quantize_cents(amount) * 100)


def from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(CENT)


def total(amounts: Iterable[Decimal]) -> Decimal:
    result = ZERO
    for a in amounts:
        result += D(a)
    return result


def format_amount(amount: Decimal, currency: str = "USD") -> str:
    q = quantize(amount, currency)
    sign = "-" if q < 0 else ""
    places = currency_decimals(currency)
    body = f"{abs(q):,.{places}f}"
    return f"{sign}{currency_symbol(currency)}{body}"


def split_amount(amount: Decimal, weights: list[int]) -> list[Decimal]:
    """Split `amount` by integer weights into cent amounts that add up exactly.

    The remainder cents go to the largest weights first, ties to the earlier entry.
    """
    if not weights or any(w < 0 for w in weights) or sum(weights) == 0:
        raise ValueError("weights must be non-negative with a positive sum")
    cents = to_cents(amount)
    wsum = sum(weights)
    parts = [cents * w // wsum for w in weights]
    leftover = cents - sum(parts)
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    for i in order[:leftover]:
        parts[i] += 1
    return [from_cents(p) for p in parts]


def percent_of(amount: Decimal, percent: Decimal) -> Decimal:
    """`percent` (e.g. Decimal("12.5")) of amount, unrounded."""
    return D(amount) * D(percent) / Decimal(100)


def clamp(value: Decimal, low: Decimal | None = None, high: Decimal | None = None) -> Decimal:
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value
