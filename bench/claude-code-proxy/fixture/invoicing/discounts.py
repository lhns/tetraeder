"""Invoice-level discounts: a percentage or a fixed amount."""

from dataclasses import dataclass

from invoicing.money import apply_rate, clamp_cents, parse_money


@dataclass(frozen=True)
class Discount:
    kind: str  # "percent" or "fixed"
    value: int  # basis points for percent, cents for fixed


def parse_discount(text: str) -> Discount:
    """"10%" and "12.5%" are percentages; anything else is a money amount."""
    text = text.strip()
    if text.endswith("%"):
        pct = text[:-1].strip()
        whole, _, frac = pct.partition(".")
        bp = int(whole) * 100 + int((frac or "0").ljust(2, "0")[:2])
        if not 0 <= bp <= 10_000:
            raise ValueError(f"percentage out of range: {text!r}")
        return Discount("percent", bp)
    return Discount("fixed", parse_money(text))


def discount_amount(subtotal: int, discount: Discount | None) -> int:
    """Cents taken off `subtotal`; never more than the subtotal."""
    if discount is None:
        return 0
    if discount.kind == "percent":
        return apply_rate(subtotal, discount.value)
    return clamp_cents(discount.value, 0, subtotal)
