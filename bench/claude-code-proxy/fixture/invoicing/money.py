"""Money as integer cents: parsing, formatting and rate arithmetic."""

import re

_MONEY_RE = re.compile(r"^(-)?\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?$")


def parse_money(text: str) -> int:
    """Parse "$1,234.56", "12", "12.5" or "-3.10" into cents."""
    cleaned = text.strip().replace(" ", "")
    m = _MONEY_RE.match(cleaned)
    if not m:
        raise ValueError(f"not a money amount: {text!r}")
    sign, whole, frac = m.groups()
    cents = int(whole.replace(",", "")) * 100 + int((frac or "0").ljust(2, "0"))
    return -cents if sign else cents


def format_money(cents: int, symbol: str = "$") -> str:
    """Format cents as "$1,234.56"; negatives as "-$5.00"."""
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    return f"{sign}{symbol}{whole:,}.{frac:02d}"


def round_half_up(numerator: int, denominator: int) -> int:
    """Integer division rounded half away from zero (not banker's rounding)."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    q, r = divmod(abs(numerator), denominator)
    if r * 2 >= denominator:
        q += 1
    return sign * q


def apply_rate(cents: int, rate_bp: int) -> int:
    """Take `rate_bp` basis points of `cents` (100 bp = 1%), rounded half up."""
    return round_half_up(cents * rate_bp, 10_000)


def clamp_cents(cents: int, low: int = 0, high: int | None = None) -> int:
    """Limit cents to [low, high]; no upper limit when high is None."""
    if cents < low:
        return low
    if high is not None and cents > high:
        return high
    return cents
