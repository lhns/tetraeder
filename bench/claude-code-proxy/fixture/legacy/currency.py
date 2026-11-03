"""Currency conversion for reports that span plans billed in different currencies."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from legacy.config import SUPPORTED_CURRENCIES
from legacy.util.money import D, quantize

# units of the currency per 1 USD, as of the last manual update
DEFAULT_RATES = {
    "USD": Decimal("1"),
    "EUR": Decimal("0.92"),
    "GBP": Decimal("0.79"),
    "CHF": Decimal("0.88"),
    "JPY": Decimal("151.2"),
}


@dataclass(frozen=True)
class RateTable:
    as_of: date
    per_usd: dict[str, Decimal]

    def rate(self, currency: str) -> Decimal:
        try:
            return self.per_usd[currency]
        except KeyError:
            raise ValueError(f"no rate for {currency}") from None

    def convert(self, amount: Decimal, source: str, target: str) -> Decimal:
        """Convert through USD and round to the target currency's decimals."""
        if source == target:
            return quantize(amount, target)
        usd = D(amount) / self.rate(source)
        return quantize(usd * self.rate(target), target)

    def supports(self, currency: str) -> bool:
        return currency in self.per_usd and currency in SUPPORTED_CURRENCIES


def default_table(as_of: date) -> RateTable:
    return RateTable(as_of, dict(DEFAULT_RATES))


def parse_rates(lines: list[str], as_of: date) -> RateTable:
    """Parse "EUR 0.92" lines; USD is always 1. Unknown or malformed lines raise ValueError."""
    rates = {"USD": Decimal("1")}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"bad rate line: {raw!r}")
        code, value = parts[0].upper(), D(parts[1])
        if code not in SUPPORTED_CURRENCIES:
            raise ValueError(f"unsupported currency: {code}")
        if value <= 0:
            raise ValueError(f"rate must be positive: {raw!r}")
        rates[code] = value
    return RateTable(as_of, rates)


def total_in(table: RateTable, amounts: list[tuple[Decimal, str]], target: str) -> Decimal:
    """Sum amounts given in several currencies, converting each before adding."""
    return sum((table.convert(a, c, target) for a, c in amounts), Decimal("0"))
