"""Sales tax / VAT by customer country and region."""

from decimal import Decimal

from legacy.customers import Address
from legacy.util.money import ZERO, quantize_cents

# percent rates by ISO country
VAT_PERCENT = {
    "DE": Decimal("19"),
    "FR": Decimal("20"),
    "NL": Decimal("21"),
    "BE": Decimal("21"),
    "AT": Decimal("20"),
    "ES": Decimal("21"),
    "IT": Decimal("22"),
    "IE": Decimal("23"),
    "SE": Decimal("25"),
    "DK": Decimal("25"),
    "GB": Decimal("20"),
    "CH": Decimal("8.1"),
}

# US state sales tax percent (state level only)
US_STATE_PERCENT = {
    "CA": Decimal("7.25"),
    "NY": Decimal("4"),
    "TX": Decimal("6.25"),
    "WA": Decimal("6.5"),
    "FL": Decimal("6"),
    "IL": Decimal("6.25"),
    "OR": Decimal("0"),
    "MT": Decimal("0"),
    "DE": Decimal("0"),
}

# digital services are exempt in these US states
US_EXEMPT_DIGITAL = frozenset({"OR", "MT", "DE"})

EU_COUNTRIES = frozenset({"DE", "FR", "NL", "BE", "AT", "ES", "IT", "IE", "SE", "DK"})


def tax_percent(address: Address, business_vat_id: str | None = None, seller_country: str = "US") -> Decimal:
    """Percent charged to a customer at `address`.

    EU business customers with a VAT id are reverse charged (0%) unless the seller is in the same country.
    """
    country = address.country.upper()
    if country == "US":
        if address.region.upper() in US_EXEMPT_DIGITAL:
            return ZERO
        return US_STATE_PERCENT.get(address.region.upper(), ZERO)
    rate = VAT_PERCENT.get(country)
    if rate is None:
        return ZERO
    if business_vat_id and country in EU_COUNTRIES and country != seller_country:
        return ZERO
    return rate


def tax_amount(net: Decimal, percent: Decimal) -> Decimal:
    if net <= 0:
        return ZERO
    return quantize_cents(net * percent / Decimal(100))


def gross(net: Decimal, percent: Decimal) -> Decimal:
    return net + tax_amount(net, percent)


def describe(address: Address, percent: Decimal) -> str:
    if percent == 0:
        return "No tax"
    label = "VAT" if address.country.upper() != "US" else f"Sales tax ({address.region.upper() or 'US'})"
    return f"{label} {percent.normalize():f}%"
