"""Tax codes and tax computation."""

from invoicing.money import apply_rate

TAX_RATES_BP = {
    "std": 2000,
    "reduced": 500,
    "zero": 0,
}


def rate_for(code: str) -> int:
    try:
        return TAX_RATES_BP[code]
    except KeyError:
        raise ValueError(f"unknown tax code: {code!r}") from None


def tax_for_line(net_cents: int, code: str) -> int:
    """Tax on one line; each line is rounded on its own."""
    return apply_rate(net_cents, rate_for(code))


def tax_for_lines(lines: list[tuple[int, str]]) -> int:
    return sum(tax_for_line(net, code) for net, code in lines)
