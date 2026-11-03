"""Regression: tax on exact half-cent amounts is rounded down on some invoices."""

from invoicing.tax import tax_for_line


def test_reduced_rate_on_250_cents_rounds_up():
    # 5% of $2.50 is 12.5 cents, which must round up to 13
    assert tax_for_line(250, "reduced") == 13


def test_standard_rate_half_cent_rounds_up():
    assert tax_for_line(675, "reduced") == 34  # 33.75
    assert tax_for_line(25, "std") == 5
    assert tax_for_line(1_025, "reduced") == 51  # 51.25 -> 51
    assert tax_for_line(1_030, "reduced") == 52  # 51.5 -> 52
