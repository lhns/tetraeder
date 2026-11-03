from datetime import date

from invoicing.dates import due_date, end_of_month


def test_end_of_month_leap_february():
    assert end_of_month(date(2024, 2, 10)) == date(2024, 2, 29)


def test_eom_terms_in_leap_year():
    assert due_date(date(2024, 2, 1), "eom") == date(2024, 2, 29)
