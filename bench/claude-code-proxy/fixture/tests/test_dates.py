from datetime import date

import pytest

from invoicing.dates import add_days, days_overdue, due_date, end_of_month, is_weekend, parse_date


def test_parse_date_formats():
    assert parse_date("2024-03-31") == date(2024, 3, 31)
    assert parse_date("31.03.2024") == date(2024, 3, 31)


@pytest.mark.parametrize("text", ["2024/03/31", "31-03-2024", "yesterday", ""])
def test_parse_date_rejects(text):
    with pytest.raises(ValueError):
        parse_date(text)


def test_add_days_crosses_month_and_year():
    assert add_days(date(2023, 12, 20), 15) == date(2024, 1, 4)


def test_end_of_month():
    assert end_of_month(date(2023, 4, 3)) == date(2023, 4, 30)
    assert end_of_month(date(2023, 12, 31)) == date(2023, 12, 31)
    assert end_of_month(date(2023, 2, 1)) == date(2023, 2, 28)


def test_due_date_terms():
    d = date(2024, 1, 15)
    assert due_date(d, "net30") == date(2024, 2, 14)
    assert due_date(d, "NET15") == date(2024, 1, 30)
    assert due_date(d, "eom") == date(2024, 1, 31)
    assert due_date(d, "receipt") == d


def test_due_date_unknown_terms():
    with pytest.raises(ValueError):
        due_date(date(2024, 1, 1), "whenever")


def test_days_overdue():
    assert days_overdue(date(2024, 1, 10), date(2024, 1, 15)) == 5
    assert days_overdue(date(2024, 1, 10), date(2024, 1, 10)) == 0
    assert days_overdue(date(2024, 1, 10), date(2024, 1, 1)) == 0


def test_is_weekend():
    assert is_weekend(date(2024, 3, 2))
    assert not is_weekend(date(2024, 3, 4))
