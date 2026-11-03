from datetime import date

import pytest

from invoicing.dates import due_date, end_of_month


@pytest.mark.parametrize(
    "d,expected",
    [
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2000, 2, 15), date(2000, 2, 29)),
        (date(2100, 2, 15), date(2100, 2, 28)),
        (date(2023, 2, 28), date(2023, 2, 28)),
        (date(2024, 4, 30), date(2024, 4, 30)),
        (date(2024, 12, 5), date(2024, 12, 31)),
        (date(2024, 1, 1), date(2024, 1, 31)),
    ],
)
def test_end_of_month(d, expected):
    assert end_of_month(d) == expected
    assert due_date(d, "eom") == expected
