import pytest

from invoicing.money import apply_rate, round_half_up


@pytest.mark.parametrize(
    "num,den,expected",
    [(1, 2, 1), (3, 2, 2), (5, 2, 3), (7, 2, 4), (-1, 2, -1), (-5, 2, -3), (0, 7, 0), (10, 4, 3), (9, 4, 2), (-9, 4, -2)],
)
def test_round_half_up_table(num, den, expected):
    assert round_half_up(num, den) == expected


def test_exact_for_huge_values():
    # float division would lose precision here
    n = 10**22 + 5
    assert round_half_up(n, 10) == 10**21 + 1
    assert round_half_up(10**22 + 4, 10) == 10**21


def test_apply_rate_ties():
    assert apply_rate(250, 500) == 13
    assert apply_rate(30, 5000) == 15
    assert apply_rate(-250, 500) == -13
