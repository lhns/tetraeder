import pytest

from invoicing.money import apply_rate, clamp_cents, format_money, parse_money, round_half_up


@pytest.mark.parametrize(
    "text,cents",
    [("12", 1200), ("$1,234.56", 123456), ("12.5", 1250), ("-3.10", -310), (" $0.05 ", 5), ("1,000", 100000)],
)
def test_parse_money(text, cents):
    assert parse_money(text) == cents


@pytest.mark.parametrize("text", ["", "abc", "1.234", "$", "1,00"])
def test_parse_money_rejects(text):
    with pytest.raises(ValueError):
        parse_money(text)


def test_format_money():
    assert format_money(123456) == "$1,234.56"
    assert format_money(-500) == "-$5.00"
    assert format_money(7) == "$0.07"
    assert format_money(100, symbol="EUR ") == "EUR 1.00"


def test_round_half_up_ties_go_away_from_zero():
    assert round_half_up(5, 10) == 1
    assert round_half_up(15, 10) == 2
    assert round_half_up(25, 10) == 3
    assert round_half_up(-25, 10) == -3
    assert round_half_up(24, 10) == 2


def test_round_half_up_needs_positive_denominator():
    with pytest.raises(ValueError):
        round_half_up(1, 0)


def test_apply_rate():
    assert apply_rate(10_000, 2000) == 2000
    assert apply_rate(999, 2000) == 200  # 199.8
    assert apply_rate(250, 500) == 13  # 12.5 rounds up


def test_clamp_cents():
    assert clamp_cents(-5) == 0
    assert clamp_cents(50, 0, 40) == 40
    assert clamp_cents(50) == 50
