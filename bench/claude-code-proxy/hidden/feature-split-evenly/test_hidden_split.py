import pytest

from invoicing.money import split_evenly


@pytest.mark.parametrize(
    "total,parts,expected",
    [
        (100, 3, [34, 33, 33]),
        (100, 4, [25, 25, 25, 25]),
        (101, 4, [26, 25, 25, 25]),
        (2, 5, [1, 1, 0, 0, 0]),
        (0, 3, [0, 0, 0]),
        (7, 1, [7]),
        (-100, 3, [-34, -33, -33]),
        (-1, 2, [-1, 0]),
    ],
)
def test_split_evenly(total, parts, expected):
    out = split_evenly(total, parts)
    assert out == expected
    assert sum(out) == total


@pytest.mark.parametrize("parts", [0, -1])
def test_split_evenly_needs_positive_parts(parts):
    with pytest.raises(ValueError):
        split_evenly(100, parts)
