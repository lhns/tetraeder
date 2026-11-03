import pytest

from invoicing.shipping import (
    FREE_SHIPPING_THRESHOLD_CENTS,
    MAX_WEIGHT_G,
    base_price,
    shipping_cost,
    split_parcels,
)


@pytest.mark.parametrize(
    "weight,price",
    [(1, 495), (500, 495), (501, 895), (2_000, 895), (2_001, 1_495), (10_000, 1_495), (10_001, 2_995), (30_000, 2_995)],
)
def test_base_price_bands(weight, price):
    assert base_price(weight) == price


@pytest.mark.parametrize("weight", [0, -5])
def test_base_price_rejects_non_positive_weight(weight):
    with pytest.raises(ValueError):
        base_price(weight)


def test_base_price_rejects_too_heavy():
    with pytest.raises(ValueError, match="too heavy"):
        base_price(MAX_WEIGHT_G + 1)


def test_zone_multipliers():
    assert shipping_cost(400, "domestic") == 495
    assert shipping_cost(400, "eu") == 743  # 742.5 rounds up
    assert shipping_cost(400, "world") == 1089  # 1089.0


def test_unknown_zone():
    with pytest.raises(ValueError, match="zone"):
        shipping_cost(400, "moon")


def test_free_shipping_only_domestic_at_threshold():
    assert shipping_cost(400, "domestic", FREE_SHIPPING_THRESHOLD_CENTS - 1) == 495
    assert shipping_cost(400, "domestic", FREE_SHIPPING_THRESHOLD_CENTS) == 0
    assert shipping_cost(400, "eu", FREE_SHIPPING_THRESHOLD_CENTS) == 743


def test_heavy_parcel_still_errors_when_free():
    with pytest.raises(ValueError):
        shipping_cost(MAX_WEIGHT_G + 1, "domestic", FREE_SHIPPING_THRESHOLD_CENTS)


def test_split_parcels():
    assert split_parcels(100) == [100]
    assert split_parcels(MAX_WEIGHT_G) == [MAX_WEIGHT_G]
    assert split_parcels(MAX_WEIGHT_G * 2 + 7) == [MAX_WEIGHT_G, MAX_WEIGHT_G, 7]
    assert sum(split_parcels(123_456)) == 123_456


def test_split_parcels_rejects_non_positive():
    with pytest.raises(ValueError):
        split_parcels(0)
