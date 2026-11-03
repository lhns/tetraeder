"""Shipping charges by parcel weight and zone."""

FREE_SHIPPING_THRESHOLD_CENTS = 10_000

# (max weight in grams, price in cents), checked in order
_BANDS = [
    (500, 495),
    (2_000, 895),
    (10_000, 1_495),
    (30_000, 2_995),
]
_ZONE_MULTIPLIER_PCT = {"domestic": 100, "eu": 150, "world": 220}
MAX_WEIGHT_G = 30_000


def base_price(weight_g: int) -> int:
    """Price in cents of the weight band; raises ValueError over the maximum parcel weight."""
    if weight_g <= 0:
        raise ValueError("weight must be positive")
    for limit, price in _BANDS:
        if weight_g <= limit:
            return price
    raise ValueError(f"parcel too heavy: {weight_g} g (maximum {MAX_WEIGHT_G} g)")


def shipping_cost(weight_g: int, zone: str = "domestic", subtotal_cents: int = 0) -> int:
    """Shipping in cents. Domestic orders at or above the threshold ship free; other zones pay a multiple."""
    if zone not in _ZONE_MULTIPLIER_PCT:
        raise ValueError(f"unknown zone: {zone!r}")
    price = base_price(weight_g)
    if zone == "domestic" and subtotal_cents >= FREE_SHIPPING_THRESHOLD_CENTS:
        return 0
    pct = _ZONE_MULTIPLIER_PCT[zone]
    return (price * pct + 50) // 100


def split_parcels(weight_g: int) -> list[int]:
    """Split a heavy order into parcels of at most MAX_WEIGHT_G; the last one takes the remainder."""
    if weight_g <= 0:
        raise ValueError("weight must be positive")
    full, rest = divmod(weight_g, MAX_WEIGHT_G)
    return [MAX_WEIGHT_G] * full + ([rest] if rest else [])
