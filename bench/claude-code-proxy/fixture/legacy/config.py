"""Runtime settings for the billing engine.

Values come from module defaults and can be overridden through LEGACY_* environment variables, read once by
`load_settings`. Tests build `Settings` directly.
"""

import os
from dataclasses import dataclass, field, replace
from decimal import Decimal

SUPPORTED_CURRENCIES = {
    "USD": {"symbol": "$", "decimals": 2},
    "EUR": {"symbol": "EUR ", "decimals": 2},
    "GBP": {"symbol": "GBP ", "decimals": 2},
    "CHF": {"symbol": "CHF ", "decimals": 2},
    "JPY": {"symbol": "JPY ", "decimals": 0},
}

DEFAULT_BILLING_DAY = 1
MAX_BILLING_DAY = 28
MIN_STATEMENT_TOTAL = Decimal("0.00")
DEFAULT_GRACE_DAYS = 14
DEFAULT_CURRENCY = "USD"


@dataclass(frozen=True)
class Settings:
    currency: str = DEFAULT_CURRENCY
    billing_day: int = DEFAULT_BILLING_DAY
    grace_days: int = DEFAULT_GRACE_DAYS
    minimum_total: Decimal = MIN_STATEMENT_TOTAL
    prorate_first_period: bool = True
    apply_promos: bool = True
    audit_enabled: bool = True
    dunning_steps: tuple[int, ...] = (3, 10, 21, 35)
    feature_flags: dict[str, bool] = field(default_factory=dict)

    def flag(self, name: str, default: bool = False) -> bool:
        return self.feature_flags.get(name, default)

    def with_flags(self, **flags: bool) -> "Settings":
        merged = {**self.feature_flags, **flags}
        return replace(self, feature_flags=merged)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    """Read settings from the environment (or the given mapping)."""
    saved = os.environ
    if environ is not None:
        os.environ = environ  # type: ignore[assignment]
    try:
        currency = os.environ.get("LEGACY_CURRENCY", DEFAULT_CURRENCY).upper()
        if currency not in SUPPORTED_CURRENCIES:
            currency = DEFAULT_CURRENCY
        day = min(max(_env_int("LEGACY_BILLING_DAY", DEFAULT_BILLING_DAY), 1), MAX_BILLING_DAY)
        return Settings(
            currency=currency,
            billing_day=day,
            grace_days=_env_int("LEGACY_GRACE_DAYS", DEFAULT_GRACE_DAYS),
            prorate_first_period=_env_bool("LEGACY_PRORATE", True),
            apply_promos=_env_bool("LEGACY_PROMOS", True),
            audit_enabled=_env_bool("LEGACY_AUDIT", True),
        )
    finally:
        os.environ = saved


def currency_decimals(code: str) -> int:
    try:
        return SUPPORTED_CURRENCIES[code]["decimals"]
    except KeyError:
        raise ValueError(f"unsupported currency: {code}") from None


def currency_symbol(code: str) -> str:
    try:
        return SUPPORTED_CURRENCIES[code]["symbol"]
    except KeyError:
        raise ValueError(f"unsupported currency: {code}") from None
