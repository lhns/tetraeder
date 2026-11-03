"""Input validation for records that arrive from outside (CSV feeds, JSON dumps)."""

import re
from typing import Any

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")


class ValidationFailure(ValueError):
    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def require(record: dict[str, Any], field: str) -> Any:
    value = record.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationFailure(field, "required")
    return value


def valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(value.strip()))


def valid_identifier(value: str) -> bool:
    return bool(_ID_RE.match(value))


def valid_country(value: str) -> bool:
    return bool(_COUNTRY_RE.match(value))


def check_email(record: dict[str, Any], field: str = "email") -> str:
    value = str(require(record, field)).strip()
    if not valid_email(value):
        raise ValidationFailure(field, f"not an email address: {value!r}")
    return value.lower()


def check_identifier(record: dict[str, Any], field: str) -> str:
    value = str(require(record, field)).strip()
    if not valid_identifier(value):
        raise ValidationFailure(field, f"bad identifier: {value!r}")
    return value


def check_country(record: dict[str, Any], field: str = "country") -> str:
    value = str(require(record, field)).strip().upper()
    if not valid_country(value):
        raise ValidationFailure(field, f"expected a 2-letter country code, got {value!r}")
    return value


def check_positive_int(record: dict[str, Any], field: str, allow_zero: bool = False) -> int:
    raw = require(record, field)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValidationFailure(field, f"not an integer: {raw!r}") from None
    if n < 0 or (n == 0 and not allow_zero):
        raise ValidationFailure(field, f"must be {'non-negative' if allow_zero else 'positive'}, got {n}")
    return n


def check_choice(record: dict[str, Any], field: str, choices: set[str]) -> str:
    value = str(require(record, field)).strip().lower()
    if value not in choices:
        raise ValidationFailure(field, f"must be one of {sorted(choices)}, got {value!r}")
    return value


def collect_errors(record: dict[str, Any], checks: list) -> list[str]:
    """Run several checks (callables taking the record) and return every failure message."""
    errors = []
    for check in checks:
        try:
            check(record)
        except ValidationFailure as e:
            errors.append(str(e))
    return errors
