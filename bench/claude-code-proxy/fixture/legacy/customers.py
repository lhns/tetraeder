"""Customer records and the in-memory registry."""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from legacy.errors import UnknownCustomer
from legacy.util.text import normalize_name
from legacy.util.validation import (
    ValidationFailure,
    check_country,
    check_email,
    check_identifier,
    require,
)

STATUSES = ("trial", "active", "suspended", "closed")
_TRANSITIONS = {
    "trial": {"active", "closed"},
    "active": {"suspended", "closed"},
    "suspended": {"active", "closed"},
    "closed": set(),
}


@dataclass
class Address:
    line1: str = ""
    line2: str = ""
    city: str = ""
    region: str = ""
    postal_code: str = ""
    country: str = "US"

    def one_line(self) -> str:
        parts = [self.line1, self.line2, self.city, f"{self.region} {self.postal_code}".strip(), self.country]
        return ", ".join(p for p in parts if p)


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    plan_id: str
    started: date
    status: str = "active"
    address: Address = field(default_factory=Address)
    promo_code: str | None = None
    credit_balance_cents: int = 0
    tags: set[str] = field(default_factory=set)

    @property
    def is_billable(self) -> bool:
        return self.status in ("trial", "active", "suspended")

    def can_move_to(self, status: str) -> bool:
        return status in _TRANSITIONS.get(self.status, set())

    def move_to(self, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        if not self.can_move_to(status):
            raise ValueError(f"cannot go from {self.status} to {status}")
        self.status = status


def customer_from_record(record: dict) -> Customer:
    """Build a customer from a decoded JSON/CSV row; raises ValidationFailure on bad input."""
    status = str(record.get("status", "active")).lower()
    if status not in STATUSES:
        raise ValidationFailure("status", f"unknown status: {status!r}")
    addr = record.get("address") or {}
    return Customer(
        customer_id=check_identifier(record, "customer_id"),
        name=normalize_name(str(require(record, "name"))),
        email=check_email(record),
        plan_id=check_identifier(record, "plan_id"),
        started=date.fromisoformat(str(require(record, "started"))),
        status=status,
        address=Address(
            line1=addr.get("line1", ""),
            line2=addr.get("line2", ""),
            city=addr.get("city", ""),
            region=addr.get("region", ""),
            postal_code=addr.get("postal_code", ""),
            country=check_country(addr, "country") if addr.get("country") else "US",
        ),
        promo_code=record.get("promo_code") or None,
        credit_balance_cents=int(record.get("credit_balance_cents", 0)),
        tags=set(record.get("tags", [])),
    )


class CustomerRegistry:
    def __init__(self, customers: list[Customer] | None = None):
        self._by_id: dict[str, Customer] = {}
        for c in customers or []:
            self.add(c)

    def add(self, customer: Customer) -> None:
        if customer.customer_id in self._by_id:
            raise ValueError(f"duplicate customer id: {customer.customer_id}")
        self._by_id[customer.customer_id] = customer

    def get(self, customer_id: str) -> Customer:
        try:
            return self._by_id[customer_id]
        except KeyError:
            raise UnknownCustomer(customer_id) from None

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def billable(self) -> list[Customer]:
        return [c for c in self if c.is_billable]

    def on_plan(self, plan_id: str) -> list[Customer]:
        return [c for c in self if c.plan_id == plan_id]

    def with_tag(self, tag: str) -> list[Customer]:
        return [c for c in self if tag in c.tags]

    def search(self, text: str) -> list[Customer]:
        """Case-insensitive match on id, name or email."""
        needle = text.strip().lower()
        if not needle:
            return []
        return [
            c
            for c in self
            if needle in c.customer_id.lower() or needle in c.name.lower() or needle in c.email.lower()
        ]

    def remove(self, customer_id: str) -> Customer:
        customer = self.get(customer_id)
        del self._by_id[customer_id]
        return customer

    @classmethod
    def from_json(cls, path: Path) -> "CustomerRegistry":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([customer_from_record(r) for r in raw])

    def counts_by_status(self) -> dict[str, int]:
        counts = {s: 0 for s in STATUSES}
        for c in self:
            counts[c.status] += 1
        return counts
