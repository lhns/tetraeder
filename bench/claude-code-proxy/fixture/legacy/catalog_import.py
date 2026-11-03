"""Import customers from the CRM export (CSV)."""

import csv
import io
from dataclasses import dataclass, field

from legacy.customers import Address, Customer, CustomerRegistry
from legacy.plans import CATALOG
from legacy.util.text import normalize_name
from legacy.util.validation import (
    ValidationFailure,
    check_choice,
    check_country,
    check_email,
    check_identifier,
    check_positive_int,
    require,
)

COLUMNS = [
    "customer_id",
    "name",
    "email",
    "plan_id",
    "started",
    "status",
    "country",
    "region",
    "postal_code",
    "credit_cents",
]


@dataclass
class ImportReport:
    imported: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.skipped

    def summary(self) -> str:
        return f"imported {self.imported}, skipped {len(self.skipped)}"


def row_to_customer(row: dict[str, str]) -> Customer:
    from datetime import date

    plan = check_identifier(row, "plan_id")
    if plan not in CATALOG:
        raise ValidationFailure("plan_id", f"unknown plan {plan!r}")
    return Customer(
        customer_id=check_identifier(row, "customer_id"),
        name=normalize_name(str(require(row, "name"))),
        email=check_email(row),
        plan_id=plan,
        started=date.fromisoformat(str(require(row, "started"))),
        status=check_choice(row, "status", {"trial", "active", "suspended", "closed"}),
        address=Address(
            country=check_country(row) if row.get("country") else "US",
            region=row.get("region", "").strip(),
            postal_code=row.get("postal_code", "").strip(),
        ),
        credit_balance_cents=check_positive_int(row, "credit_cents", allow_zero=True) if row.get("credit_cents") else 0,
    )


def import_csv(text: str, registry: CustomerRegistry | None = None) -> tuple[CustomerRegistry, ImportReport]:
    registry = registry or CustomerRegistry()
    report = ImportReport()
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in COLUMNS[:5] if c not in (reader.fieldnames or [])]
    if missing:
        report.skipped.append(f"header: missing columns {missing}")
        return registry, report
    for lineno, row in enumerate(reader, start=2):
        try:
            registry.add(row_to_customer(row))
            report.imported += 1
        except (ValidationFailure, ValueError) as e:
            report.skipped.append(f"line {lineno}: {e}")
    return registry, report
