"""Manual adjustments entered by support (goodwill credits, corrections)."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from legacy.billing.statements import Statement, StatementLine
from legacy.util.money import D, quantize_cents

MAX_ADJUSTMENT = Decimal("5000.00")


@dataclass(frozen=True)
class Adjustment:
    customer_id: str
    day: date
    amount: Decimal  # negative = reduces the statement
    reason: str
    approved_by: str | None = None

    def needs_approval(self) -> bool:
        return abs(self.amount) > Decimal("250.00")


class AdjustmentRejected(ValueError):
    pass


def validate(adj: Adjustment) -> None:
    if not adj.reason.strip():
        raise AdjustmentRejected("a reason is required")
    if adj.amount == 0:
        raise AdjustmentRejected("zero adjustment")
    if abs(adj.amount) > MAX_ADJUSTMENT:
        raise AdjustmentRejected(f"above the limit of {MAX_ADJUSTMENT}")
    if adj.needs_approval() and not adj.approved_by:
        raise AdjustmentRejected("approval required above 250.00")


def apply_adjustments(st: Statement, adjustments: list[Adjustment]) -> Statement:
    """Add adjustments for this statement's customer and period as lines. Returns the same statement."""
    for adj in adjustments:
        if adj.customer_id != st.customer_id or not st.period.contains(adj.day):
            continue
        validate(adj)
        st.lines.append(StatementLine("adjustment", f"Adjustment: {adj.reason}", quantize_cents(D(adj.amount))))
    return st


def adjustments_total(adjustments: list[Adjustment]) -> Decimal:
    return sum((a.amount for a in adjustments), Decimal("0"))
