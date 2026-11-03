"""Build a customer's statement for one billing period."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from legacy.billing import proration, taxes
from legacy.billing.credits import CreditLedger, apply_credit, ledger_for
from legacy.config import Settings
from legacy.customers import Customer
from legacy.metering.aggregate import totals_by_meter
from legacy.metering.events import UsageEvent
from legacy.plans import Plan, get_plan
from legacy.pricing.engine import MeterCharge, price_all, unbilled_meters
from legacy.pricing.promos import apply_promos
from legacy.util.money import ZERO, quantize, total
from legacy.util.timeutil import Period


@dataclass(frozen=True)
class StatementLine:
    kind: str  # base, usage, topup, promo, tax, credit
    description: str
    amount: Decimal
    quantity: int | None = None


@dataclass
class Statement:
    customer_id: str
    period: Period
    currency: str
    lines: list[StatementLine] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[StatementLine]:
        return [ln for ln in self.lines if ln.kind == kind]

    @property
    def subtotal(self) -> Decimal:
        return total(ln.amount for ln in self.lines if ln.kind in ("base", "usage", "topup"))

    @property
    def discount(self) -> Decimal:
        return total(ln.amount for ln in self.lines if ln.kind == "promo")

    @property
    def tax(self) -> Decimal:
        return total(ln.amount for ln in self.lines if ln.kind == "tax")

    @property
    def credit_applied(self) -> Decimal:
        return total(ln.amount for ln in self.lines if ln.kind == "credit")

    @property
    def total(self) -> Decimal:
        return total(ln.amount for ln in self.lines)


def _usage_lines(charges: list[MeterCharge], plan: Plan) -> list[StatementLine]:
    lines = []
    for c in charges:
        if c.amount == 0 and c.used == 0:
            continue
        desc = f"{c.meter} ({c.used} used, {c.included} included, {c.mode})"
        lines.append(StatementLine("usage", desc, c.amount, c.billed_units))
    return lines


def build_statement(
    customer: Customer,
    period: Period,
    events: list[UsageEvent],
    settings: Settings | None = None,
    ledger: CreditLedger | None = None,
    today: date | None = None,
) -> Statement:
    """Price one period for one customer.

    Order: prorated base fee, usage per meter, minimum-commit top-up, promo discount on the sum, tax on the
    discounted sum, then credit. The result is never negative; unused credit stays in the ledger.
    """
    settings = settings or Settings()
    plan = get_plan(customer.plan_id)
    st = Statement(customer.customer_id, period, plan.currency)
    mine = [e for e in events if e.customer_id == customer.customer_id]

    if settings.prorate_first_period:
        base = proration.prorated_base_fee(plan.base_fee, period, customer.started)
    else:
        base = quantize(plan.base_fee, plan.currency)
    if base:
        st.lines.append(StatementLine("base", f"{plan.name} base fee", base))

    usage = totals_by_meter(mine, period)
    charges = price_all(plan.meters, usage)
    st.lines.extend(_usage_lines(charges, plan))
    for name in unbilled_meters(plan.meters, usage):
        st.notes.append(f"meter {name} is not part of plan {plan.plan_id}; not charged")

    if plan.min_commit:
        topup = proration.minimum_commit_topup(st.subtotal, plan.min_commit, period, customer.started)
        if topup:
            st.lines.append(StatementLine("topup", "Minimum commitment top-up", topup))

    if settings.apply_promos and customer.promo_code:
        codes = [customer.promo_code]
        discount, applied = apply_promos(codes, plan.plan_id, period.start, st.subtotal)
        if applied:
            st.lines.append(StatementLine("promo", "Promo " + ", ".join(applied), -discount))
        else:
            st.notes.append(f"promo {customer.promo_code} not applicable")

    percent = taxes.tax_percent(customer.address)
    tax = taxes.tax_amount(st.subtotal + st.discount, percent)
    if tax:
        st.lines.append(StatementLine("tax", taxes.describe(customer.address, percent), tax))

    ledger = ledger or ledger_for(customer, period.start)
    applied_credit, _ = apply_credit(ledger, today or period.end, st.total)
    if applied_credit:
        st.lines.append(StatementLine("credit", "Credit applied", -applied_credit))

    if st.total < settings.minimum_total:
        st.notes.append("total below minimum; clamped")
    return st
