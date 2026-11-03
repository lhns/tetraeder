"""Management reports over many statements."""

from collections import defaultdict
from decimal import Decimal

from legacy.billing.statements import Statement
from legacy.util.money import ZERO, format_amount, total
from legacy.util.text import render_table


def revenue_by_plan(statements: list[Statement], plan_of: dict[str, str]) -> dict[str, Decimal]:
    """Sum of statement totals per plan id (customers missing in `plan_of` go to "unknown")."""
    out: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for st in statements:
        out[plan_of.get(st.customer_id, "unknown")] += st.total
    return dict(out)


def top_customers(statements: list[Statement], n: int = 5) -> list[tuple[str, Decimal]]:
    by_customer: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for st in statements:
        by_customer[st.customer_id] += st.total
    return sorted(by_customer.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def usage_share(statements: list[Statement]) -> dict[str, Decimal]:
    """Fraction of usage revenue per meter name (first word of the usage line description)."""
    sums: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for st in statements:
        for ln in st.of_kind("usage"):
            sums[ln.description.split()[0]] += ln.amount
    whole = total(sums.values())
    if whole == 0:
        return {}
    return {m: v / whole for m, v in sorted(sums.items())}


def discount_rate(statements: list[Statement]) -> Decimal:
    """Total discounts as a share of subtotals."""
    sub = total(s.subtotal for s in statements)
    if sub == 0:
        return ZERO
    return -total(s.discount for s in statements) / sub


def revenue_table(statements: list[Statement], plan_of: dict[str, str]) -> str:
    rows = [[plan, format_amount(amount)] for plan, amount in sorted(revenue_by_plan(statements, plan_of).items())]
    return render_table(rows, ["plan", "revenue"])


def mrr(statements: list[Statement]) -> Decimal:
    """Recurring part of revenue: base fees plus top-ups."""
    return total(ln.amount for st in statements for ln in st.lines if ln.kind in ("base", "topup"))
