"""Aggregates over many invoices."""

from collections import defaultdict
from datetime import date

from invoicing.dates import days_overdue
from invoicing.models import Invoice, invoice_total
from invoicing.money import format_money


def customer_totals(invoices: list[Invoice]) -> dict[str, int]:
    """Grand total in cents per customer, in order of first appearance."""
    totals: dict[str, int] = defaultdict(int)
    for inv in invoices:
        totals[inv.customer] += invoice_total(inv).total
    return dict(totals)


def overdue_invoices(invoices: list[Invoice], today: date) -> list[Invoice]:
    return [inv for inv in invoices if days_overdue(inv.due, today) > 0]


def format_summary(invoices: list[Invoice], today: date) -> str:
    totals = customer_totals(invoices)
    lines = [f"{name:<24}{format_money(cents):>14}" for name, cents in totals.items()]
    lines.append(f"{'All customers':<24}{format_money(sum(totals.values())):>14}")
    late = overdue_invoices(invoices, today)
    if late:
        lines.append("")
        lines.append(f"{len(late)} overdue: " + ", ".join(i.number for i in late))
    return "\n".join(lines)
