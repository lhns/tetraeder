"""Plain-text reports."""

from invoicing.models import Invoice, compute_total
from invoicing.money import format_money


def format_invoice(inv: Invoice) -> str:
    """One invoice as a fixed-width table with a totals block."""
    lines = [
        f"Invoice {inv.number}  customer: {inv.customer}",
        f"Issued {inv.issued.isoformat()}  terms {inv.terms}  due {inv.due.isoformat()}",
        "",
        f"{'Description':<24}{'Qty':>5}{'Unit':>12}{'Net':>14}",
    ]
    for item in inv.items:
        net = item.qty * item.unit_cents
        lines.append(f"{item.description:<24}{item.qty:>5}{format_money(item.unit_cents):>12}{format_money(net):>14}")
    t = compute_total(inv)
    lines.append("")
    lines.append(f"{'Subtotal':<41}{format_money(t.subtotal):>14}")
    if t.discount:
        lines.append(f"{'Discount':<41}{format_money(-t.discount):>14}")
    lines.append(f"{'Tax':<41}{format_money(t.tax):>14}")
    lines.append(f"{'Total':<41}{format_money(t.total):>14}")
    return "\n".join(lines)


SORT_KEYS = ("number", "issued", "customer", "total")


def sort_invoices(invoices: list[Invoice], sort_by: str | None) -> list[Invoice]:
    """number, issued and customer ascending; total largest first; ties by invoice number."""
    if sort_by is None:
        return list(invoices)
    if sort_by == "number":
        return sorted(invoices, key=lambda i: i.number)
    if sort_by == "issued":
        return sorted(invoices, key=lambda i: (i.issued, i.number))
    if sort_by == "customer":
        return sorted(invoices, key=lambda i: (i.customer, i.number))
    if sort_by == "total":
        return sorted(invoices, key=lambda i: (-compute_total(i).total, i.number))
    raise ValueError(f"unknown sort key: {sort_by!r}")


def format_list(invoices: list[Invoice], sort_by: str | None = None) -> str:
    """One line per invoice: number, customer, due date, total."""
    rows = []
    for inv in sort_invoices(invoices, sort_by):
        total = compute_total(inv).total
        rows.append(f"{inv.number:<12}{inv.customer:<20}{inv.due.isoformat():<12}{format_money(total):>14}")
    return "\n".join(rows) if rows else "(no invoices)"
