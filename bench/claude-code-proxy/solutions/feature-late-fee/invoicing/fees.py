"""Late fees on overdue invoices."""

from datetime import date

from invoicing.dates import days_overdue
from invoicing.models import Invoice, compute_total
from invoicing.money import apply_rate

PERIOD_DAYS = 30


def late_fee(invoice: Invoice, today: date, monthly_rate_bp: int = 150) -> int:
    """Fee in cents: `monthly_rate_bp` of the grand total for every started 30-day period past the due date."""
    late = days_overdue(invoice.due, today)
    if late == 0:
        return 0
    periods = -(-late // PERIOD_DAYS)
    return periods * apply_rate(compute_total(invoice).total, monthly_rate_bp)
