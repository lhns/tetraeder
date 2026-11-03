"""Date parsing and payment-term arithmetic."""

import calendar
import re
from datetime import date, timedelta

_TERMS_RE = re.compile(r"^net(\d{1,3})$")


def parse_date(text: str) -> date:
    """Accept ISO "2024-03-31" and European "31.03.2024"."""
    text = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        y, m, d = (int(p) for p in text.split("-"))
    elif re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", text):
        d, m, y = (int(p) for p in text.split("."))
    else:
        raise ValueError(f"unrecognised date: {text!r}")
    return date(y, m, d)


def add_days(d: date, days: int) -> date:
    return d + timedelta(days=days)


def end_of_month(d: date) -> date:
    last = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last)


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def due_date(issued: date, terms: str) -> date:
    """Due date for terms "net<N>", "eom" (end of month) or "receipt" (due on issue)."""
    terms = terms.strip().lower()
    if terms == "receipt":
        return issued
    if terms == "eom":
        return end_of_month(issued)
    m = _TERMS_RE.match(terms)
    if not m:
        raise ValueError(f"unknown payment terms: {terms!r}")
    return add_days(issued, int(m.group(1)))


def days_overdue(due: date, today: date) -> int:
    """Whole days past the due date; 0 when not yet due."""
    return max(0, (today - due).days)
