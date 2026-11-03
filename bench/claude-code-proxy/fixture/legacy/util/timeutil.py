"""Calendar helpers for billing periods."""

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from legacy.errors import InvalidPeriod


@dataclass(frozen=True)
class Period:
    """Half-open date range [start, end): the end day belongs to the next period."""

    start: date
    end: date

    def __post_init__(self):
        if self.end <= self.start:
            raise InvalidPeriod(f"empty or reversed period: {self.start} .. {self.end}")

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def contains(self, d: date) -> bool:
        return self.start <= d < self.end

    def overlap(self, other: "Period") -> "Period | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Period(start, end) if start < end else None

    def label(self) -> str:
        return f"{self.start.isoformat()}..{(self.end - timedelta(days=1)).isoformat()}"


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def add_months(d: date, months: int) -> date:
    """Move `d` by whole months, clamping the day to the target month's length."""
    index = d.year * 12 + (d.month - 1) + months
    year, month0 = divmod(index, 12)
    month = month0 + 1
    return date(year, month, min(d.day, days_in_month(year, month)))


def month_start(d: date) -> date:
    return d.replace(day=1)


def next_month_start(d: date) -> date:
    return add_months(month_start(d), 1)


def period_for_month(year: int, month: int) -> Period:
    start = date(year, month, 1)
    return Period(start, next_month_start(start))


def cycle_containing(d: date, billing_day: int) -> Period:
    """The monthly cycle that contains `d` when cycles start on `billing_day` (1..28)."""
    if not 1 <= billing_day <= 28:
        raise InvalidPeriod(f"billing day must be 1..28, got {billing_day}")
    start = d.replace(day=billing_day)
    if d.day < billing_day:
        start = add_months(start, -1)
    return Period(start, add_months(start, 1))


def parse_timestamp(text: str) -> datetime:
    """ISO 8601 with optional trailing Z; naive values are taken as UTC."""
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_date(ts: datetime) -> date:
    return ts.astimezone(timezone.utc).date()


def iter_days(period: Period):
    d = period.start
    while d < period.end:
        yield d
        d += timedelta(days=1)


def whole_months_between(a: date, b: date) -> int:
    """Complete months from a to b (b >= a)."""
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day:
        months -= 1
    return max(months, 0)
