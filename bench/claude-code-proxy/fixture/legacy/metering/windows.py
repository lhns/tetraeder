"""Sliding and tumbling windows over a billing period, used by the usage reports and rate alerts."""

from dataclasses import dataclass
from datetime import date, timedelta

from legacy.metering.events import UsageEvent
from legacy.util.timeutil import Period


@dataclass(frozen=True)
class Window:
    period: Period
    total: int

    @property
    def mean_per_day(self) -> float:
        return self.total / self.period.days


def tumbling(period: Period, size_days: int) -> list[Period]:
    """Consecutive windows of `size_days`; the last one may be shorter."""
    if size_days < 1:
        raise ValueError("size_days must be at least 1")
    windows = []
    start = period.start
    while start < period.end:
        end = min(start + timedelta(days=size_days), period.end)
        windows.append(Period(start, end))
        start = end
    return windows


def count_in_window(events: list[UsageEvent], window: Period, meter: str) -> int:
    return sum(e.units for e in events if e.meter == meter and window.contains(e.day))


def tumbling_totals(events: list[UsageEvent], period: Period, size_days: int, meter: str) -> list[Window]:
    return [Window(w, count_in_window(events, w, meter)) for w in tumbling(period, size_days)]


def busiest_window(events: list[UsageEvent], period: Period, size_days: int, meter: str) -> Window | None:
    windows = tumbling_totals(events, period, size_days, meter)
    best = max(windows, key=lambda w: w.total, default=None)
    return best if best is not None and best.total > 0 else None


def sliding_max(events: list[UsageEvent], period: Period, size_days: int, meter: str) -> int:
    """Highest total over any run of `size_days` consecutive days inside the period."""
    by_day: dict[date, int] = {}
    for e in events:
        if e.meter == meter and period.contains(e.day):
            by_day[e.day] = by_day.get(e.day, 0) + e.units
    best = 0
    day = period.start
    while day < period.end:
        total = sum(by_day.get(day + timedelta(days=i), 0) for i in range(size_days))
        best = max(best, total)
        day += timedelta(days=1)
    return best


def exceeds_rate(events: list[UsageEvent], period: Period, meter: str, per_day_limit: int) -> list[date]:
    """Days on which the meter's counter total went over `per_day_limit`."""
    by_day: dict[date, int] = {}
    for e in events:
        if e.meter == meter and period.contains(e.day):
            by_day[e.day] = by_day.get(e.day, 0) + e.units
    return sorted(d for d, n in by_day.items() if n > per_day_limit)
