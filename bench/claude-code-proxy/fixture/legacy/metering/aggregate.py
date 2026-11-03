"""Roll usage events up into per-meter totals for a period."""

from collections import defaultdict
from datetime import date

from legacy.metering.events import UsageEvent, dedupe, sort_events
from legacy.util.timeutil import Period


def in_period(events: list[UsageEvent], period: Period) -> list[UsageEvent]:
    return [e for e in events if period.contains(e.day)]


def daily_totals(events: list[UsageEvent], period: Period) -> dict[str, dict[date, int]]:
    """meter -> day -> value. Counters add up within a day; gauges keep the day's highest reading."""
    out: dict[str, dict[date, int]] = defaultdict(dict)
    for e in in_period(dedupe(events), period):
        days = out[e.meter]
        if e.is_gauge:
            days[e.day] = max(days.get(e.day, 0), e.units)
        else:
            days[e.day] = days.get(e.day, 0) + e.units
    return {meter: dict(sorted(days.items())) for meter, days in out.items()}


def totals_by_meter(events: list[UsageEvent], period: Period) -> dict[str, int]:
    """Billable usage per meter.

    Counters (api_calls, egress_gb) are summed over the period. Gauges (storage_gb, seats) bill the peak daily
    reading of the period.
    """
    result: dict[str, int] = {}
    for meter, days in daily_totals(events, period).items():
        if not days:
            continue
        if meter in _gauge_names(events):
            result[meter] = max(days.values())
        else:
            result[meter] = sum(days.values())
    return result


def _gauge_names(events: list[UsageEvent]) -> set[str]:
    return {e.meter for e in events if e.is_gauge}


def peak_day(events: list[UsageEvent], period: Period, meter: str) -> tuple[date, int] | None:
    days = daily_totals(events, period).get(meter)
    if not days:
        return None
    day = max(days, key=lambda d: (days[d], -d.toordinal()))
    return day, days[day]


def average_daily(events: list[UsageEvent], period: Period, meter: str) -> float:
    """Mean per calendar day of the period (days without events count as zero)."""
    days = daily_totals(events, period).get(meter, {})
    return sum(days.values()) / period.days


def active_days(events: list[UsageEvent], period: Period) -> int:
    days = set()
    for by_day in daily_totals(events, period).values():
        days.update(by_day)
    return len(days)


def merge_totals(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Add two counter-style total dicts."""
    merged = dict(a)
    for meter, units in b.items():
        merged[meter] = merged.get(meter, 0) + units
    return merged


def sorted_events_for_audit(events: list[UsageEvent], period: Period) -> list[UsageEvent]:
    return sort_events(in_period(dedupe(events), period))
