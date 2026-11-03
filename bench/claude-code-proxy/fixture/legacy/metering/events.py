"""Usage events as they arrive from the collectors."""

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime

from legacy.errors import InvalidUsage
from legacy.util.timeutil import parse_timestamp, to_date

KNOWN_METERS = ("api_calls", "storage_gb", "seats", "egress_gb")

# meters that report a level (how much is in use) rather than a count of things that happened
GAUGE_METERS = frozenset({"storage_gb", "seats"})


@dataclass(frozen=True)
class UsageEvent:
    event_id: str
    customer_id: str
    meter: str
    units: int
    at: datetime

    @property
    def day(self) -> date:
        return to_date(self.at)

    @property
    def is_gauge(self) -> bool:
        return self.meter in GAUGE_METERS


def event_from_dict(record: dict) -> UsageEvent:
    """Validate and convert one decoded record; raises InvalidUsage."""
    try:
        event_id = str(record["event_id"]).strip()
        customer_id = str(record["customer_id"]).strip()
        meter = str(record["meter"]).strip().lower()
        raw_units = record["units"]
        at = parse_timestamp(str(record["at"]))
    except KeyError as e:
        raise InvalidUsage(f"missing field {e.args[0]!r}") from None
    except ValueError as e:
        raise InvalidUsage(f"bad timestamp: {e}") from None
    if not event_id or not customer_id:
        raise InvalidUsage("event_id and customer_id must not be empty")
    if meter not in KNOWN_METERS:
        raise InvalidUsage(f"unknown meter: {meter!r}")
    try:
        units = int(raw_units)
    except (TypeError, ValueError):
        raise InvalidUsage(f"units is not an integer: {raw_units!r}") from None
    if units < 0:
        raise InvalidUsage(f"negative units: {units}")
    return UsageEvent(event_id, customer_id, meter, units, at)


def events_from_csv(text: str) -> tuple[list[UsageEvent], list[str]]:
    """Parse a collector CSV with header event_id,customer_id,meter,units,at.

    Bad rows are collected as "line N: message" instead of aborting the whole feed.
    """
    events: list[UsageEvent] = []
    problems: list[str] = []
    reader = csv.DictReader(io.StringIO(text))
    for lineno, row in enumerate(reader, start=2):
        try:
            events.append(event_from_dict(row))
        except InvalidUsage as e:
            problems.append(f"line {lineno}: {e}")
    return events, problems


def dedupe(events: list[UsageEvent]) -> list[UsageEvent]:
    """Drop repeated deliveries: the first event with a given id wins."""
    seen: set[str] = set()
    unique = []
    for e in events:
        if e.event_id in seen:
            continue
        seen.add(e.event_id)
        unique.append(e)
    return unique


def for_customer(events: list[UsageEvent], customer_id: str) -> list[UsageEvent]:
    return [e for e in events if e.customer_id == customer_id]


def sort_events(events: list[UsageEvent]) -> list[UsageEvent]:
    return sorted(events, key=lambda e: (e.at, e.event_id))


def latest(events: list[UsageEvent], meter: str) -> UsageEvent | None:
    candidates = [e for e in events if e.meter == meter]
    return max(candidates, key=lambda e: (e.at, e.event_id), default=None)
