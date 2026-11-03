"""Which billing period a customer is in and when the next one starts."""

from datetime import date

from legacy.customers import Customer
from legacy.errors import InvalidPeriod
from legacy.util.timeutil import Period, add_months, cycle_containing


def customer_billing_day(customer: Customer, default_day: int) -> int:
    """Customers are billed on their own start day (capped at 28), or the default if tagged `calendar`."""
    if "calendar" in customer.tags:
        return default_day
    return min(customer.started.day, 28)


def current_period(customer: Customer, today: date, default_day: int = 1) -> Period:
    return cycle_containing(today, customer_billing_day(customer, default_day))


def previous_period(period: Period) -> Period:
    return Period(add_months(period.start, -1), period.start)


def next_period(period: Period) -> Period:
    return Period(period.end, add_months(period.end, 1))


def periods_between(first: Period, last_day: date) -> list[Period]:
    """Consecutive monthly periods starting at `first` until one contains `last_day`."""
    if last_day < first.start:
        raise InvalidPeriod("last_day is before the first period")
    out = [first]
    while not out[-1].contains(last_day):
        out.append(next_period(out[-1]))
    return out


def is_first_period(customer: Customer, period: Period) -> bool:
    return period.contains(customer.started)


def trial_ends(customer: Customer, trial_days: int) -> date:
    from datetime import timedelta

    return customer.started + timedelta(days=trial_days)


def in_trial(customer: Customer, trial_days: int, day: date) -> bool:
    return trial_days > 0 and day < trial_ends(customer, trial_days)


def periods_due(customers: list[Customer], today: date, default_day: int = 1) -> dict[str, Period]:
    """The period that just ended for every billable customer whose cycle rolled over today."""
    due = {}
    for c in customers:
        if not c.is_billable:
            continue
        cur = current_period(c, today, default_day)
        if cur.start == today:
            due[c.customer_id] = previous_period(cur)
    return due
