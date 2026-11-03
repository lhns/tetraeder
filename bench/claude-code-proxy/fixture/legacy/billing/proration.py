"""Proration of fees for periods a customer was only part of."""

from datetime import date
from decimal import Decimal

from legacy.util.money import ZERO, D, quantize_cents
from legacy.util.timeutil import Period


def fraction_active(period: Period, active_from: date | None, active_until: date | None = None) -> Decimal:
    """Share of the period's days in which the customer was active, as a Decimal between 0 and 1.

    `active_from` is the first active day; `active_until` the first day no longer active.
    """
    start = max(period.start, active_from) if active_from else period.start
    end = min(period.end, active_until) if active_until else period.end
    if end <= start:
        return ZERO
    return Decimal((end - start).days) / Decimal(period.days)


def prorate(amount: Decimal, period: Period, active_from: date | None, active_until: date | None = None) -> Decimal:
    return quantize_cents(D(amount) * fraction_active(period, active_from, active_until))


def prorated_base_fee(base_fee: Decimal, period: Period, started: date) -> Decimal:
    """Base fee for a customer who started inside the period."""
    if started <= period.start:
        return quantize_cents(base_fee)
    return prorate(base_fee, period, started)


def refund_for_cancellation(amount: Decimal, period: Period, cancelled_on: date) -> Decimal:
    """Unused part of an already paid amount when the customer cancels mid-period (positive number)."""
    if cancelled_on >= period.end:
        return ZERO
    if cancelled_on <= period.start:
        return quantize_cents(amount)
    unused = Decimal((period.end - cancelled_on).days) / Decimal(period.days)
    return quantize_cents(D(amount) * unused)


def plan_change_adjustment(old_fee: Decimal, new_fee: Decimal, period: Period, change_day: date) -> Decimal:
    """Extra (positive) or credit (negative) for switching plans on `change_day`.

    The old plan is charged until the day before the change, the new plan from the change day on.
    """
    if not period.contains(change_day):
        raise ValueError("change day is outside the period")
    before = fraction_active(period, None, change_day)
    after = fraction_active(period, change_day)
    charged = D(old_fee) * before + D(new_fee) * after
    return quantize_cents(charged - D(old_fee))


def minimum_commit_topup(charged: Decimal, min_commit: Decimal, period: Period, started: date) -> Decimal:
    """What must be added so a customer pays at least the (prorated) minimum commitment."""
    commit = prorated_base_fee(min_commit, period, started)
    return max(ZERO, commit - charged)
