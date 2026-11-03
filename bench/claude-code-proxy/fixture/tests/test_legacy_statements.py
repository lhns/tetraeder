from datetime import date, datetime, timezone
from decimal import Decimal

from legacy.billing.cycle import current_period, periods_due
from legacy.billing.proration import fraction_active, prorated_base_fee
from legacy.billing.statements import build_statement
from legacy.customers import Address, Customer
from legacy.metering.aggregate import totals_by_meter
from legacy.metering.events import UsageEvent
from legacy.util.timeutil import Period


def ev(i, customer, meter, units, day):
    return UsageEvent(str(i), customer, meter, units, datetime(2024, 5, day, 12, tzinfo=timezone.utc))


MAY = Period(date(2024, 5, 1), date(2024, 6, 1))


def growth_customer(**kw):
    base = dict(customer_id="c1", name="Acme", email="a@acme.io", plan_id="growth", started=date(2023, 1, 1))
    base.update(kw)
    return Customer(**base)


def test_totals_sum_counters_and_peak_gauges():
    events = [
        ev(1, "c1", "api_calls", 100, 2),
        ev(2, "c1", "api_calls", 50, 3),
        ev(3, "c1", "storage_gb", 40, 2),
        ev(4, "c1", "storage_gb", 55, 20),
        ev(5, "c1", "api_calls", 50, 3),
        ev(5, "c1", "api_calls", 50, 3),  # duplicate delivery
    ]
    assert totals_by_meter(events, MAY) == {"api_calls": 200, "storage_gb": 55}


def test_events_outside_period_are_ignored():
    events = [UsageEvent("1", "c1", "api_calls", 10, datetime(2024, 6, 1, tzinfo=timezone.utc))]
    assert totals_by_meter(events, MAY) == {}


def test_statement_for_established_customer():
    events = [ev(1, "c1", "api_calls", 25_000, 5), ev(2, "c1", "storage_gb", 60, 7)]
    st = build_statement(growth_customer(), MAY, events)
    assert st.of_kind("base")[0].amount == Decimal("49.00")
    usage = {ln.description.split()[0]: ln.amount for ln in st.of_kind("usage")}
    assert usage == {"api_calls": Decimal("25.00"), "storage_gb": Decimal("12.50")}
    assert st.total == Decimal("86.50")


def test_statement_with_tax_and_promo():
    c = growth_customer(promo_code="WELCOME10", address=Address(country="DE"))
    st = build_statement(c, MAY, [])
    assert st.discount == Decimal("-4.90")
    assert st.tax == Decimal("8.38")  # 19% of 44.10
    assert st.total == Decimal("52.48")


def test_statement_credit_is_used_up():
    c = growth_customer(credit_balance_cents=3_000)
    st = build_statement(c, MAY, [])
    assert st.credit_applied == Decimal("-30.00")
    assert st.total == Decimal("19.00")


def test_first_period_is_prorated():
    c = growth_customer(started=date(2024, 5, 16))
    assert prorated_base_fee(Decimal("49.00"), MAY, c.started) == Decimal("25.29")
    assert fraction_active(MAY, date(2024, 5, 16)) == Decimal(16) / Decimal(31)


def test_unknown_meter_is_noted_not_charged():
    st = build_statement(growth_customer(), MAY, [ev(1, "c1", "egress_gb", 500, 3)])
    assert st.of_kind("usage") == []
    assert any("egress_gb" in n for n in st.notes)


def test_cycles():
    c = growth_customer(started=date(2023, 3, 15))
    assert current_period(c, date(2024, 5, 20)) == Period(date(2024, 5, 15), date(2024, 6, 15))
    assert current_period(c, date(2024, 5, 3)) == Period(date(2024, 4, 15), date(2024, 5, 15))
    due = periods_due([c], date(2024, 5, 15))
    assert due == {"c1": Period(date(2024, 4, 15), date(2024, 5, 15))}
    assert periods_due([c], date(2024, 5, 16)) == {}
