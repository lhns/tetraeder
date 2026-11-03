from datetime import date
from decimal import Decimal

import pytest

from legacy.billing.credits import CreditLedger
from legacy.billing.dunning import schedule, should_suspend
from legacy.catalog_import import import_csv
from legacy.config import Settings, load_settings
from legacy.currency import parse_rates
from legacy.customers import CustomerRegistry, customer_from_record
from legacy.metering.events import events_from_csv
from legacy.metering.quotas import alert_level, default_quotas, new_alerts
from legacy.plans import get_plan
from legacy.util.money import format_amount, split_amount
from legacy.util.timeutil import Period, add_months, cycle_containing
from legacy.util.validation import ValidationFailure

CSV = """customer_id,name,email,plan_id,started,status,country,region,postal_code,credit_cents
c1,  acme   corp ,ops@acme.io,growth,2023-01-05,active,DE,,10115,0
c2,Bad Mail,nope,growth,2023-01-05,active,US,CA,94105,0
c3,Ghost,g@x.io,platinum,2023-01-05,active,US,CA,94105,0
"""


def test_import_csv_reports_bad_rows():
    reg, report = import_csv(CSV)
    assert report.imported == 1 and len(report.skipped) == 2
    assert reg.get("c1").name == "Acme Corp"
    assert "line 3" in report.skipped[0] and "line 4" in report.skipped[1]


def test_registry_search_and_status_transitions():
    reg = CustomerRegistry()
    c = customer_from_record(
        {"customer_id": "c9", "name": "Zed", "email": "z@z.io", "plan_id": "starter", "started": "2024-01-01"}
    )
    reg.add(c)
    assert reg.search("Z@Z") == [c]
    assert not c.can_move_to("trial")
    c.move_to("suspended")
    with pytest.raises(ValueError):
        c.move_to("trial")
    with pytest.raises(ValidationFailure):
        customer_from_record({"customer_id": "c9", "name": "Zed", "email": "x", "plan_id": "starter", "started": "2024-01-01"})


def test_events_csv():
    events, problems = events_from_csv(
        "event_id,customer_id,meter,units,at\n"
        "1,c1,api_calls,10,2024-05-01T10:00:00Z\n"
        "2,c1,bogus,10,2024-05-01T10:00:00Z\n"
        "3,c1,api_calls,-4,2024-05-01T10:00:00Z\n"
    )
    assert len(events) == 1 and len(problems) == 2


def test_quotas():
    q = default_quotas(get_plan("growth"))["api_calls"]
    assert alert_level(2_500, q) == 50
    assert alert_level(4_000, q) == 80
    assert alert_level(10, q) is None
    assert new_alerts(2_000, 5_200, q) == [50, 80, 100]


def test_credit_ledger():
    ledger = CreditLedger("c1")
    ledger.add(date(2024, 1, 1), 500, "gift")
    assert ledger.use(date(2024, 1, 2), 800, "statement") == 500
    assert ledger.balance_cents == 0


def test_dunning_schedule():
    sched = schedule(date(2024, 5, 31), Settings())
    assert [r.template for r in sched] == ["friendly", "firm", "final", "suspension"]
    assert sched[0].send_on == date(2024, 6, 17)
    assert not should_suspend(date(2024, 5, 31), Settings(), date(2024, 7, 1))
    assert should_suspend(date(2024, 5, 31), Settings(), date(2024, 8, 1))


def test_money_and_time_helpers():
    assert format_amount(Decimal("-1234.5")) == "-$1,234.50"
    assert split_amount(Decimal("10.00"), [1, 1, 1]) == [Decimal("3.34"), Decimal("3.33"), Decimal("3.33")]
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert cycle_containing(date(2024, 2, 10), 15) == Period(date(2024, 1, 15), date(2024, 2, 15))


def test_settings_from_env_and_rates():
    s = load_settings({"LEGACY_CURRENCY": "eur", "LEGACY_BILLING_DAY": "40", "LEGACY_PROMOS": "no"})
    assert (s.currency, s.billing_day, s.apply_promos) == ("EUR", 28, False)
    table = parse_rates(["EUR 0.5", "# c"], date(2024, 1, 1))
    assert table.convert(Decimal("10"), "USD", "EUR") == Decimal("5.00")
