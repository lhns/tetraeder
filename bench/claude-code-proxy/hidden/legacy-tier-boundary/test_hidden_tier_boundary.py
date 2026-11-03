from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from legacy.billing.statements import build_statement
from legacy.customers import Customer
from legacy.metering.events import UsageEvent
from legacy.util.timeutil import Period

MAY = Period(date(2024, 5, 1), date(2024, 6, 1))


def storage_line(plan, meter, units):
    c = Customer("c1", "Acme", "a@acme.io", plan, date(2023, 1, 1))
    ev = UsageEvent("1", "c1", meter, units, datetime(2024, 5, 10, tzinfo=timezone.utc))
    st = build_statement(c, MAY, [ev])
    (line,) = [ln for ln in st.of_kind("usage") if ln.description.startswith(meter)]
    return line.amount


@pytest.mark.parametrize(
    "plan,meter,used,expected",
    [
        ("growth", "storage_gb", 110, "25.00"),  # 100 billable GB, first tier limit, 0.25
        ("growth", "storage_gb", 111, "20.20"),  # 101 billable GB, second tier, 0.20
        ("growth", "storage_gb", 1010, "200.00"),  # 1000 billable GB at 0.20
        ("scale", "storage_gb", 600, "90.00"),  # 500 billable at 0.18
        ("scale", "egress_gb", 1000, "90.00"),  # 1000 at 0.09
        ("scale", "egress_gb", 1001, "70.07"),  # 1001 at 0.07
        ("enterprise", "storage_gb", 3000, "240.00"),  # 2000 billable at 0.12
    ],
)
def test_usage_at_a_tier_limit_gets_that_tiers_price(plan, meter, used, expected):
    assert storage_line(plan, meter, used) == Decimal(expected)
