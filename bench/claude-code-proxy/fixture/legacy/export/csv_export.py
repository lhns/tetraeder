"""CSV export of statements."""

import csv
import io

from legacy.billing.statements import Statement
from legacy.util.money import quantize

HEADER = ["customer_id", "period_start", "period_end", "kind", "description", "quantity", "amount", "currency"]


def statement_rows(st: Statement) -> list[list[str]]:
    rows = []
    for ln in st.lines:
        rows.append(
            [
                st.customer_id,
                st.period.start.isoformat(),
                st.period.end.isoformat(),
                ln.kind,
                ln.description,
                "" if ln.quantity is None else str(ln.quantity),
                str(quantize(ln.amount, st.currency)),
                st.currency,
            ]
        )
    return rows


def to_csv(statements: list[Statement]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(HEADER)
    for st in statements:
        writer.writerows(statement_rows(st))
    return buf.getvalue()


def totals_csv(statements: list[Statement]) -> str:
    """One row per statement with subtotal, discount, tax, credit and total."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["customer_id", "period_start", "subtotal", "discount", "tax", "credit", "total", "currency"])
    for st in statements:
        writer.writerow(
            [
                st.customer_id,
                st.period.start.isoformat(),
                quantize(st.subtotal, st.currency),
                quantize(st.discount, st.currency),
                quantize(st.tax, st.currency),
                quantize(st.credit_applied, st.currency),
                quantize(st.total, st.currency),
                st.currency,
            ]
        )
    return buf.getvalue()


def parse_totals_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))
