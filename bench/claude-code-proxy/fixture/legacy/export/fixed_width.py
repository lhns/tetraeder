"""Fixed-width batch file for the accounting mainframe import.

Record layout (80 columns):
  1-1    record type: H header, D detail, T trailer
  2-13   customer id, left aligned
  14-21  period start, YYYYMMDD
  22-29  period end, YYYYMMDD
  30-37  kind, left aligned
  38-67  description, left aligned, cut at 30
  68-79  amount in cents, zero padded, sign in column 68
  80-80  newline
"""

from legacy.billing.statements import Statement
from legacy.errors import ExportError
from legacy.util.money import to_cents
from legacy.util.text import pad_right

WIDTH = 79


def _amount_field(cents: int) -> str:
    if abs(cents) > 99_999_999_999:
        raise ExportError(f"amount does not fit the field: {cents}")
    sign = "-" if cents < 0 else "0"
    return sign + f"{abs(cents):011d}"


def detail_record(st: Statement, index: int) -> str:
    ln = st.lines[index]
    rec = (
        "D"
        + pad_right(st.customer_id, 12)
        + st.period.start.strftime("%Y%m%d")
        + st.period.end.strftime("%Y%m%d")
        + pad_right(ln.kind, 8)
        + pad_right(ln.description, 30)
        + _amount_field(to_cents(ln.amount))
    )
    if len(rec) != WIDTH:
        raise ExportError(f"record has {len(rec)} columns, expected {WIDTH}")
    return rec


def header_record(batch_date: str, count: int) -> str:
    return ("H" + pad_right(batch_date, 8) + f"{count:06d}").ljust(WIDTH)


def trailer_record(total_cents: int, lines: int) -> str:
    return ("T" + f"{lines:06d}" + _amount_field(total_cents)).ljust(WIDTH)


def to_fixed_width(statements: list[Statement], batch_date: str) -> str:
    out = []
    n_lines = sum(len(s.lines) for s in statements)
    out.append(header_record(batch_date, len(statements)))
    grand = 0
    for st in statements:
        for i, ln in enumerate(st.lines):
            out.append(detail_record(st, i))
            grand += to_cents(ln.amount)
    out.append(trailer_record(grand, n_lines))
    return "\n".join(out) + "\n"


def parse_amount(field: str) -> int:
    sign = -1 if field[0] == "-" else 1
    return sign * int(field[1:])
