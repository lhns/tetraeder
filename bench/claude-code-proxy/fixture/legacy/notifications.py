"""E-mail texts for statements and reminders."""

from dataclasses import dataclass
from string import Template

from legacy.billing.dunning import Reminder
from legacy.billing.statements import Statement
from legacy.customers import Customer
from legacy.util.money import format_amount
from legacy.util.text import wrap_lines

_TEMPLATES = {
    "statement": Template(
        "Hello $name,\n\nyour statement for $period is ready. The total is $total.\n"
        "$notes\nThank you for using our service."
    ),
    "friendly": Template(
        "Hello $name,\n\na friendly reminder: the statement for $period ($total) is still open."
    ),
    "firm": Template(
        "Dear $name,\n\nthe statement for $period ($total) is overdue. Please pay it this week."
    ),
    "final": Template(
        "Dear $name,\n\nthis is the final reminder for the statement for $period ($total). "
        "Your account will be suspended if we do not receive payment."
    ),
    "suspension": Template(
        "Dear $name,\n\nyour account was suspended because the statement for $period ($total) is unpaid."
    ),
}


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    body: str


def render(name: str, **values: str) -> str:
    try:
        text = _TEMPLATES[name].substitute(**values)
    except KeyError as e:
        raise ValueError(f"template {name!r} needs value {e.args[0]!r}") from None
    return "\n".join(wrap_lines(text))


def statement_email(customer: Customer, st: Statement) -> Email:
    notes = "\n".join(f"Note: {n}" for n in st.notes)
    body = render(
        "statement",
        name=customer.name,
        period=st.period.label(),
        total=format_amount(st.total, st.currency),
        notes=notes,
    )
    return Email(customer.email, f"Statement {st.period.label()}", body)


def reminder_email(customer: Customer, st: Statement, reminder: Reminder) -> Email:
    body = render(
        reminder.template,
        name=customer.name,
        period=st.period.label(),
        total=format_amount(st.total, st.currency),
    )
    return Email(customer.email, f"Reminder {reminder.step}: statement {st.period.label()}", body)
