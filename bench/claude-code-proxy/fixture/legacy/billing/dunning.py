"""Payment reminders for unpaid statements."""

from dataclasses import dataclass
from datetime import date, timedelta

from legacy.config import Settings


@dataclass(frozen=True)
class Reminder:
    step: int  # 1-based
    send_on: date
    template: str


TEMPLATES = ("friendly", "firm", "final", "suspension")


def due_date(statement_end: date, grace_days: int) -> date:
    return statement_end + timedelta(days=grace_days)


def schedule(statement_end: date, settings: Settings) -> list[Reminder]:
    """All reminders for a statement: each dunning step counts days after the due date."""
    due = due_date(statement_end, settings.grace_days)
    out = []
    for i, days in enumerate(settings.dunning_steps):
        template = TEMPLATES[min(i, len(TEMPLATES) - 1)]
        out.append(Reminder(i + 1, due + timedelta(days=days), template))
    return out


def next_reminder(statement_end: date, settings: Settings, today: date, already_sent: int = 0) -> Reminder | None:
    for r in schedule(statement_end, settings):
        if r.step > already_sent and r.send_on >= today:
            return r
    return None


def reminders_due(statement_end: date, settings: Settings, today: date, already_sent: int = 0) -> list[Reminder]:
    """Reminders whose date has passed and that were not sent yet, oldest first."""
    return [r for r in schedule(statement_end, settings) if r.step > already_sent and r.send_on <= today]


def should_suspend(statement_end: date, settings: Settings, today: date) -> bool:
    """Suspend once the last reminder date has passed."""
    sched = schedule(statement_end, settings)
    return bool(sched) and today > sched[-1].send_on


def days_late(statement_end: date, settings: Settings, today: date) -> int:
    return max(0, (today - due_date(statement_end, settings.grace_days)).days)
