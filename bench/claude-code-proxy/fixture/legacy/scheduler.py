"""Billing run: price every billable customer whose cycle ended, record it, queue the e-mails."""

from dataclasses import dataclass, field
from datetime import date

from legacy.audit import AuditLog
from legacy.billing.credits import CreditLedger, ledger_for
from legacy.billing.cycle import periods_due
from legacy.billing.statements import Statement, build_statement
from legacy.config import Settings
from legacy.customers import CustomerRegistry
from legacy.errors import BillingError
from legacy.metering.events import UsageEvent
from legacy.notifications import Email, statement_email


@dataclass
class RunResult:
    day: date
    statements: list[Statement] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        return f"{self.day}: {len(self.statements)} statements, {len(self.failures)} failures"


class BillingRun:
    def __init__(
        self,
        registry: CustomerRegistry,
        settings: Settings | None = None,
        audit: AuditLog | None = None,
    ):
        self.registry = registry
        self.settings = settings or Settings()
        self.audit = audit or AuditLog(enabled=self.settings.audit_enabled)
        self.ledgers: dict[str, CreditLedger] = {}

    def ledger(self, customer_id: str, opened: date) -> CreditLedger:
        if customer_id not in self.ledgers:
            self.ledgers[customer_id] = ledger_for(self.registry.get(customer_id), opened)
        return self.ledgers[customer_id]

    def run(self, day: date, events: list[UsageEvent]) -> RunResult:
        """Bill every customer whose cycle rolled over on `day`. One bad customer does not stop the others."""
        result = RunResult(day)
        due = periods_due(self.registry.billable(), day, self.settings.billing_day)
        for customer_id, period in sorted(due.items()):
            customer = self.registry.get(customer_id)
            try:
                st = build_statement(
                    customer, period, events, self.settings, self.ledger(customer_id, period.start), day
                )
            except BillingError as e:
                result.failures[customer_id] = str(e)
                self.audit.record("scheduler", "statement_failed", customer_id, error=str(e))
                continue
            result.statements.append(st)
            result.emails.append(statement_email(customer, st))
            self.audit.record("scheduler", "statement_built", customer_id, total=str(st.total), period=period.label())
        return result

    def rerun_failures(self, previous: RunResult, events: list[UsageEvent]) -> RunResult:
        """Run again for just the customers that failed (after fixing their data)."""
        retry = RunResult(previous.day)
        for customer_id in sorted(previous.failures):
            customer = self.registry.get(customer_id)
            due = periods_due([customer], previous.day, self.settings.billing_day)
            if customer_id not in due:
                continue
            try:
                st = build_statement(
                    customer, due[customer_id], events, self.settings, self.ledger(customer_id, previous.day), previous.day
                )
            except BillingError as e:
                retry.failures[customer_id] = str(e)
                continue
            retry.statements.append(st)
            retry.emails.append(statement_email(customer, st))
        return retry
