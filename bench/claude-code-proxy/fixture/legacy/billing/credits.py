"""Customer credit balances: prepaid credits and goodwill adjustments that reduce statements."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from legacy.customers import Customer
from legacy.util.money import ZERO, from_cents, to_cents


@dataclass(frozen=True)
class CreditEntry:
    day: date
    cents: int  # positive = credit added, negative = credit used
    reason: str


@dataclass
class CreditLedger:
    customer_id: str
    entries: list[CreditEntry] = field(default_factory=list)

    @property
    def balance_cents(self) -> int:
        return sum(e.cents for e in self.entries)

    def add(self, day: date, cents: int, reason: str) -> None:
        if cents <= 0:
            raise ValueError("credit must be positive")
        self.entries.append(CreditEntry(day, cents, reason))

    def use(self, day: date, cents: int, reason: str) -> int:
        """Use up to `cents`; returns what was actually used."""
        if cents < 0:
            raise ValueError("amount must not be negative")
        used = min(cents, self.balance_cents)
        if used:
            self.entries.append(CreditEntry(day, -used, reason))
        return used

    def history(self, since: date | None = None) -> list[CreditEntry]:
        return [e for e in self.entries if since is None or e.day >= since]


def ledger_for(customer: Customer, opened: date) -> CreditLedger:
    ledger = CreditLedger(customer.customer_id)
    if customer.credit_balance_cents > 0:
        ledger.add(opened, customer.credit_balance_cents, "opening balance")
    return ledger


def apply_credit(ledger: CreditLedger, day: date, amount: Decimal, reason: str = "statement") -> tuple[Decimal, Decimal]:
    """Take credit off a statement amount. Returns (credit applied, amount still due)."""
    if amount <= 0:
        return ZERO, amount
    used = ledger.use(day, to_cents(amount), reason)
    applied = from_cents(used)
    return applied, amount - applied


def expire_old_credits(ledger: CreditLedger, today: date, max_age_days: int = 365) -> int:
    """Remove credits older than `max_age_days` that were not used; returns the cents that expired."""
    expired = 0
    kept: list[CreditEntry] = []
    for e in ledger.entries:
        if e.cents > 0 and (today - e.day).days > max_age_days and ledger.balance_cents > 0:
            expired += min(e.cents, ledger.balance_cents)
            kept.append(CreditEntry(today, -min(e.cents, ledger.balance_cents), f"expired: {e.reason}"))
        else:
            kept.append(e)
    ledger.entries = kept
    return expired
