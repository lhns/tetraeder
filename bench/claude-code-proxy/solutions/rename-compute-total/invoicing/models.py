"""Invoice data model and totals."""

from dataclasses import dataclass, field
from datetime import date

from invoicing.dates import due_date
from invoicing.discounts import Discount, discount_amount
from invoicing.tax import tax_for_line


@dataclass(frozen=True)
class LineItem:
    description: str
    qty: int
    unit_cents: int
    tax_code: str = "std"


@dataclass
class Invoice:
    number: str
    customer: str
    issued: date
    terms: str = "net30"
    discount: Discount | None = None
    items: list[LineItem] = field(default_factory=list)

    @property
    def due(self) -> date:
        return due_date(self.issued, self.terms)


@dataclass(frozen=True)
class Totals:
    subtotal: int
    discount: int
    tax: int
    total: int


def line_net(item: LineItem) -> int:
    return item.qty * item.unit_cents


def invoice_total(invoice: Invoice) -> Totals:
    """Subtotal, discount, tax and grand total of an invoice, in cents.

    The discount is spread over the lines in proportion to their net amount before tax is computed, so a discounted
    invoice is taxed on what the customer actually pays.
    """
    nets = [line_net(i) for i in invoice.items]
    subtotal = sum(nets)
    discount = discount_amount(subtotal, invoice.discount)
    tax = 0
    remaining = discount
    for idx, (item, net) in enumerate(zip(invoice.items, nets)):
        if subtotal == 0:
            share = 0
        elif idx == len(nets) - 1:
            share = remaining
        else:
            share = discount * net // subtotal
        remaining -= share
        tax += tax_for_line(net - share, item.tax_code)
    return Totals(subtotal, discount, tax, subtotal - discount + tax)
