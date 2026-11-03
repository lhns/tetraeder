import importlib

import pytest

from invoicing.models import Totals, invoice_total


@pytest.mark.parametrize("module", ["models", "report", "summary", "cli"])
def test_new_name_used_everywhere(module):
    mod = importlib.import_module(f"invoicing.{module}")
    assert hasattr(mod, "invoice_total")
    assert not hasattr(mod, "compute_total")


def test_invoice_total_still_works(sample_invoices):
    t = invoice_total(sample_invoices[1])
    assert t == Totals(3400, 340, 612, 3672)
