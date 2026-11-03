from datetime import date

import pytest

from invoicing import cli
from invoicing.report import format_invoice, format_list
from invoicing.storage import invoice_from_dict, invoice_to_dict, load_invoices
from invoicing.summary import customer_totals, format_summary, overdue_invoices


def test_storage_round_trip(sample_invoices, invoice_file):
    assert load_invoices(invoice_file) == sample_invoices
    assert invoice_from_dict(invoice_to_dict(sample_invoices[1])) == sample_invoices[1]


def test_storage_rejects_non_list(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        load_invoices(p)


def test_format_invoice_contains_totals(sample_invoices):
    text = format_invoice(sample_invoices[1])
    assert "Invoice INV-001" in text
    assert "due 2024-02-14" in text
    assert "Discount" in text and "-$3.40" in text
    assert text.splitlines()[-1].endswith("$36.72")


def test_format_list(sample_invoices):
    lines = format_list(sample_invoices).splitlines()
    assert [l.split()[0] for l in lines] == ["INV-002", "INV-001", "INV-003"]
    assert format_list([]) == "(no invoices)"


def test_customer_totals_keeps_first_seen_order(sample_invoices):
    totals = customer_totals(sample_invoices)
    assert list(totals) == ["Zeta Ltd", "Acme Corp"]
    assert totals == {"Zeta Ltd": 47_398, "Acme Corp": 3_672 + 4_500}


def test_overdue(sample_invoices):
    late = overdue_invoices(sample_invoices, date(2024, 3, 2))
    assert [i.number for i in late] == ["INV-002", "INV-001", "INV-003"]
    assert overdue_invoices(sample_invoices, date(2024, 1, 1)) == []


def test_format_summary(sample_invoices):
    text = format_summary(sample_invoices, date(2024, 3, 2))
    assert "All customers" in text
    assert "3 overdue" in text


def test_cli_show(invoice_file, capsys):
    assert cli.main(["show", str(invoice_file), "INV-001"]) == 0
    assert "Acme Corp" in capsys.readouterr().out


def test_cli_show_missing(invoice_file, capsys):
    assert cli.main(["show", str(invoice_file), "NOPE"]) == 1
    assert "no such invoice" in capsys.readouterr().err


def test_cli_list_and_summary(invoice_file, capsys):
    assert cli.main(["list", str(invoice_file)]) == 0
    assert "INV-003" in capsys.readouterr().out
    assert cli.main(["summary", str(invoice_file), "--today", "2024-03-02"]) == 0
    assert "overdue" in capsys.readouterr().out


def test_cli_parse(tmp_path, capsys):
    p = tmp_path / "inv.txt"
    p.write_text("Invoice: X-1\nCustomer: C\nIssued: 2024-01-01\nItem: a | 2 | 10.00\n")
    assert cli.main(["parse", str(p)]) == 0
    assert capsys.readouterr().out.strip() == "X-1: $24.00"
    p.write_text("Customer: C\n")
    assert cli.main(["parse", str(p)]) == 2
