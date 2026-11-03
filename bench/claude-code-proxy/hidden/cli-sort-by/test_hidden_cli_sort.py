import pytest

from invoicing import cli
from invoicing.report import format_list


def numbers(capsys, *extra):
    assert cli.main(["list", *extra]) == 0
    return [line.split()[0] for line in capsys.readouterr().out.splitlines()]


def test_default_keeps_file_order(invoice_file, capsys):
    assert numbers(capsys, str(invoice_file)) == ["INV-002", "INV-001", "INV-003"]


@pytest.mark.parametrize(
    "key,expected",
    [
        ("number", ["INV-001", "INV-002", "INV-003"]),
        ("issued", ["INV-001", "INV-002", "INV-003"]),
        ("customer", ["INV-001", "INV-003", "INV-002"]),
        ("total", ["INV-002", "INV-003", "INV-001"]),
    ],
)
def test_sort_by(invoice_file, capsys, key, expected):
    assert numbers(capsys, str(invoice_file), "--sort-by", key) == expected


def test_unknown_key_is_a_usage_error(invoice_file):
    with pytest.raises(SystemExit) as e:
        cli.main(["list", str(invoice_file), "--sort-by", "color"])
    assert e.value.code == 2


def test_format_list_sort_by(sample_invoices):
    lines = format_list(sample_invoices, sort_by="total").splitlines()
    assert [l.split()[0] for l in lines] == ["INV-002", "INV-003", "INV-001"]
    lines = format_list(sample_invoices).splitlines()
    assert [l.split()[0] for l in lines] == ["INV-002", "INV-001", "INV-003"]
