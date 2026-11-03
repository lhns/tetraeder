"""Command line interface: `python -m invoicing.cli <command> ...`."""

import argparse
import sys
from datetime import date
from pathlib import Path

from invoicing.models import Invoice, compute_total
from invoicing.money import format_money
from invoicing.parsing import ParseError, parse_invoice_text
from invoicing.report import format_invoice, format_list
from invoicing.storage import load_invoices
from invoicing.summary import format_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="invoicing")
    sub = p.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="print one invoice")
    show.add_argument("file", type=Path)
    show.add_argument("number")

    lst = sub.add_parser("list", help="one line per invoice")
    lst.add_argument("file", type=Path)

    summ = sub.add_parser("summary", help="totals per customer and overdue invoices")
    summ.add_argument("file", type=Path)
    summ.add_argument("--today", help="ISO date used for overdue checks (default: today)")

    parse = sub.add_parser("parse", help="parse a text invoice and print its total")
    parse.add_argument("file", type=Path)
    return p


def _find(invoices: list[Invoice], number: str) -> Invoice | None:
    return next((i for i in invoices if i.number == number), None)


def cmd_show(args) -> int:
    inv = _find(load_invoices(args.file), args.number)
    if inv is None:
        print(f"no such invoice: {args.number}", file=sys.stderr)
        return 1
    print(format_invoice(inv))
    return 0


def cmd_list(args) -> int:
    print(format_list(load_invoices(args.file)))
    return 0


def cmd_summary(args) -> int:
    today = date.fromisoformat(args.today) if args.today else date.today()
    print(format_summary(load_invoices(args.file), today))
    return 0


def cmd_parse(args) -> int:
    try:
        inv = parse_invoice_text(args.file.read_text(encoding="utf-8"))
    except ParseError as e:
        print(f"parse error: {e}", file=sys.stderr)
        return 2
    print(f"{inv.number}: {format_money(compute_total(inv).total)}")
    return 0


COMMANDS = {"show": cmd_show, "list": cmd_list, "summary": cmd_summary, "parse": cmd_parse}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
