# invoicing

Parses simple invoice files, prices them (discounts, tax by code, payment terms) and prints reports.

```
python -m invoicing.cli show invoices.json INV-001
python -m invoicing.cli list invoices.json
python -m invoicing.cli summary invoices.json
python -m invoicing.cli parse invoice.txt
```

Money is always integer cents. Tax rates are in basis points. `legacy/` is the old billing engine that is still used
for usage-based plans; it is not covered by the CLI.

Run the tests with `python -m pytest -q`.
