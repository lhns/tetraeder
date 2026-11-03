# How the invoice text parser works

The parser lives in `invoicing/parsing.py`. It turns a plain-text invoice into an `Invoice` model object
(`invoicing/models.py`). The only public entry point is `parse_invoice_text(text)`.

## Input format

One `Key: value` pair per line. Blank lines and lines starting with `#` are ignored. Keys are case-insensitive
(they are title-cased before use).

```
Invoice: INV-001
Customer: Acme Corp
Issued: 2024-01-15
Terms: net30
Discount: 10%
Item: Widget | 2 | $4.50 | std
```

`Item` may appear many times. An item has three or four fields separated by `|`: description, quantity, unit
price and an optional tax code (default `std`).

## Flow

1. `split_headers(text)` walks the lines. It returns a dict of header fields and a list of raw item strings. A line
   without a colon raises `ParseError("line N: ...")`.
2. `parse_invoice_text` requires at least one item, then builds the `Invoice` from the small field parsers:
   - `parse_number` (the `Invoice` header) and `parse_customer` are required, stripped strings.
   - `parse_issued` is required and parsed with `dates.parse_date`.
   - `parse_terms` defaults to `net30` and must start with `net`, `eom` or `receipt`.
   - `parse_discount_field` is optional and uses `discounts.parse_discount` ("10%" or "$5.00").
   - `parse_item` splits one item line, checks the quantity is a positive integer and converts the price with
     `money.parse_money` (money is integer cents).

## Errors

Everything the parser rejects is reported as `ParseError`, a `ValueError` subclass. Missing required headers say
`missing field: <name>`. Errors from the date, money and discount helpers are wrapped, so callers only need to
catch `ParseError`. The CLI (`invoicing parse FILE`) prints it and exits with status 2.
