"""Text helpers for statements and exports."""

import re
import textwrap
import unicodedata


def slugify(text: str) -> str:
    """Lowercase ASCII slug with single dashes."""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")


def pad_right(text: str, width: int) -> str:
    return text[:width].ljust(width)


def pad_left(text: str, width: int) -> str:
    return text[-width:].rjust(width) if len(text) > width else text.rjust(width)


def wrap_lines(text: str, width: int = 72) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def plural(n: int, singular: str, plural_form: str | None = None) -> str:
    word = singular if n == 1 else (plural_form or singular + "s")
    return f"{n} {word}"


def mask(value: str, keep: int = 4, char: str = "*") -> str:
    """Hide all but the last `keep` characters (account numbers in logs)."""
    if len(value) <= keep:
        return char * len(value)
    return char * (len(value) - keep) + value[-keep:]


def normalize_name(name: str) -> str:
    return " ".join(name.split()).title()


def truncate(text: str, limit: int, ellipsis: str = "...") -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - len(ellipsis), 0)] + ellipsis


def render_table(rows: list[list[str]], header: list[str] | None = None, gap: int = 2) -> str:
    """Left-aligned fixed-width table."""
    all_rows = ([header] if header else []) + rows
    if not all_rows:
        return ""
    widths = [max(len(r[c]) for r in all_rows if c < len(r)) for c in range(max(len(r) for r in all_rows))]
    lines = []
    for r in all_rows:
        lines.append((" " * gap).join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip())
    return "\n".join(lines)
