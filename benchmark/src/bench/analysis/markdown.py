"""Markdown tables, so a note or a chapter can quote the store directly."""


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        if value and (abs(value) >= 1e6 or abs(value) < 10 ** -(digits + 1)):
            return f"{value:.{digits}g}"
        return f"{value:,.{digits}f}".replace(",", " ")
    return str(value)


def pct(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f} %"


def table(headers, rows, align=None):
    """A GitHub-flavoured Markdown table. `align` is a string of l/r/c."""
    align = align or "l" * len(headers)
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in cells)) if cells else len(headers[i])
        for i in range(len(headers))
    ]
    rule = {
        "l": lambda w: "-" * max(3, w),
        "r": lambda w: "-" * max(2, w - 1) + ":",
        "c": lambda w: ":" + "-" * max(1, w - 2) + ":",
    }
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    lines.append("|" + "|".join(rule[align[i]](widths[i] + 2) for i in range(len(headers))) + "|")
    for row in cells:
        lines.append(
            "| "
            + " | ".join(
                row[i].rjust(widths[i]) if align[i] == "r" else row[i].ljust(widths[i])
                for i in range(len(headers))
            )
            + " |"
        )
    return "\n".join(lines)


def section(title, body, level=3):
    return f"{'#' * level} {title}\n\n{body}\n"
