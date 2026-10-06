import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deviations import manifest as mf  # noqa: E402

MANIFEST_PATH = mf.default_manifest_path()


@pytest.fixture(scope="session")
def manifest():
    return mf.load(MANIFEST_PATH)


@pytest.fixture
def spec_dict(tmp_path):
    return {
        "version": 1,
        "campaign": "unit-test",
        "image": "sota-deps:latest",
        "methods": ["ACMH", "ACMM"],
        "scenes": ["courtyard"],
        "widths": [1600, 3200],
        "configurations": ["author"],
        "repeats": 2,
        "timeout_seconds": 600,
        "gpu_index": 0,
        "order_seed": 7,
        "paths": {
            "dataset_root": str(tmp_path / "datasets"),
            "results_root": str(tmp_path / "results"),
            "work_root": str(tmp_path / "work"),
            "artifacts_root": str(tmp_path / "clouds"),
        },
    }


@pytest.fixture
def write_spec(tmp_path):
    def _write(data, name="spec.yaml"):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write


def braces_balance(text):
    """(open, close) counts of unescaped braces, comments removed.

    No TeX distribution is installed on the harness host, so compiling a
    generated table cannot be part of the test suite; a brace count catches the
    failure mode such a table actually has -- an unescaped cell running into
    the surrounding markup.
    """
    stripped = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    stripped = stripped.replace("\\\\", "").replace("\\{", "").replace("\\}", "")
    return stripped.count("{"), stripped.count("}")


# -- Typst table data -------------------------------------------------------

# The typographic tokens a JSON cell spells as Unicode and LaTeX as macros.
# Longest first: the thin space is a prefix of the +- token.
_TYPST_TOKENS = (
    (" ± ", "\\,$\\pm$\\,"),
    ("–", "--"),
    (" ", "~"),
    (" ", "\\,"),
)


def _typst_text_tex(text):
    from deviations import latex as tex

    out, rest = [], text
    while rest:
        for token, spelling in _TYPST_TOKENS:
            if rest.startswith(token):
                out.append(spelling)
                rest = rest[len(token):]
                break
        else:
            out.append(tex.escape(rest[0]))
            rest = rest[1:]
    return "".join(out)


def typst_cell_tex(cell):
    """The LaTeX a `.table.json` cell must correspond to, decoded independently.

    Written from the JSON format's documentation rather than from
    `deviations.typeset`, so a test comparing this to the `.tex` checks the
    two writers against each other instead of one writer against itself.
    """
    from deviations import latex as tex

    if isinstance(cell, list):
        return "".join(typst_cell_tex(run) for run in cell)
    if isinstance(cell, str):
        return _typst_text_tex(cell)
    assert isinstance(cell, dict) and len(cell) == 1, cell
    (kind, value), = cell.items()
    wrap = {
        "ident": "{}",
        "mono": "\\texttt{{{}}}",
        "emph": "\\emph{{{}}}",
        "sup": "$^{{{}}}$",
    }
    if kind == "url":
        return f"\\url{{{value}}}"
    if kind == "sup":
        return wrap[kind].format(value)
    return wrap[kind].format(tex.escape(value))


def tex_data_rows(text, head):
    """The cells of every data row of a generated table, header rows excluded."""
    head_row = " & ".join(head) + " \\\\"
    return [
        line[: -len(" \\\\")].split(" & ")
        for line in text.splitlines()
        if line.endswith("\\\\") and not line.startswith("\\") and line != head_row
    ]


# -- PDF geometry -----------------------------------------------------------


def _pdf_words(pdf):
    """[[(xMin, yMin, xMax, yMax, text)] per page] from `pdftotext -bbox`."""
    import html
    import re
    import subprocess

    out = subprocess.run(
        ["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout
    pages = []
    pattern = re.compile(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*)</word>'
    )
    for line in out.splitlines():
        if line.lstrip().startswith("<page"):
            pages.append([])
        match = pattern.search(line)
        if match and pages:
            *box, text = match.groups()
            pages[-1].append((*map(float, box), html.unescape(text)))
    return pages


def table_overflow(pdf, head, aligns, left, right, stop="Generated", tol=0.3):
    """(words checked, problems) for one generated table in a PDF.

    On every page whose words include `head` as a line, each word below it
    (up to the footer line starting with `stop`) must stay inside
    [left, right], must not overlap the next word on its line, and must end
    left of the next column's header when that column is left-aligned.
    Words whose baselines are within 2.5 pt share a line, since a monospace
    and a serif word on one line differ by about that much.
    """
    problems, checked = [], 0
    for number, page in enumerate(_pdf_words(pdf), 1):
        lines, anchors = {}, []
        for word in sorted(page, key=lambda w: w[3]):
            key = next((a for a in anchors if abs(a - word[3]) < 2.5), None)
            if key is None:
                anchors.append(word[3])
                key = word[3]
            lines.setdefault(key, []).append(word)
        rows = [sorted(lines[y]) for y in sorted(lines)]
        start = next(
            (i for i, row in enumerate(rows) if [w[4] for w in row][: len(head)] == list(head)),
            None,
        )
        if start is None:
            continue
        edges = [w[0] for w in rows[start][: len(head)]]
        for row in rows[start + 1:]:
            if row[0][4].startswith(stop):
                break
            for i, word in enumerate(row):
                checked += 1
                x0, _, x1, _, text = word
                if x0 < left - tol or x1 > right + tol:
                    problems.append((number, text, "outside the text block", x0, x1))
                if i + 1 < len(row) and x1 > row[i + 1][0] + tol:
                    problems.append((number, text, f"overlaps {row[i + 1][4]!r}", x1, row[i + 1][0]))
                column = max((c for c, e in enumerate(edges) if e <= x0 + tol), default=0)
                if (
                    column + 1 < len(edges)
                    and aligns[column + 1] == "left"
                    and x1 > edges[column + 1] + tol
                ):
                    problems.append((number, text, f"into column {head[column + 1]!r}", x1))
    return checked, problems
