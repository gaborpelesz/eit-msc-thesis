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
