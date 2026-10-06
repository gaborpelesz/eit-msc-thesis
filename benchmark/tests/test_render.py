"""`deviations render`: determinism, JSON round-trip, LaTeX escaping.

The store's copy of the deviation document and this package's must never drift,
so one test compares the two builders on the real manifest.
"""

import json
import shutil
import subprocess

import pytest
from bench import store as st
from deviations import latex as tex
from deviations import manifest as mf
from conftest import braces_balance
from deviations import render as rd
from deviations import typeset as ts
from deviations import typst

REPO_ROOT = mf.repo_root(mf.default_manifest_path().parent)


@pytest.fixture(scope="module")
def document(manifest_module):
    return rd.document(manifest_module, REPO_ROOT)


@pytest.fixture(scope="module")
def manifest_module():
    return mf.load(mf.default_manifest_path())


def test_document_matches_what_the_harness_embeds(manifest_module, document):
    """R-STO-02: the checked-in artefact and each run record describe one tree.

    `generated_at` is the campaign's start time in the harness copy and null in
    the rendered one; everything else has to be identical.
    """
    embedded = st.deviations_document(manifest_module, REPO_ROOT)
    assert embedded.pop("generated_at") is not None
    assert document["generated_at"] is None
    assert {k: v for k, v in document.items() if k != "generated_at"} == embedded


def test_render_is_deterministic(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        rd.render(mf.default_manifest_path(), methods_dir=out, thesis_dir=out)
    for name in (
        "DEVIATIONS.md",
        "deviations.json",
        "deviations.tex",
        "methods-provenance.tex",
        "deviations.typ",
        "deviations.table.json",
        "methods-provenance.typ",
        "methods-provenance.table.json",
    ):
        assert (first / name).read_text() == (second / name).read_text()


def test_json_round_trips(document):
    assert json.loads(rd.json_text(document)) == document


def test_json_keys_are_sorted(document):
    """Deterministic key order at every level, so a diff shows only real change."""

    def check(pairs):
        keys = [k for k, _ in pairs]
        assert keys == sorted(keys)
        return dict(pairs)

    json.loads(rd.json_text(document), object_pairs_hook=check)


def test_every_manifest_method_appears(manifest_module, document):
    names = [m["name"] for m in manifest_module["methods"]]
    assert [m["name"] for m in document["methods"]] == names
    for artefact in (rd.markdown(document), rd.tex_deviations(document),
                     rd.tex_provenance(document)):
        for name in names:
            assert tex.escape(name) in artefact or name in artefact


def test_excluded_method_is_rendered_with_zero_commits(document):
    excluded = [m for m in document["methods"] if m["status"] == "excluded"]
    assert excluded, "DVP-MVS is excluded from the campaign (D19-b, F-018)"
    for method in excluded:
        assert method["fork_deviations"] == []
        assert "pristine fork, excluded from the campaign" in rd.tex_deviations(document)


def test_provenance_is_stated_for_every_method(document):
    for method in document["methods"]:
        assert method["provenance"] in mf.PROVENANCES
    # CUMVS is third-party prior art, never the author's work (CLAUDE.md).
    cumvs = next(m for m in document["methods"] if m["name"] == "CUMVS")
    assert cumvs["provenance"] == "third-party-optimization"
    assert "third-party-optimization" in rd.tex_provenance(document)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a_b", "a\\_b"),
        ("50%", "50\\%"),
        ("a & b", "a \\& b"),
        ("#1", "\\#1"),
        ("$x$", "\\$x\\$"),
        ("{}", "\\{\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
        ("a\\b", "a\\textbackslash{}b"),
    ],
)
def test_escape(raw, expected):
    assert tex.escape(raw) == expected


def test_tex_has_no_unescaped_specials(document):
    """`_ % #` may only appear escaped.

    `&` is excluded because it is also the column separator, and `\\url{}` is
    excluded because it takes its argument verbatim; the underscore in
    `HPM-MVS_plusplus` is the case that makes this test worth having.
    """
    for artefact in (rd.tex_deviations(document), rd.tex_provenance(document)):
        for line in artefact.splitlines():
            if line.startswith("%") or "\\url{" in line:
                continue
            for index, character in enumerate(line):
                if character in "_%#":
                    assert index and line[index - 1] == "\\", f"{character} in {line!r}"


def test_tex_is_footnote_free(document):
    for artefact in (rd.tex_deviations(document), rd.tex_provenance(document)):
        assert "\\footnote" not in artefact.replace("\\footnotesize", "")


def test_tex_carries_label_and_booktabs_fallback(document):
    assert "\\label{tab:deviations}" in rd.tex_deviations(document)
    assert "\\label{tab:methods-provenance}" in rd.tex_provenance(document)
    for artefact in (rd.tex_deviations(document), rd.tex_provenance(document)):
        assert "\\providecommand{\\toprule}" in artefact
        assert artefact.startswith("% GENERATED by `uv run deviations render`")


def test_tex_rows_have_the_declared_column_count(document):
    for artefact, columns in (
        (rd.tex_deviations(document), 6),
        (rd.tex_provenance(document), 7),
    ):
        for line in artefact.splitlines():
            # Data rows only: the caption, the repeated heads and the
            # `continued` markers also end in `\\`.
            if not line.endswith("\\\\") or line.startswith("\\"):
                continue
            assert line.count(" & ") == columns - 1, line


def test_class_counts(document):
    counts = {m["name"]: rd.class_counts(m) for m in document["methods"]}
    assert set(counts["CUMVS"]) <= set(mf.CLASSES_ALLOWED_IN_FOREIGN_FORK)
    assert sum(counts["DVP-MVS"].values()) == 0


def test_generated_tex_has_balanced_braces(document):
    """No TeX is installed on the harness host, so this stands in for a compile."""
    for artefact in (rd.tex_deviations(document), rd.tex_provenance(document)):
        assert braces_balance(artefact)[0] == braces_balance(artefact)[1]


# -- Typst ------------------------------------------------------------------


def _plain(cell):
    """The text of a JSON cell, runs concatenated."""
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        return next(iter(cell.values()))
    return "".join(_plain(run) for run in cell)


def _tex_data_rows(text, head):
    head_row = " & ".join(head) + " \\\\"
    return [
        line
        for line in text.splitlines()
        if line.endswith("\\\\") and not line.startswith("\\") and line != head_row
    ]


@pytest.mark.parametrize("builder", ["deviations_table", "provenance_table"])
def test_typst_data_matches_the_tex_row_for_row(document, builder):
    """The two writers read one Table, so they must agree cell for cell."""
    table = getattr(rd, builder)(document)
    data = json.loads(typst.json_text(table, rd.COMMAND, rd.SOURCE))
    tex_rows = _tex_data_rows(ts.tex_body(table), table.head)
    assert len(data["rows"]) == len(tex_rows) == len(table.rows)
    for row, line in zip(data["rows"], tex_rows):
        assert len(row) == len(table.head)
        for cell, tex_cell in zip(row, line[: -len(" \\\\")].split(" & ")):
            text = _plain(cell)
            if tex_cell.startswith("\\url{"):
                assert tex_cell == f"\\url{{{text}}}"
            else:
                assert ts.cell_tex(ts.cell(text.replace("\u2013", "--"))) in tex_cell


def test_typst_carries_label_caption_and_header(document):
    for table, stem in (
        (rd.deviations_table(document), "deviations"),
        (rd.provenance_table(document), "methods-provenance"),
    ):
        source = typst.source_text(table, stem, rd.COMMAND, rd.SOURCE)
        assert source.startswith("// GENERATED by `uv run deviations render` -- do not edit.")
        assert f"<{table.label}>" in source
        assert f'json("{stem}.table.json")' in source
        assert "table.header(\n        repeat: true," in source
        assert "set block(breakable: true)" in source
        data = json.loads(typst.json_text(table, rd.COMMAND, rd.SOURCE))
        assert data["label"] == table.label
        assert data["head"] == table.head
        assert data["generated"].startswith("GENERATED by")


def test_typst_json_is_deterministic_and_one_row_per_line(document):
    table = rd.deviations_table(document)
    first = typst.json_text(table, rd.COMMAND, rd.SOURCE)
    assert first == typst.json_text(table, rd.COMMAND, rd.SOURCE)
    rows = [line for line in first.splitlines() if line.startswith("    [")]
    assert len(rows) == len(table.rows)


def test_typst_states_provenance_for_every_method(document):
    data = json.loads(
        typst.json_text(rd.provenance_table(document), rd.COMMAND, rd.SOURCE)
    )
    stated = {_plain(row[0]): _plain(row[1]) for row in data["rows"]}
    # CUMVS is third-party prior art, never the author's work (CLAUDE.md).
    assert stated["CUMVS"] == "third-party-optimization"
    assert set(stated.values()) <= set(mf.PROVENANCES)


@pytest.mark.parametrize(
    "colspec,widths,aligns",
    [
        ("@{}lllrrrrrrl@{}", ["auto"] * 10, ["left"] * 3 + ["right"] * 6 + ["left"]),
        (
            "@{}llllp{0.16\\linewidth}p{0.30\\linewidth}@{}",
            ["auto"] * 4 + ["16%", "30%"],
            ["left"] * 6,
        ),
        ("c", ["auto"], ["center"]),
    ],
)
def test_typst_columns_follow_the_latex_colspec(colspec, widths, aligns):
    assert typst.columns(colspec) == (widths, aligns)


def test_typst_columns_refuse_what_they_cannot_read():
    with pytest.raises(ValueError):
        typst.columns("@{}lX@{}")


def test_typst_cells_carry_runs_not_markup():
    assert typst.cell_data(ts.cell("a_b & 50%")) == "a_b & 50%"
    assert typst.cell_data([ts.mono("abc")]) == {"mono": "abc"}
    assert typst.cell_data([ts.text("0.5"), ts.sup("*")]) == ["0.5", {"sup": "*"}]
    assert typst.cell_data([ts.text("1.0"), ts.PM, ts.text("0.1")]) == "1.0\u2009\u00b1\u20090.1"
    assert typst.cell_data([ts.DASH]) == "\u2013"
    assert typst.cell_data([]) == ""


def test_typst_label_falls_back_to_the_label_function():
    table = ts.Table("l", ["a"], [[ts.cell("x")]], ts.cell("c"), "tab:odd name")
    assert '#label("tab:odd name")' in typst.source_text(table, "odd", "cmd", "src")


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_generated_typst_compiles(tmp_path):
    """The `.typ` files are only useful if Typst accepts them."""
    rd.render(mf.default_manifest_path(), methods_dir=tmp_path / "m", thesis_dir=tmp_path / "generated")
    (tmp_path / "main.typ").write_text(
        '#include "generated/deviations.typ"\n'
        '#include "generated/methods-provenance.typ"\n'
        "@tab:deviations @tab:methods-provenance\n"
    )
    result = subprocess.run(
        ["typst", "compile", "--root", str(tmp_path), str(tmp_path / "main.typ"),
         str(tmp_path / "main.pdf")],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "main.pdf").stat().st_size > 0
