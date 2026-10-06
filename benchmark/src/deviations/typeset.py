"""A typesetting-neutral table, rendered to LaTeX and to Typst from one source.

The generated tables used to be assembled directly as LaTeX strings. The
manuscript is moving to Typst, which cannot read LaTeX, so every table is now
built once as a `Table` of `Seg` runs and then written out twice:

* `tex(table)` -- the LaTeX that `deviations.latex` has always produced,
  byte for byte, so the `.tex` artefacts do not change while both exist;
* `deviations.typst` -- a JSON data file plus a small `.typ` that loads it.

A cell is a list of segments. Each segment says what its text *is* (plain,
monospace, emphasised, a URL, a superscript) and never carries markup of
either language; the two writers add their own. A `lit` segment is the one
exception: a typographic token (an en dash, a thin space, `+-`) whose LaTeX
spelling is a macro and whose Typst spelling is a Unicode character, kept
apart from manifest text so that a real Unicode character in the manifest is
never re-spelt on the LaTeX side.
"""

from dataclasses import dataclass

from . import latex as tex


@dataclass(frozen=True)
class Seg:
    kind: str  # text | mono | emph | url | sup | lit
    text: str
    latex: str = ""  # `lit` only: how LaTeX spells the token


def text(value):
    return Seg("text", str(value))


def mono(value):
    return Seg("mono", str(value))


def emph(value):
    return Seg("emph", str(value))


def url(value):
    return Seg("url", str(value))


def sup(value):
    return Seg("sup", str(value))


# A missing value. LaTeX's `--` ligature is an en dash; a Typst string is not
# run through ligatures, so it gets the character itself.
DASH = Seg("lit", "–", "--")
NBSP = Seg("lit", " ", "~")
THIN = Seg("lit", " ", "\\,")
PM = Seg("lit", " ± ", "\\,$\\pm$\\,")


def cell(*parts):
    """A cell from segments and plain strings (a string is a `text` segment)."""
    return [part if isinstance(part, Seg) else text(part) for part in parts]


@dataclass
class Table:
    """One generated table, independent of the language it is written in.

    `colspec` is the LaTeX column specification; the Typst writer derives its
    column widths and alignments from it, so the two cannot disagree. `head`
    is written verbatim on the LaTeX side, as it always was. `long` selects a
    `longtable` (caption on top, splits across pages) over a `table` float.
    """

    colspec: str
    head: list
    rows: list
    caption: list
    label: str
    footer: list = None
    long: bool = False


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def seg_tex(seg):
    if seg.kind == "text":
        return tex.escape(seg.text)
    if seg.kind == "mono":
        return f"\\texttt{{{tex.escape(seg.text)}}}"
    if seg.kind == "emph":
        return f"\\emph{{{tex.escape(seg.text)}}}"
    if seg.kind == "url":
        # `\url{}` takes its argument verbatim; escaping it would print the
        # backslashes.
        return f"\\url{{{seg.text}}}"
    if seg.kind == "sup":
        return f"$^{{{seg.text}}}$"
    if seg.kind == "lit":
        return seg.latex
    raise ValueError(f"unknown segment kind {seg.kind!r}")


def cell_tex(segments):
    return "".join(seg_tex(s) for s in segments)


def tex_body(table):
    """The table environment alone, without the generated header comment."""
    body = [tex.row([cell_tex(c) for c in row]) for row in table.rows]
    footer = cell_tex(table.footer) if table.footer else None
    write = tex.longtable if table.long else tex.table
    return write(
        table.colspec, table.head, body, cell_tex(table.caption), table.label, footer
    )


def tex_document(table, command, source):
    """The complete `.tex` artefact: header, booktabs fallback, table."""
    return tex.header(command, source) + tex.BOOKTABS_FALLBACK + "\n" + tex_body(table)
