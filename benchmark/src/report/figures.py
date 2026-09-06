"""Generated figures, as pgfplots sources.

They are emitted as LaTeX rather than as rendered images on purpose: a `.tex`
figure is a text file whose every coordinate is greppable back to the store,
it is diffable in review, and it needs no plotting library in the harness
environment. That last point is not a preference here -- `benchmark/uv.lock`
and `benchmark/pyproject.toml` are hashed into the Docker image's build
manifest, and `bench run` refuses to start a campaign whose working tree
differs from the image, so adding a plotting dependency would stop the fleet
until the image was rebuilt.

The class file does not load pgfplots; each figure names the two preamble
lines it needs in its own header comment.
"""

from deviations import latex as tex

from .load import OK
from .tables import COMMAND

PREAMBLE_NOTE = (
    "% Requires in the preamble:  \\usepackage{pgfplots}\n"
    "%                            \\pgfplotsset{compat=1.18}\n"
)

# Bars are stacked in schedule order, so a reader sweeps a method's row the way
# the method runs: set up, load, build, match, read back, write, fuse.
BAR_COLOURS = (
    "gray!40",
    "blue!45",
    "cyan!45",
    "orange!55",
    "red!55",
    "violet!45",
    "brown!45",
    "green!45",
    "black!25",
)


def _labels(names):
    return ", ".join("{" + tex.escape(n) + "}" for n in names)


def phase_shares(campaign, rows, command=COMMAND):
    """Stacked bars: exclusive phase share per method."""
    from .load import PHASE_GROUPS

    groups = [name for name, _ in PHASE_GROUPS]
    keys = [(r["method"], r["configuration"]) for r in rows]
    ticks = [
        m if len(campaign.configurations) == 1 else f"{m} ({c})" for m, c in keys
    ]
    lines = [
        tex.header(command, str(campaign.directory)),
        PREAMBLE_NOTE,
        "\\begin{figure}[H]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  ybar stacked, bar width=12pt, width=\\linewidth, height=0.5\\linewidth,",
        f"  xmin=0.5, xmax={len(rows) + 0.5}, ymin=0, ymax=100,",
        "  ylabel={exclusive time (\\% of \\texttt{run})},",
        f"  xtick={{{','.join(str(i + 1) for i in range(len(rows)))}}},",
        f"  xticklabels={{{_labels(ticks)}}},",
        "  x tick label style={rotate=45, anchor=east, font=\\footnotesize},",
        "  legend style={font=\\footnotesize, at={(0.5,-0.35)}, anchor=north,",
        "                legend columns=5, draw=none},",
        "  tick label style={font=\\footnotesize}, ymajorgrids, every axis plot/.append style={draw=none},",
        "]",
    ]
    for index, group in enumerate(groups):
        colour = BAR_COLOURS[index % len(BAR_COLOURS)]
        coordinates = " ".join(
            f"({position + 1},{row['shares'].get(group, 0.0):.2f})"
            for position, row in enumerate(rows)
        )
        lines.append(f"\\addplot+[fill={colour}] coordinates {{{coordinates}}};")
        lines.append(f"\\addlegendentry{{{tex.escape(group)}}}")
    lines += [
        "\\end{axis}",
        "\\end{tikzpicture}",
        "\\caption{Exclusive time per phase group, as a percentage of each method's "
        "measured \\texttt{run} span. Same data as Table~\\ref{tab:results-%s-phases}.}"
        % campaign.name,
        f"\\label{{fig:results-{campaign.name}-phase-shares}}",
        "\\end{figure}",
    ]
    return "\n".join(lines) + "\n"


def wall_vs_f1(campaign, command=COMMAND):
    """Scatter: what each method's runtime buys in reconstruction quality."""
    points = {}
    for record in campaign.records:
        if record.get("status") != OK or record.get("f1_primary") is None:
            continue
        points.setdefault(record["method"], []).append(
            (record["wall_time_s"], record["f1_primary"], record["configuration"])
        )
    lines = [
        tex.header(command, str(campaign.directory)),
        PREAMBLE_NOTE,
        "\\begin{figure}[H]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  width=\\linewidth, height=0.6\\linewidth,",
        "  xlabel={wall time of the measured stage (s)},",
        f"  ylabel={{F1 at {campaign.primary_tolerance:g}\\,m}},",
        "  legend style={font=\\footnotesize, at={(1.02,1)}, anchor=north west, draw=none},",
        "  tick label style={font=\\footnotesize}, grid=both,",
        "]",
    ]
    for method in campaign.methods:
        if method not in points:
            continue
        coordinates = " ".join(
            f"({wall:.3f},{f1:.6f})" for wall, f1, _ in sorted(points[method])
        )
        lines.append(f"\\addplot+[only marks] coordinates {{{coordinates}}};")
        lines.append(f"\\addlegendentry{{{tex.escape(method)}}}")
    failed = sorted(
        {r["method"] for r in campaign.records if r.get("status") != OK}
        - set(points)
    )
    note = (
        " Not plotted, no completed run: " + tex.escape(", ".join(failed)) + "."
        if failed
        else ""
    )
    lines += [
        "\\end{axis}",
        "\\end{tikzpicture}",
        "\\caption{Wall time against F1 at the primary tolerance, one mark per "
        "completed run.%s}" % note,
        f"\\label{{fig:results-{campaign.name}-wall-f1}}",
        "\\end{figure}",
    ]
    return "\n".join(lines) + "\n"
