"""The generated result tables.

Every cell is a number read from the store; the only text that is not is a
method name, a provenance and a status, all of which come from the manifest or
the record. Each table is built once as a `deviations.typeset.Table` and
written as LaTeX (geometry and escaping shared with the deviation artefacts,
`deviations.latex`) and as Typst data plus a `.typ` (`deviations.typst`).
"""

from deviations import typeset as ts

from .load import MIB, OK

COMMAND = "python -m report render"

PROVENANCE_SHORT = {
    "published-reference": "published ref.",
    "third-party-optimization": "third-party opt.",
    "own": "own",
}


def num(value, digits=1):
    return "--" if value is None else f"{value:,.{digits}f}"


def num_cell(value, digits=1):
    return [ts.DASH] if value is None else [ts.text(num(value, digits))]


def pm_cell(mean, sd, digits=1):
    """mean +- sd; a single run has no SD and must not pretend to."""
    if mean is None:
        return [ts.DASH]
    if sd is None:
        return num_cell(mean, digits)
    return [ts.text(num(mean, digits)), ts.PM, ts.text(num(sd, digits))]


def pm(mean, sd, digits=1):
    """`mean $\\pm$ sd`, as LaTeX."""
    return ts.cell_tex(pm_cell(mean, sd, digits))


def provenance_cell(value):
    return [ts.text(PROVENANCE_SHORT.get(value, value))]


def status_cell(counts):
    """Every status, ok first, so a failure can never be read as an absence."""
    if not counts:
        return [ts.DASH]
    ordered = sorted(counts.items(), key=lambda kv: (kv[0] != OK, kv[0]))
    out = []
    for index, (name, count) in enumerate(ordered):
        if index:
            out.append(ts.text(", "))
        out += [ts.text(name), ts.NBSP, ts.text(count)]
    return out


def label(campaign, kind):
    return f"tab:results-{campaign.name}-{kind}"


def stem(campaign, kind):
    """File stem shared by the `.tex`, `.typ` and `.table.json` of one table."""
    return f"results-{campaign.name}-{kind}"


def document(table, campaign, command=COMMAND):
    """The `.tex` artefact for a result table."""
    return ts.tex_document(table, command, str(campaign.directory))


# ---------------------------------------------------------------------------


SUMMARY_CAPTION = ts.cell(
    "Per method: provenance, configuration, completed runs, wall time of the "
    "measured stage and of the stage plus its converter, F1 at the primary "
    "tolerance, peak device memory, and the outcome of every run."
)


def summary_table(campaign, command=COMMAND):
    counts = campaign.status_counts()
    rows = {(r["method"], r["configuration"]): r for r in campaign.summary_rows()}
    body = []
    for method in campaign.methods:
        for configuration in campaign.configurations:
            row = rows.get((method, configuration))
            if row is None:
                continue
            peak = row["peak_device_bytes"]
            body.append(
                [
                    [ts.text(method)],
                    provenance_cell(campaign.provenance(method)),
                    [ts.text(configuration)],
                    [ts.text(f"{row['n_ok']}/{row['runs']}")],
                    pm_cell(row["wall_mean"], row["wall_sd"]),
                    num_cell(row["wall_prep_mean"]),
                    num_cell(row["convert_mean"]),
                    pm_cell(row["f1_mean"], row["f1_sd"], digits=4),
                    num_cell(peak / MIB, 0) if peak else [ts.DASH],
                    status_cell(counts.get((method, configuration))),
                ]
            )
    attribution = {
        r["attribution"] for r in rows.values() if r["attribution"]
    }
    footer = ts.cell(
        campaign.footer(command),
        " Wall time is the measured container alone; ",
        ts.emph("+prep"),
        " adds the converter, which is timed as its own phase. Peak device memory "
        f"is the {'/'.join(sorted(attribution)) or 'process'}-attributed maximum "
        "over the run's telemetry samples, not a mean. "
        f"F1 is at tolerance {num(campaign.primary_tolerance, 2)}",
        ts.THIN,
        "m.",
    )
    return ts.Table(
        colspec="@{}lllrrrrrrl@{}",
        head=[
            "method",
            "provenance",
            "conf.",
            "n ok/all",
            "wall s",
            "+prep s",
            "conv. s",
            "F1",
            "peak MiB",
            "status",
        ],
        rows=body,
        caption=SUMMARY_CAPTION,
        label=label(campaign, "summary"),
        footer=footer,
    )


def summary(campaign, command=COMMAND):
    return document(summary_table(campaign, command), campaign, command)


# ---------------------------------------------------------------------------


PHASES_CAPTION = ts.cell(
    "Exclusive time per phase group as a percentage of the method's measured ",
    ts.mono("run"),
    " span. The residual is time inside ",
    ts.mono("run"),
    " that no phase span claims, so the columns and the residual sum to 100",
    ts.THIN,
    "%.",
)


def phases_table(campaign, rows, disagreements, command=COMMAND):
    from .load import PHASE_GROUPS

    groups = [name for name, _ in PHASE_GROUPS]
    body = []
    unmapped = set()
    truncated = []
    for row in rows:
        unmapped |= set(row["unmapped"])
        if not row["base_is_run"]:
            truncated.append(row["run_key"])
        cells = [
            [ts.text(row["method"])],
            [ts.text(row["configuration"])],
            [ts.text(num(row["base_s"]))] + ([] if row["base_is_run"] else [ts.sup("*")]),
        ]
        cells += [
            [ts.text(f"{row['shares'][g]:.1f}")] if row["shares"].get(g) else [ts.DASH]
            for g in groups
        ]
        # Floating-point summation leaves a residual of -1e-14 on a trace that
        # balances exactly; printing it as "-0.00" invites the wrong question.
        residual = row["residual"] if abs(row["residual"]) >= 0.005 else 0.0
        cells.append([ts.text(f"{residual:+.2f}")])
        body.append(cells)

    footer = ts.cell(
        campaign.footer(command),
        " One run per method, the longest ",
        ts.mono("status=ok"),
        " repeat, named in the store; a phase share is structural, not a quantity "
        "to average over repeats. The converter is excluded: it runs outside the ",
        ts.mono("run"),
        " span.",
    )
    # A method with no completed run has no decomposition, so this table is
    # shorter than the summary table; say which methods are missing and why,
    # rather than letting a reader conclude they were not benchmarked.
    absent = [m for m in campaign.methods if m not in {r["method"] for r in rows}]
    if absent:
        footer += ts.cell(" No completed run, hence no row: " + ", ".join(absent) + ".")
    if truncated:
        footer += ts.cell(
            " ",
            ts.sup("*"),
            " no closed ",
            ts.mono("run"),
            " span (truncated trace); the base is the summed exclusive time "
            "instead: " + ", ".join(truncated) + ".",
        )
    if unmapped:
        footer += ts.cell(
            " Spans outside every group, counted in the residual: "
            + ", ".join(sorted(unmapped))
            + "."
        )
    if disagreements:
        footer += ts.cell(
            " Phase totals re-derived from ",
            ts.mono("phases.txt"),
            " disagree with the stored totals for: "
            + "; ".join(f"{key} ({', '.join(names)})" for key, names in disagreements)
            + ".",
        )
    return ts.Table(
        colspec="@{}ll" + "r" * (len(groups) + 2) + "@{}",
        head=["method", "conf.", "run s"] + groups + ["resid."],
        rows=body,
        caption=PHASES_CAPTION,
        label=label(campaign, "phases"),
        footer=footer,
    )


def phases(campaign, rows, disagreements, command=COMMAND):
    return document(phases_table(campaign, rows, disagreements, command), campaign, command)


# ---------------------------------------------------------------------------


TOLERANCES_CAPTION = ts.cell(
    "F1 at every evaluated tolerance, mean over the completed runs of each "
    "method. Accuracy and completeness at the primary tolerance are given "
    "alongside, since F1 alone hides which of the two a method trades away."
)


def tolerances_table(campaign, command=COMMAND):
    values = {
        (r["method"], r["configuration"], round(r["tolerance"], 6)): r
        for r in campaign.tolerance_rows()
    }
    columns = campaign.tolerances
    primary = campaign.primary_tolerance
    body = []
    for method in campaign.methods:
        for configuration in campaign.configurations:
            cells = [[ts.text(method)], [ts.text(configuration)]]
            present = [
                values.get((method, configuration, round(t, 6))) for t in columns
            ]
            if not any(present):
                continue
            cells.append([ts.text(max((r["n"] for r in present if r), default=0))])
            cells += [pm_cell(r["f1_mean"], r["f1_sd"], 4) if r else [ts.DASH] for r in present]
            at_primary = values.get((method, configuration, round(primary or 0, 6)))
            cells.append(num_cell(at_primary["accuracy_mean"], 4) if at_primary else [ts.DASH])
            cells.append(
                num_cell(at_primary["completeness_mean"], 4) if at_primary else [ts.DASH]
            )
            body.append(cells)
    footer = ts.cell(
        campaign.footer(command),
        f" Tolerances in metres; {num(primary, 2)}",
        ts.THIN,
        "m is the primary one. "
        "A run that produced no point cloud contributes no F1 at any tolerance "
        "and is counted in the status column of the summary table, not as a zero.",
    )
    return ts.Table(
        colspec="@{}llr" + "r" * len(columns) + "rr@{}",
        head=["method", "conf.", "n"]
        + [f"F1@{t:g}" for t in columns]
        + [f"acc@{primary:g}", f"comp@{primary:g}"],
        rows=body,
        caption=TOLERANCES_CAPTION,
        label=label(campaign, "f1-tolerances"),
        footer=footer,
    )


def tolerances(campaign, command=COMMAND):
    return document(tolerances_table(campaign, command), campaign, command)
