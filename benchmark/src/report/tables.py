"""The generated LaTeX result tables.

Every cell is a number read from the store; the only text that is not is a
method name, a provenance and a status, all of which come from the manifest or
the record. Table geometry and escaping are shared with the deviation
artefacts (`deviations.latex`).
"""

from deviations import latex as tex

from .load import MIB, OK

COMMAND = "python -m report render"

PROVENANCE_SHORT = {
    "published-reference": "published ref.",
    "third-party-optimization": "third-party opt.",
    "own": "own",
}


def num(value, digits=1):
    return "--" if value is None else f"{value:,.{digits}f}"


def pm(mean, sd, digits=1):
    """`mean $\\pm$ sd`; a single run has no SD and must not pretend to."""
    if mean is None:
        return "--"
    if sd is None:
        return num(mean, digits)
    return f"{num(mean, digits)}\\,$\\pm$\\,{num(sd, digits)}"


def provenance_cell(value):
    return tex.escape(PROVENANCE_SHORT.get(value, value))


def status_cell(counts):
    """Every status, ok first, so a failure can never be read as an absence."""
    if not counts:
        return "--"
    ordered = sorted(counts.items(), key=lambda kv: (kv[0] != OK, kv[0]))
    return ", ".join(f"{tex.escape(name)}~{count}" for name, count in ordered)


def label(campaign, kind):
    return f"tab:results-{campaign.name}-{kind}"


# ---------------------------------------------------------------------------


SUMMARY_CAPTION = (
    "Per method: provenance, configuration, completed runs, wall time of the "
    "measured stage and of the stage plus its converter, F1 at the primary "
    "tolerance, peak device memory, and the outcome of every run."
)


def summary(campaign, command=COMMAND):
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
                tex.row(
                    [
                        tex.escape(method),
                        provenance_cell(campaign.provenance(method)),
                        tex.escape(configuration),
                        f"{row['n_ok']}/{row['runs']}",
                        pm(row["wall_mean"], row["wall_sd"]),
                        num(row["wall_prep_mean"]),
                        num(row["convert_mean"]),
                        pm(row["f1_mean"], row["f1_sd"], digits=4),
                        num(peak / MIB, 0) if peak else "--",
                        status_cell(counts.get((method, configuration))),
                    ]
                )
            )
    attribution = {
        r["attribution"] for r in rows.values() if r["attribution"]
    }
    footer = tex.escape(campaign.footer(command)) + (
        " Wall time is the measured container alone; \\emph{+prep} adds the converter, "
        "which is timed as its own phase. Peak device memory is the "
        f"{tex.escape('/'.join(sorted(attribution)) or 'process')}-attributed maximum "
        "over the run's telemetry samples, not a mean. "
        f"F1 is at tolerance {num(campaign.primary_tolerance, 2)}\\,m."
    )
    return tex.header(command, str(campaign.directory)) + tex.BOOKTABS_FALLBACK + "\n" + tex.table(
        "@{}lllrrrrrrl@{}",
        [
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
        body,
        SUMMARY_CAPTION,
        label(campaign, "summary"),
        footer,
    )


# ---------------------------------------------------------------------------


PHASES_CAPTION = (
    "Exclusive time per phase group as a percentage of the method's measured "
    "\\texttt{run} span. The residual is time inside \\texttt{run} that no phase "
    "span claims, so the columns and the residual sum to 100\\,\\%."
)


def phases(campaign, rows, disagreements, command=COMMAND):
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
            tex.escape(row["method"]),
            tex.escape(row["configuration"]),
            num(row["base_s"]) + ("" if row["base_is_run"] else "$^{*}$"),
        ]
        cells += [
            f"{row['shares'][g]:.1f}" if row["shares"].get(g) else "--" for g in groups
        ]
        # Floating-point summation leaves a residual of -1e-14 on a trace that
        # balances exactly; printing it as "-0.00" invites the wrong question.
        residual = row["residual"] if abs(row["residual"]) >= 0.005 else 0.0
        cells.append(f"{residual:+.2f}")
        body.append(tex.row(cells))

    notes = []
    # A method with no completed run has no decomposition, so this table is
    # shorter than the summary table; say which methods are missing and why,
    # rather than letting a reader conclude they were not benchmarked.
    absent = [m for m in campaign.methods if m not in {r["method"] for r in rows}]
    if absent:
        notes.append(
            "No completed run, hence no row: " + tex.escape(", ".join(absent)) + "."
        )
    if truncated:
        notes.append(
            "$^{*}$ no closed \\texttt{run} span (truncated trace); the base is the "
            "summed exclusive time instead: " + tex.escape(", ".join(truncated)) + "."
        )
    if unmapped:
        notes.append(
            "Spans outside every group, counted in the residual: "
            + tex.escape(", ".join(sorted(unmapped)))
            + "."
        )
    if disagreements:
        notes.append(
            "Phase totals re-derived from \\texttt{phases.txt} disagree with the "
            "stored totals for: "
            + tex.escape(
                "; ".join(f"{key} ({', '.join(names)})" for key, names in disagreements)
            )
            + "."
        )
    footer = " ".join(
        [
            tex.escape(campaign.footer(command)),
            "One run per method, the longest \\texttt{status=ok} repeat, named in the "
            "store; a phase share is structural, not a quantity to average over "
            "repeats. The converter is excluded: it runs outside the \\texttt{run} span.",
        ]
        + notes
    )
    return tex.header(command, str(campaign.directory)) + tex.BOOKTABS_FALLBACK + "\n" + tex.table(
        "@{}ll" + "r" * (len(groups) + 2) + "@{}",
        ["method", "conf.", "run s"] + groups + ["resid."],
        body,
        PHASES_CAPTION,
        label(campaign, "phases"),
        footer,
    )


# ---------------------------------------------------------------------------


TOLERANCES_CAPTION = (
    "F1 at every evaluated tolerance, mean over the completed runs of each "
    "method. Accuracy and completeness at the primary tolerance are given "
    "alongside, since F1 alone hides which of the two a method trades away."
)


def tolerances(campaign, command=COMMAND):
    values = {
        (r["method"], r["configuration"], round(r["tolerance"], 6)): r
        for r in campaign.tolerance_rows()
    }
    columns = campaign.tolerances
    primary = campaign.primary_tolerance
    body = []
    for method in campaign.methods:
        for configuration in campaign.configurations:
            cells = [tex.escape(method), tex.escape(configuration)]
            present = [
                values.get((method, configuration, round(t, 6))) for t in columns
            ]
            if not any(present):
                continue
            cells.append(str(max((r["n"] for r in present if r), default=0)))
            cells += [pm(r["f1_mean"], r["f1_sd"], 4) if r else "--" for r in present]
            at_primary = values.get((method, configuration, round(primary or 0, 6)))
            cells.append(num(at_primary["accuracy_mean"], 4) if at_primary else "--")
            cells.append(num(at_primary["completeness_mean"], 4) if at_primary else "--")
            body.append(tex.row(cells))
    footer = tex.escape(campaign.footer(command)) + (
        f" Tolerances in metres; {num(primary, 2)}\\,m is the primary one. "
        "A run that produced no point cloud contributes no F1 at any tolerance "
        "and is counted in the status column of the summary table, not as a zero."
    )
    return tex.header(command, str(campaign.directory)) + tex.BOOKTABS_FALLBACK + "\n" + tex.table(
        "@{}llr" + "r" * len(columns) + "rr@{}",
        ["method", "conf.", "n"]
        + [f"F1@{t:g}" for t in columns]
        + [f"acc@{primary:g}", f"comp@{primary:g}"],
        body,
        TOLERANCES_CAPTION,
        label(campaign, "f1-tolerances"),
        footer,
    )
