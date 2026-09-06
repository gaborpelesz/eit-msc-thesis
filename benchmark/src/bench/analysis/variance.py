"""The variance pilot's tables and the D17 repeat-count arithmetic.

R-STA-02 asks that every aggregate carry its repeat count and a dispersion;
R-STA-03 asks that the campaign's repeat count come from a measured variance
rather than from a convention. This module answers both from the store.
"""

from . import frames as fr
from . import markdown as md
from . import stats

SIZING_REPEATS = (1, 3, 5)


def metric_rows(records):
    """(column, label, [values]) for every metric a variance table covers.

    The quality and phase columns are discovered from the records themselves,
    so a method that emits a phase no other method has still gets a row.
    """
    rows = [(column, label, [fr.metric(r, column) for r in records])
            for column, label in fr.BASE_METRICS]

    tolerances = sorted({t for r in records for t in fr.quality_series(r)})
    for tolerance in tolerances:
        rows.append(
            (
                f"f1@{tolerance}",
                f"F1 @ {tolerance * 100:g} cm",
                [fr.quality_series(r).get(tolerance) for r in records],
            )
        )

    phases = sorted({p for r in records for p in fr.phase_totals(r)})
    for phase in phases:
        rows.append(
            (
                f"phase:{phase}",
                f"phase `{phase}` total (s)",
                [fr.phase_totals(r).get(phase) for r in records],
            )
        )
    return [row for row in rows if any(v is not None for v in row[2])]


def by_method(campaign_dirs, resamples=stats.BOOTSTRAP_RESAMPLES,
              statuses=fr.DEFAULT_STATUSES, require_clock_hold=False):
    """{method: [{metric, label, n, mean, sd, cv, min, max, ci...}]}."""
    kept, _ = fr.select(campaign_dirs, statuses, require_clock_hold)
    grouped = fr.group(kept)
    out = {}
    for method, records in sorted(grouped.items()):
        rows = []
        for column, label, values in metric_rows(records):
            summary = stats.describe(values, resamples=resamples)
            rows.append({"metric": column, "label": label, **summary})
        out[method] = rows
    return out


def sizing(campaign_dirs, metrics=("wall_time_s", "f1@0.02"), repeats=SIZING_REPEATS,
           resamples=stats.BOOTSTRAP_RESAMPLES, statuses=fr.DEFAULT_STATUSES,
           require_clock_hold=False):
    """The D17 table: what CI half-width N repeats would buy, per method.

    The SD is the one this pilot measured. Projecting from it assumes the
    campaign's runs are drawn from the same distribution as the pilot's -- same
    machine, same scene, same width, same configuration -- which is exactly why
    the pilot has to be re-run on the campaign machine.
    """
    out = {}
    kept, _ = fr.select(campaign_dirs, statuses, require_clock_hold)
    for method, records in sorted(fr.group(kept).items()):
        available = {column: values for column, _, values in metric_rows(records)}
        labels = {column: label for column, label, _ in metric_rows(records)}
        rows = []
        for column in metrics:
            values = available.get(column)
            if values is None:
                continue
            summary = stats.describe(values, resamples=resamples)
            for n in repeats:
                rows.append(
                    {
                        "metric": column,
                        "label": labels.get(column, column),
                        "pilot_n": summary["n"],
                        "mean": summary["mean"],
                        "sd": summary["sd"],
                        "cv": summary["cv"],
                        **stats.projected_halfwidth(summary["mean"], summary["sd"], n),
                    }
                )
        out[method] = rows
    return out


def status_rows(campaign_dirs):
    """One row per run: what it was, how it ended, and whether it is quotable."""
    rows = []
    for record in fr.records(campaign_dirs):
        hold = record.get("clock_hold") or {}
        debug = record.get("debug_output")
        rows.append(
            {
                "campaign": record.get("campaign"),
                "run_key": record.get("run_key"),
                "method": record.get("method"),
                "provenance": record.get("provenance"),
                "repeat": record.get("repeat"),
                "status": record.get("status"),
                "wall_time_s": record.get("wall_time_s"),
                "f1_primary": record.get("f1_primary"),
                "clock_fraction": hold.get("fraction_at_expected"),
                "clock_held": hold.get("held"),
                "instrumented": record.get("instrumented", record.get("phase_timer")),
                "debug_output": (
                    debug.get("setting") if isinstance(debug, dict) else debug
                ),
                "mvs_bench_sync": record.get("mvs_bench_sync"),
                "warnings": len(record.get("warnings") or []),
            }
        )
    return rows


def render_status(campaign_dirs):
    rows = status_rows(campaign_dirs)
    body = md.table(
        ["campaign", "run", "status", "wall s", "F1@2cm", "clock frac", "held",
         "timer", "debug", "SYNC", "warn"],
        [
            [
                r["campaign"], r["run_key"], r["status"], md.fmt(r["wall_time_s"], 1),
                md.fmt(r["f1_primary"], 4), md.fmt(r["clock_fraction"], 4),
                md.fmt(r["clock_held"]), md.fmt(r["instrumented"]),
                md.fmt(r["debug_output"]), md.fmt(r["mvs_bench_sync"] or "—"),
                str(r["warnings"]),
            ]
            for r in rows
        ],
        align="lllrrrlllll",
    )
    return body


def render(campaign_dirs, resamples=stats.BOOTSTRAP_RESAMPLES,
           statuses=fr.DEFAULT_STATUSES, require_clock_hold=False):
    """The whole variance report as Markdown."""
    all_records = fr.records(campaign_dirs)
    _, excluded = fr.select(campaign_dirs, statuses, require_clock_hold)
    parts = [
        md.section("Runs", render_status(campaign_dirs)),
        fr.exclusion_note(excluded, len(all_records)) + "\n",
    ]
    for method, rows in by_method(
        campaign_dirs, resamples=resamples, statuses=statuses,
        require_clock_hold=require_clock_hold,
    ).items():
        body = md.table(
            ["metric", "n", "mean", "SD", "CV", "min", "max",
             "bootstrap 95 % CI", "half-width"],
            [
                [
                    r["label"], str(r["n"]), md.fmt(r["mean"]), md.fmt(r["sd"]),
                    md.pct(None if r["cv"] is None else 100 * r["cv"]),
                    md.fmt(r["min"]), md.fmt(r["max"]),
                    f"{md.fmt(r['ci_low'])} – {md.fmt(r['ci_high'])}"
                    if r["ci_low"] is not None else "—",
                    md.pct(r["ci_halfwidth_pct"]),
                ]
                for r in rows
            ],
            align="lrrrrrrlr",
        )
        parts.append(md.section(method, body))

    for method, rows in sizing(
        campaign_dirs, resamples=resamples, statuses=statuses,
        require_clock_hold=require_clock_hold,
    ).items():
        if not rows:
            continue
        body = md.table(
            ["metric", "pilot n", "mean", "SD", "CV", "N", "t half-width",
             "t half-width / mean", "z half-width / mean"],
            [
                [
                    r["label"], str(r["pilot_n"]), md.fmt(r["mean"]), md.fmt(r["sd"]),
                    md.pct(None if r["cv"] is None else 100 * r["cv"]), str(r["n"]),
                    md.fmt(r["t_halfwidth"]), md.pct(r["t_halfwidth_pct"]),
                    md.pct(r["z_halfwidth_pct"]),
                ]
                for r in rows
            ],
            align="lrrrrrrrr",
        )
        parts.append(md.section(f"{method}: repeat count (D17)", body))
    return "\n".join(parts)
