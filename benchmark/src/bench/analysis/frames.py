"""Reading the store for analysis: one or several campaigns at a time.

`bench.store.connect` serves one campaign, which is what a canned report needs.
A pair of arms lives in two campaigns -- a run key does not distinguish them --
so everything here takes a list of campaign directories and keeps the
`campaign` column that every record already carries.
"""

from pathlib import Path

from .. import store as st

# The metrics a variance table covers, in the order it prints them. Quality and
# phase columns are added per campaign, because the tolerances and the phase
# vocabulary differ between methods.
BASE_METRICS = [
    ("wall_time_s", "wall time (s)"),
    ("wall_time_with_preprocess_s", "wall + preprocess (s)"),
    ("preprocess_convert_s", "preprocess.convert (s)"),
    ("peak_device_mem_bytes", "peak GPU memory (MiB)"),
    ("peak_host_rss_bytes", "peak host RSS (MiB)"),
    ("energy_j", "GPU energy (J)"),
    ("io_write_bytes", "I/O written (MiB)"),
    ("intermediates_bytes", "scratch footprint (MiB)"),
]

# Columns reported in MiB rather than bytes.
MIB_COLUMNS = {
    "peak_device_mem_bytes",
    "peak_host_rss_bytes",
    "io_write_bytes",
    "io_read_bytes",
    "intermediates_bytes",
    "point_cloud_size_bytes",
}
MIB = 1024 * 1024


def connect(campaign_dirs):
    """A DuckDB connection with a `runs` view over every finished run given.

    Only directories whose name is a bare run key are read, so an aborted or
    superseded attempt can never enter an aggregate (`store.is_run_key_dir`).
    """
    import duckdb

    records = []
    for campaign_dir in _as_list(campaign_dirs):
        records += [str(d / "run.json") for d in st.finished_run_dirs(campaign_dir)]
    if not records:
        raise FileNotFoundError(f"{campaign_dirs}: no finished runs")
    con = duckdb.connect()
    con.execute(
        "CREATE VIEW runs AS SELECT * FROM "
        f"read_json_auto({st._sql_list(records)}, union_by_name=true)"
    )
    return con


def _as_list(value):
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(v) for v in value]


def records(campaign_dirs):
    """Every finished record, as dictionaries, in run-key order."""
    rows = []
    for campaign_dir in _as_list(campaign_dirs):
        rows += list(st.iter_records(campaign_dir))
    return sorted(rows, key=lambda r: (r.get("campaign", ""), r.get("run_key", "")))


# A run that crashed, timed out or produced nothing has a wall time, but it is
# not the method's wall time, so aggregates take `ok` runs only. R-FAIL-01
# keeps the failure in the store and R-REP-02 keeps it visible: nothing is
# dropped silently, and every table says what it left out.
DEFAULT_STATUSES = ("ok",)


def select(campaign_dirs, statuses=DEFAULT_STATUSES, require_clock_hold=False):
    """(kept, excluded) -- the records an aggregate may use, and why not the rest.

    `require_clock_hold` is off by default. D24 says a run whose SM clock did
    not hold the duty-cycle threshold should be excluded from comparisons, but
    on the development machine that is CUMVS's normal state (M-008), and
    dropping every run of one method silently would hide the finding rather
    than report it. The count is always in the header; the exclusion is opt-in.
    """
    kept, excluded = [], []
    for record in records(campaign_dirs):
        status = record.get("status")
        if statuses and status not in statuses:
            excluded.append((record.get("run_key"), f"status `{status}`"))
            continue
        held = (record.get("clock_hold") or {}).get("held")
        if require_clock_hold and held is not True:
            fraction = (record.get("clock_hold") or {}).get("fraction_at_expected")
            excluded.append(
                (record.get("run_key"), f"clock_hold.held={held} (fraction {fraction})")
            )
            continue
        kept.append(record)
    return kept, excluded


def exclusion_note(excluded, total):
    if not excluded:
        return f"All {total} runs are in the aggregates below."
    lines = [
        f"{len(excluded)} of {total} runs are **excluded** from the aggregates below "
        "and are kept in the store (R-FAIL-01):"
    ]
    lines += [f"  - `{key}` — {why}" for key, why in excluded]
    return "\n".join(lines)


def campaign_settings(campaign_dirs):
    """What distinguishes one arm from another, read back from the records.

    A campaign whose records disagree about a setting is reported as the set of
    values rather than as one: the arm is then not what the specification said,
    and the analysis has to show that rather than average over it.
    """
    settings = {}
    for record in records(campaign_dirs):
        campaign = record.get("campaign")
        entry = settings.setdefault(
            campaign,
            {
                "campaign": campaign,
                "runs": 0,
                "phase_timer": set(),
                "debug_output": set(),
                "method_env": set(),
                "image_digest": set(),
                "spec_sha256": set(),
            },
        )
        entry["runs"] += 1
        entry["phase_timer"].add(record.get("phase_timer"))
        debug = record.get("debug_output")
        entry["debug_output"].add(
            debug.get("setting") if isinstance(debug, dict) else debug
        )
        entry["method_env"].add(record.get("method_env_json") or "{}")
        entry["image_digest"].add(record.get("image_digest"))
        entry["spec_sha256"].add(record.get("spec_sha256"))
    for entry in settings.values():
        for key, value in list(entry.items()):
            if isinstance(value, set):
                entry[key] = sorted(v for v in value if v is not None) or [None]
    return settings


def metric(record, column):
    """One metric of one record, in the unit the tables print."""
    value = record.get(column)
    if value is None:
        return None
    if column in MIB_COLUMNS:
        return float(value) / MIB
    return float(value)


def quality_series(record):
    """{tolerance: f1} for one record; a `no_output` run scores 0 (R-REP-02)."""
    if record.get("status") == "no_output":
        return {}
    return {
        float(q["tolerance"]): q.get("f1")
        for q in (record.get("quality") or [])
        if q.get("tolerance") is not None
    }


def phase_totals(record, top_level_only=True, source="method"):
    """{phase name: total_s} from the record's phase totals.

    Top-level means a name with no dot in it: `patchmatch.init` is a child of
    `patchmatch` and its time is already inside the parent's total, so summing
    both would double count (R-TIM-05).

    Dots do not express the whole hierarchy -- ACMH's `image_pass` contains
    `device_upload`, `patchmatch` and `depthmap_write`, none of which is dotted
    -- so these totals still overlap and must not be added up. Each is reported
    for its own dispersion, not as a share of a partition.
    """
    totals = {}
    for row in record.get("phases") or []:
        if source and row.get("source") != source:
            continue
        name = row.get("name")
        if top_level_only and "." in str(name):
            continue
        totals[name] = totals.get(name, 0.0) + float(row.get("total_s") or 0.0)
    return totals


def phase_shares(record):
    """Every phase's total as a fraction of the trace's `run` span.

    The share, not the seconds, is what the SYNC pair moves: a synchronisation
    that reattributes work between two spans changes both shares and may leave
    the total untouched.

    A span's total includes its nested spans, so these shares overlap and do
    not sum to 1. Only the shift between two arms is meaningful here.
    """
    totals = phase_totals(record, top_level_only=False)
    root = totals.get("run")
    if not root:
        return {}
    return {name: value / root for name, value in totals.items()}


def group(rows, key=lambda r: r.get("method")):
    grouped = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    return grouped
