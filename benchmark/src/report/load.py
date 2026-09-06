"""Reading a campaign directory, and the aggregates the tables are built from.

Scalar aggregates go through DuckDB over the store's own views
(`bench.store.connect`), so the manuscript and `bench query` read the numbers
the same way. The phase breakdown is done in Python instead: it needs the
canonical grouping from `methods.yaml` and a cross-check against the raw
`phases.txt`, neither of which is an aggregate.
"""

import json
from pathlib import Path

import yaml
from bench import phases as ph
from bench import store as st
from deviations import manifest as mf


class ReportError(Exception):
    """A campaign that may not be turned into a table."""


# Statistics are computed over `status = 'ok'` runs only. Every other status is
# reported as a count: a timeout and a segfault are results, but averaging a
# crashed run's wall time into a method's mean would be a lie about what it
# costs to complete the scene.
OK = "ok"

MIB = 1024.0 * 1024.0


class Campaign:
    """One campaign directory, with everything a table needs about it."""

    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ReportError(f"{self.directory}: not a campaign directory")
        self.manifest = manifest
        self.spec = self._read_yaml("spec.yaml")
        self.fingerprint = self._read_json("fingerprint.json")
        self.deviations = self._read_json("deviations.json")
        self.records = list(st.iter_records(self.directory))
        if not self.records:
            raise ReportError(f"{self.directory}: no finished runs")
        self.name = self.records[0].get("campaign") or self.directory.name
        self._check_provenance()

    # -- campaign metadata --------------------------------------------------

    def _read_yaml(self, name):
        path = self.directory / name
        if not path.exists():
            raise ReportError(f"{path}: missing; is this a campaign directory?")
        return yaml.safe_load(path.read_text())

    def _read_json(self, name):
        path = self.directory / name
        if not path.exists():
            raise ReportError(f"{path}: missing; is this a campaign directory?")
        return json.loads(path.read_text())

    @property
    def methods(self):
        """Method order: the manifest's, which is the lineage order."""
        seen = {r["method"] for r in self.records}
        ordered = [m["name"] for m in self.manifest["methods"] if m["name"] in seen]
        return ordered + sorted(seen - set(ordered))

    @property
    def configurations(self):
        return sorted({r["configuration"] for r in self.records})

    @property
    def tolerances(self):
        return sorted(
            {q["tolerance"] for r in self.records for q in (r.get("quality") or [])}
        )

    @property
    def primary_tolerance(self):
        return self.records[0].get("primary_tolerance")

    @property
    def repeats(self):
        return self.spec.get("repeats")

    def entry(self, method):
        for entry in self.manifest.get("methods") or []:
            if entry["name"] == method:
                return entry
        raise ReportError(
            f"{self.name}: method `{method}` ran but is not in the manifest; "
            "its provenance cannot be stated"
        )

    def provenance(self, method):
        """R-REP: refuse rather than publish a method with no provenance."""
        entry = self.entry(method)
        value = entry.get("provenance")
        if not value:
            raise ReportError(
                f"{self.name}: `{method}` has no `provenance` in the manifest; "
                "a result table may not omit it (CLAUDE.md, Method provenance)"
            )
        if value not in mf.PROVENANCES:
            raise ReportError(
                f"{self.name}: `{method}` has provenance `{value}`, which is not one "
                f"of {', '.join(mf.PROVENANCES)}"
            )
        return value

    def _check_provenance(self):
        """Every method that ran must have a provenance, and it must agree.

        The record carries the provenance the manifest stated when the run was
        made; a disagreement means the table would describe a different method
        from the one that ran.
        """
        for method in self.methods:
            stated = self.provenance(method)
            for record in self.records:
                if record["method"] == method and record.get("provenance") != stated:
                    raise ReportError(
                        f"{record['run_key']} recorded provenance "
                        f"`{record.get('provenance')}` but the manifest now says "
                        f"`{stated}`; regenerate or re-run rather than publishing either"
                    )

    def footer(self, command):
        """The provenance line every generated table carries.

        A table with no campaign, no hardware and no image on it can be
        mistaken for another campaign's; this line makes that impossible.
        """
        fp = self.fingerprint
        image = fp.get("image_digest") or fp.get("image_id") or ""
        scenes = ", ".join(sorted({r["scene"] for r in self.records}))
        widths = ", ".join(str(w) for w in sorted({r["width"] for r in self.records}))
        return (
            f"Campaign {self.name}; scenes {scenes}; width {widths}; "
            f"{self.repeats} repeat(s) planned, {len(self.records)} finished run(s). "
            f"{fp.get('gpu_model')}, driver {fp.get('driver_version')}, "
            f"CUDA {fp.get('cuda_version')}, host {fp.get('hostname')}. "
            f"Image {fp.get('image')} {image[:19]}. "
            f"Statistics over status=ok runs only; every other status is counted in "
            f"the status column. Generated by {command} from {self.directory}."
        )

    # -- DuckDB ------------------------------------------------------------

    def connect(self):
        return st.connect(self.directory)

    def rows(self, statement):
        con = self.connect()
        cursor = con.execute(statement)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # -- aggregates --------------------------------------------------------

    def summary_rows(self):
        """Per (method, configuration): n, wall time, F1, peak memory, statuses."""
        return self.rows(
            f"""
            SELECT method, configuration,
                   count(*) AS runs,
                   sum(CASE WHEN status = '{OK}' THEN 1 ELSE 0 END) AS n_ok,
                   avg(CASE WHEN status = '{OK}' THEN wall_time_s END) AS wall_mean,
                   stddev_samp(CASE WHEN status = '{OK}' THEN wall_time_s END) AS wall_sd,
                   avg(CASE WHEN status = '{OK}' THEN wall_time_with_preprocess_s END)
                       AS wall_prep_mean,
                   avg(CASE WHEN status = '{OK}' THEN preprocess_convert_s END)
                       AS convert_mean,
                   avg(CASE WHEN status = '{OK}' THEN f1_primary END) AS f1_mean,
                   stddev_samp(CASE WHEN status = '{OK}' THEN f1_primary END) AS f1_sd,
                   max(CASE WHEN status = '{OK}' THEN peak_device_mem_bytes END)
                       AS peak_device_bytes,
                   max(CASE WHEN status = '{OK}' THEN peak_host_rss_bytes END)
                       AS peak_host_bytes,
                   any_value(device_mem_attribution) AS attribution
            FROM runs GROUP BY method, configuration
            """
        )

    def status_counts(self):
        """{(method, configuration): {status: count}} -- nothing is dropped."""
        counts = {}
        for row in self.rows(
            "SELECT method, configuration, status, count(*) AS runs "
            "FROM runs GROUP BY method, configuration, status"
        ):
            key = (row["method"], row["configuration"])
            counts.setdefault(key, {})[row["status"]] = row["runs"]
        return counts

    def tolerance_rows(self):
        """Per (method, configuration, tolerance): mean and SD of F1.

        `quality` is a list of structs in the record, so the aggregate has to
        unnest it; a run with no point cloud has an empty list and contributes
        no row rather than a zero.
        """
        return self.rows(
            f"""
            SELECT method, configuration, q.tolerance AS tolerance,
                   count(*) AS n,
                   avg(q.f1) AS f1_mean, stddev_samp(q.f1) AS f1_sd,
                   avg(q.accuracy) AS accuracy_mean,
                   avg(q.completeness) AS completeness_mean
            FROM runs, UNNEST(quality) AS t(q)
            WHERE status = '{OK}'
            GROUP BY method, configuration, q.tolerance
            """
        )


# ---------------------------------------------------------------------------
# Phase breakdown
# ---------------------------------------------------------------------------

# Exclusive (`self_s`) time is folded into these groups. The mapping is checked
# against `canonical_phases` in methods.yaml at run time, so a phase added to
# the vocabulary cannot silently vanish from the breakdown. `preprocess.*` is
# deliberately absent: it happens outside the method's `run` span and is
# reported as its own column in the summary table (CLAUDE.md: runtime is
# reported both with and without preprocessing).
PHASE_GROUPS = (
    ("setup", ("problem_list", "scene_scan", "device_init")),
    ("input", ("input_load", "device_upload")),
    ("pyramid", ("scale", "upsample")),
    ("prior", ("prior_build", "edge_prior", "confidence_eval")),
    ("patchmatch", ("patchmatch",)),
    ("readback", ("result_readback",)),
    ("write", ("depthmap_write", "output_write")),
    ("fusion", ("fusion",)),
    ("pass self", ("image_pass",)),
)

HARNESS_PREFIX = "preprocess"

# The span the shares are a share of. It is not in `canonical_phases`: it is
# the whole measured stage, emitted by bench_timer.h around main().
RUN_SPAN = "run"


def group_of(name):
    """The group a phase belongs to, by exact name or by dotted parent."""
    parent = name.split(".", 1)[0]
    for group, members in PHASE_GROUPS:
        if name in members or parent in members:
            return group
    return None


def check_group_map(manifest):
    """Refuse to render a breakdown whose grouping the manifest has outgrown."""
    unmapped = [
        name
        for name in manifest.get("canonical_phases") or []
        if not name.startswith(HARNESS_PREFIX) and group_of(name) is None
    ]
    if unmapped:
        raise ReportError(
            "canonical phases with no group in report.load.PHASE_GROUPS: "
            + ", ".join(unmapped)
            + " -- add them to the mapping rather than letting them fall into the "
            "residual"
        )


def method_phase_totals(record):
    """{phase: self_s} for the phases the method's own timer emitted."""
    return {
        row["name"]: row["self_s"]
        for row in (record.get("phases") or [])
        if row.get("source", "method") == "method"
    }


def run_span_seconds(record):
    for row in record.get("phases") or []:
        if row["name"] == RUN_SPAN and row.get("source", "method") == "method":
            return row["total_s"]
    return None


def phase_shares(record):
    """(base seconds, {group: %}, residual %, unmapped names, base_is_run).

    The base is the closed `run` span. A truncated trace has none -- a timeout
    or a SIGKILL leaves it open -- and then the summed exclusive time stands in,
    so the percentages still mean "share of what ran"; the caller is told which
    happened by `base_is_run`.
    """
    totals = method_phase_totals(record)
    groups, unmapped = {}, {}
    for name, seconds in totals.items():
        if name == RUN_SPAN:
            continue
        group = group_of(name)
        if group is None:
            unmapped[name] = unmapped.get(name, 0.0) + seconds
        else:
            groups[group] = groups.get(group, 0.0) + seconds
    # The `run` span's own exclusive time is deliberately NOT counted as
    # accounted for: it is time inside the measured stage that no phase claims,
    # which is exactly what the residual is meant to expose.
    accounted = sum(groups.values()) + sum(unmapped.values())
    run_total = run_span_seconds(record)
    base = run_total if run_total else accounted
    if not base:
        return 0.0, {}, 0.0, sorted(unmapped), bool(run_total)
    shares = {g: 100.0 * s / base for g, s in groups.items()}
    residual = 100.0 * (base - accounted) / base
    return base, shares, residual, sorted(unmapped), bool(run_total)


def trace_disagreement(run_dir, record, tolerance_s=1e-3):
    """Phases re-derived from `phases.txt` vs the totals the record stores.

    The record's totals were derived once, at write time; re-parsing the raw
    trace here means a table never depends on that derivation being right.
    Returns the phase names whose exclusive time differs.
    """
    path = Path(run_dir) / "phases.txt"
    if not path.exists():
        return []
    reparsed = {row["name"]: row["self_s"] for row in ph.totals(ph.parse_file(path))}
    stored = method_phase_totals(record)
    names = set(reparsed) | set(stored)
    return sorted(
        name
        for name in names
        if abs(reparsed.get(name, 0.0) - stored.get(name, 0.0)) > tolerance_s
    )


def phase_rows(campaign):
    """One row per (method, configuration), from the longest ok run.

    A phase share is a structural property of a method's schedule, not a
    quantity to average: repeats of one method produce the same decomposition
    to within noise, and averaging percentages across runs of different lengths
    would weight them wrongly. The run used is named in the table so the choice
    is checkable.
    """
    check_group_map(campaign.manifest)
    by_key = {}
    for record in campaign.records:
        if record.get("status") != OK:
            continue
        key = (record["method"], record["configuration"])
        current = by_key.get(key)
        if current is None or (record.get("wall_time_s") or 0) > (
            current.get("wall_time_s") or 0
        ):
            by_key[key] = record

    rows, disagreements = [], []
    for (method, configuration), record in sorted(by_key.items()):
        base, shares, residual, unmapped, base_is_run = phase_shares(record)
        differing = trace_disagreement(campaign.directory / record["run_key"], record)
        if differing:
            disagreements.append((record["run_key"], differing))
        rows.append(
            {
                "method": method,
                "configuration": configuration,
                "run_key": record["run_key"],
                "base_s": base,
                "base_is_run": base_is_run,
                "shares": shares,
                "residual": residual,
                "unmapped": unmapped,
            }
        )
    rows.sort(key=lambda r: (campaign.methods.index(r["method"]), r["configuration"]))
    return rows, disagreements
