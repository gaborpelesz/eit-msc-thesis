"""The result store: campaign layout, the run record, and DuckDB queries.

R-STO-01..R-STO-06, R-EXP-02, R-EXP-06, R-EXP-07, R-RUN-02, R-OBS-01,
R-REP-01, R-REP-02, R-STA-02.

The store is a directory tree; there is no database. A finished run directory
is never rewritten: `write_record` refuses to overwrite one, and queries read
only directories whose name is a bare run key, so a superseded run kept under
`<key>.<timestamp>.superseded` is visible on disk but never re-enters an
aggregate by accident.
"""

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from deviations import manifest as mf

RECORD_SCHEMA_VERSION = 1

CANNED_REPORTS = {
    "status": """
        SELECT status, count(*) AS runs
        FROM runs GROUP BY status ORDER BY runs DESC
    """,
    "wall": """
        SELECT method, configuration, width, count(*) AS repeats,
               avg(wall_time_s) AS mean_s, stddev_samp(wall_time_s) AS sd_s,
               min(wall_time_s) AS min_s, max(wall_time_s) AS max_s,
               sum(CASE WHEN status <> 'ok' THEN 1 ELSE 0 END) AS failed
        FROM runs GROUP BY method, configuration, width
        ORDER BY method, configuration, width
    """,
    "memory": """
        SELECT run_key, method, configuration, width,
               peak_device_mem_bytes, peak_host_rss_bytes,
               device_mem_attribution, sample_interval_ms
        FROM runs ORDER BY peak_device_mem_bytes DESC NULLS LAST
    """,
    "quality": """
        SELECT method, configuration, width, count(*) AS repeats,
               avg(CASE WHEN status = 'no_output' THEN 0 ELSE f1_primary END) AS mean_f1,
               stddev_samp(CASE WHEN status = 'no_output' THEN 0 ELSE f1_primary END) AS sd_f1,
               sum(CASE WHEN status = 'no_output' THEN 1 ELSE 0 END) AS no_output_runs,
               any_value(primary_tolerance) AS tolerance
        FROM runs GROUP BY method, configuration, width
        ORDER BY method, configuration, width
    """,
}


def is_run_key_dir(path):
    """Only bare run keys are part of the queryable store.

    `<key>.tmp`, `<key>.<timestamp>.aborted` and `<key>.<timestamp>.superseded`
    stay on disk as evidence and stay out of every aggregate.
    """
    return path.is_dir() and "." not in path.name and (path / "run.json").exists()


def finished_run_dirs(campaign_dir):
    campaign_dir = Path(campaign_dir)
    if not campaign_dir.is_dir():
        return []
    return sorted(p for p in campaign_dir.iterdir() if is_run_key_dir(p))


def iter_records(campaign_dir):
    for directory in finished_run_dirs(campaign_dir):
        with open(directory / "run.json") as f:
            yield json.load(f)


def deviations_document(manifest, repo_root):
    """The generated `deviations.json` embedded in every run record.

    Built from methods.yaml plus each fork's `upstream-base..HEAD` log, which
    CLAUDE.md defines as the exhaustive deviation list.
    """
    repo_root = Path(repo_root)
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_version": manifest.get("version"),
        "audit_source": manifest.get("audit_source"),
        "audit_date": str(manifest.get("audit_date")),
        "global_deviations": manifest.get("global_deviations") or [],
        "methods": [],
    }
    for entry in manifest.get("methods") or []:
        fork = repo_root / entry["path"]
        head = mf.head_sha(fork) if mf.is_initialised_submodule(fork) else None
        commits = mf.commits_since(fork, "upstream-base") if head else None
        document["methods"].append(
            {
                "name": entry["name"],
                "provenance": entry.get("provenance"),
                "status": entry.get("status", "active"),
                "path": entry["path"],
                "upstream": entry.get("upstream"),
                "upstream_base": entry.get("upstream_base"),
                "fork_sha": head,
                "harness_deviations": entry.get("harness_deviations") or [],
                "fork_deviations": [
                    {
                        "sha": commit["sha"],
                        "subject": commit["subject"],
                        **{
                            key.lower().replace("-", "_"): value
                            for key, value in mf.parse_trailers(commit["message"]).items()
                        },
                    }
                    for commit in (commits or [])
                ],
            }
        )
    return document


def invocation_context(spec, entry, run, result=None):
    """What this run actually executed, as the in-force test needs it.

    Taken from the run's own argv when there is one, and from the plan the
    runner would execute otherwise, so that `bench plan`-time callers see the
    same answer as a finished record.
    """
    from . import runner as rn

    preprocess = ((result or {}).get("preprocess") or {}).get("argv") or []
    command = (result or {}).get("command") or []
    method_argv = command[command.index(spec.image) + 1:] if spec.image in command else []
    if not preprocess or not method_argv:
        try:
            plan = rn.plan_commands(spec, entry, run)
        except rn.RunnerError:
            plan = {"preprocess": {"argv": []}, "measured": {"argv": []}}
        preprocess = preprocess or plan["preprocess"]["argv"]
        method_argv = method_argv or plan["measured"]["argv"]
    argv = [str(t) for t in list(preprocess) + list(method_argv)]
    template = [str(t) for t in (entry.get("invocation") or [])]
    executed = {t.split("=", 1)[0] for t in argv if t.startswith("-")}
    return {
        "flags": sorted(executed),
        # Flags methods.yaml puts in this method's invocation that this run did
        # not execute -- today only `debug_output: upstream` can produce one.
        # Their absence is a deliberate setting, not an inapplicable flag, and
        # the in-force reason has to say so.
        "removed_flags": sorted(
            {t.split("=", 1)[0] for t in template if t.startswith("-")} - executed
        ),
        # The shared converter is the one normalization whose presence is not a
        # flag: CUMVS preprocesses with its own initializer under every
        # configuration, so the substitution never happens for it.
        "shared_converter": any("colmap2mvsnet_acm_perf" in t for t in preprocess),
        "preprocess_argv": list(preprocess),
        "debug_output": spec.debug_output,
    }


def normalization_in_force(deviation, context, run):
    """(in force, why) for one declared `normalization` deviation.

    R-REP-01 asks what was in force for *this* run, so the answer is read off
    the executed invocation rather than off the manifest text: a global
    normalization that no argv carries (the `--no-debug-output` flag, which no
    fork implements yet) and one that does not apply to the method (the shared
    converter, for CUMVS) are not in force, whatever the manifest declares.
    """
    reversible = deviation.get("reversible")
    text = "" if reversible in (None, False, True) else str(reversible).strip()
    if not text or text.lower() in ("false", "none", "no"):
        return False, "no reversal is declared and no argv shows it in force"
    if text == "configuration author":
        if run.configuration == "author":
            return False, "reversed by the `author` configuration"
        if not context["shared_converter"]:
            return False, "this run did not preprocess with the shared converter"
        return True, "the shared converter preprocessed this run"
    if text.startswith("flag "):
        flag = text.split(None, 1)[1].split("=", 1)[0].split()[0]
        if flag in context["flags"]:
            return True, f"{flag} is in the executed argv"
        if flag in context.get("removed_flags", ()):
            return False, (
                f"{flag} is in this method's methods.yaml invocation but was removed "
                f"from this run's argv by the specification (`debug_output: "
                f"{context.get('debug_output')}`)"
            )
        return False, f"{flag} is not in the executed argv"
    return False, f"unrecognised reversal {text!r}; not claimed to be in force"


def declared_normalizations(manifest, entry, spec, run, result=None):
    """Every declared `normalization`, each with its in-force verdict."""
    context = invocation_context(spec, entry, run, result)
    declared = []
    for deviation in manifest.get("global_deviations") or []:
        if deviation.get("class") != "normalization" or deviation.get("status") != "active":
            continue
        declared.append({**deviation, "scope": "global"})
    for deviation in entry.get("harness_deviations") or []:
        if deviation.get("class") == "normalization":
            declared.append({**deviation, "scope": "method"})
    for deviation in declared:
        in_force, why = normalization_in_force(deviation, context, run)
        deviation["in_force"] = in_force
        deviation["in_force_reason"] = why
    return declared


def active_normalizations(manifest, entry, spec, run, result=None):
    """The `normalization` deviations in force for one run (R-REP-01)."""
    return [
        d for d in declared_normalizations(manifest, entry, spec, run, result) if d["in_force"]
    ]


def _summarise(text, limit=120):
    collapsed = " ".join(str(text).split())
    return collapsed[:limit]


def parameters(spec, manifest, entry, run, repo_root):
    """The complete parameter set the run was executed with (R-RUN-02).

    Most methods compile their parameters in, so what varies between runs of
    one fork is the configuration, the converter and, for MP-MVS, a config
    file whose contents are captured here.
    """
    configuration = (manifest.get("configurations") or {}).get(run.configuration, {})
    params = {
        "configuration": run.configuration,
        "neighbours": configuration.get("neighbours"),
        "converter_kind": configuration.get("converter"),
        "converter_args": spec.converter_args.get(run.configuration, []),
        "initializer_args": spec.initializer_args.get(run.configuration, []),
        "padding": spec.padding,
        "timeout_s": spec.timeout_s,
        "invocation": entry.get("invocation"),
        "config_file": None,
        "phase_timer": spec.phase_timer,
        "debug_output": spec.debug_output,
        "method_env": dict(spec.method_env),
        "seed_exposed": False,  # R-STA-04: no fork exposes one; recorded, not set.
    }
    config_path = entry.get("config_path")
    if config_path:
        host_path = Path(repo_root) / entry["path"] / Path(config_path).name
        candidates = [
            Path(repo_root) / entry["path"] / "config" / Path(config_path).name,
            host_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                params["config_file"] = {
                    "container_path": config_path,
                    "host_path": str(candidate),
                    "content": candidate.read_text(errors="replace"),
                }
                break
    return params


def converter_record(spec, entry, run, repo_root):
    """R-EXP-06: what preprocessed this run, and what identifies it.

    CUMVS's initializer is a binary that exists only inside the image, so the
    host has no file to hash; its identity is the hash of the source the image
    was built from (`identity: "source"`, D24 / M-006 open 3), and the image
    manifest in the fingerprint binds that source to the binary that ran.
    """
    from . import image_manifest as imf
    from .evaluate import sha256_file

    record = {
        "kind": None,
        "path": None,
        "sha256": None,
        "identity": "file",
        "source_path": None,
        "source_sha256": None,
    }
    if entry.get("converter") is None:
        source = imf.initializer_source(entry)
        record.update(
            kind="initializer",
            # The container path, because that is where the artefact that ran
            # lives; the host copy of it does not exist.
            path=f"{spec.container['method_root']}/{Path(entry['path']).name}/"
            f"{entry.get('initializer') or ''}",
            identity="source",
            source_path=source,
            source_sha256=(
                imf.converter_identity(repo_root, entry).get("sha256") if source else None
            ),
        )
        return record
    if run.configuration == "author":
        path = Path(repo_root) / entry["path"] / entry["converter"]
        record["kind"] = "fork"
    else:
        path = Path(__file__).resolve().parents[1] / "eval" / "colmap2mvsnet_acm_perf.py"
        record["kind"] = "shared"
    try:
        digest = sha256_file(path)
    except OSError:
        digest = None
    record.update(path=str(path), sha256=digest)
    return record


def build_record(spec, manifest, entry, run, result, fingerprint, deviations, repo_root):
    """Assemble run.json (R-STO-03).

    Free-form structures (`status_evidence`, `parameters`, `deviations`) are
    stored as JSON strings next to flat scalars, so that DuckDB's
    `read_json_auto` unifies every record of a campaign without preprocessing
    (R-STO-06) no matter how the evidence of one failure differs from another.
    """
    from . import runner as rn

    measured_env = rn.method_env(spec)
    telemetry = result.get("telemetry") or {}
    quality = result.get("quality")
    primary = None
    if quality:
        primary = next(
            (q for q in quality if abs(q["tolerance"] - spec.primary_tolerance) < 1e-9), None
        )
    cloud = result.get("point_cloud") or {}
    declared = declared_normalizations(manifest, entry, spec, run, result)
    normalizations = [d for d in declared if d["in_force"]]
    preprocess = result.get("preprocess") or {}
    fork_sha = next(
        (
            m["fork_sha"]
            for m in deviations.get("methods", [])
            if m["name"] == entry["name"]
        ),
        None,
    )

    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "run_key": run.key,
        "campaign": spec.campaign,
        "method": run.method,
        "scene": run.scene,
        "width": run.width,
        "configuration": run.configuration,
        "repeat": run.repeat,
        "provenance": entry.get("provenance"),
        "status": result.get("status"),
        "exit_code": result.get("exit_code"),
        "timed_out": bool((result.get("status_evidence") or {}).get("timed_out")),
        "status_evidence_json": json.dumps(result.get("status_evidence") or {}, default=str),
        "wall_time_s": result.get("wall_time_s"),
        "wall_time_with_preprocess_s": result.get("wall_time_with_preprocess_s"),
        "container_wall_time_s": result.get("container_wall_time_s"),
        "container_startup_s": result.get("container_startup_s"),
        "preprocess_convert_s": preprocess.get("duration_s"),
        # R-EXP-06/R-EXP-08: the neighbour count of a normalized configuration
        # reaches the method only through this argv, so it is recorded as run,
        # not only as the configuration that was asked for.
        "preprocess_argv": preprocess.get("argv") or [],
        "container_started_at": result.get("container_started_at"),
        "container_finished_at": result.get("container_finished_at"),
        "clock_hold": result.get("clock_hold")
        or {
            "expected_mhz": None,
            "min_mhz": None,
            "max_mhz": None,
            "samples": 0,
            "fraction_at_expected": None,
            "held": None,
            "method": None,
        },
        "gpu_state_evidence_json": json.dumps(
            result.get("gpu_state_evidence") or {}, default=str
        ),
        "warnings": result.get("warnings") or [],
        "phases": result.get("phases") or [],
        "phases_by_pass": result.get("phases_by_pass") or [],
        "phase_trace_errors": result.get("phase_trace_errors") or [],
        # R-TIM-09: false means the measured process was given no
        # MVS_BENCH_PHASES, so it wrote no trace and `phases` holds only the
        # harness's own spans. An absent phases.txt is then expected, not a
        # failure.
        "phase_timer": spec.phase_timer,
        "instrumented": spec.phase_timer,
        # R-EXP-11: which half of the debug-artefact pair this run is, and why.
        "debug_output": rn.debug_output_state(spec, entry),
        # R-TIM-08: the whole environment the measured process was given, as a
        # string so that DuckDB unifies campaigns that set different keys, plus
        # the one key the SYNC pair turns on, flat enough to group by.
        "method_env_json": json.dumps(measured_env, sort_keys=True),
        "mvs_bench_sync": measured_env.get("MVS_BENCH_SYNC"),
        "intermediates_bytes": result.get("intermediates_bytes"),
        "sample_interval_ms": telemetry.get("sample_interval_ms"),
        "sample_count": telemetry.get("sample_count"),
        "sampler_missed_ticks": telemetry.get("missed_ticks"),
        "peak_host_rss_bytes": telemetry.get("peak_host_rss_bytes"),
        "peak_host_uss_bytes": telemetry.get("peak_host_uss_bytes"),
        "peak_device_mem_bytes": telemetry.get("peak_device_mem_bytes"),
        "device_mem_attribution": telemetry.get("device_mem_attribution"),
        "device_mem_total_bytes": telemetry.get("device_mem_total_bytes"),
        "energy_j": telemetry.get("energy_j"),
        "mean_gpu_power_w": telemetry.get("mean_gpu_power_w"),
        "peak_gpu_temp_c": telemetry.get("peak_gpu_temp_c"),
        "io_read_bytes": telemetry.get("io_read_bytes"),
        "io_write_bytes": telemetry.get("io_write_bytes"),
        "io_attribution": telemetry.get("io_attribution"),
        "telemetry_json": json.dumps(telemetry, default=str),
        "quality": quality or [],
        "primary_tolerance": spec.primary_tolerance,
        "f1_primary": (primary or {}).get("f1"),
        "accuracy_primary": (primary or {}).get("accuracy"),
        "completeness_primary": (primary or {}).get("completeness"),
        "evaluator_sha": spec.evaluator_sha,
        "point_cloud_path": cloud.get("path"),
        "point_cloud_size_bytes": cloud.get("size_bytes"),
        "point_cloud_sha256": cloud.get("sha256"),
        "point_count": cloud.get("point_count"),
        "fork_sha": fork_sha,
        "upstream_base": entry.get("upstream_base"),
        "converter": converter_record(spec, entry, run, repo_root),
        "neighbour_list": result.get("neighbour_list") or {"path": None, "sha256": None},
        "parameters_json": json.dumps(
            parameters(spec, manifest, entry, run, repo_root), default=str
        ),
        "normalizations": [_summarise(n.get("description")) for n in normalizations],
        "normalizations_json": json.dumps(normalizations, default=str),
        # Everything the manifest declares for this method, each with the
        # verdict above: the declared text stays readable next to what the run
        # actually carried.
        "declared_normalizations_json": json.dumps(declared, default=str),
        "deviations_json": json.dumps(deviations, default=str),
        "fingerprint": fingerprint,
        "image": spec.image,
        "image_digest": fingerprint.get("image_digest"),
        # D24.1: the image's own build manifest, which `bench run` checked
        # against the working tree before the campaign started. The manifest
        # itself is in `fingerprint.image_manifest`.
        "image_manifest_sha256": fingerprint.get("image_manifest_sha256"),
        "gpu_index": spec.gpu_index,
        "hostname": socket.gethostname(),
        "spec_sha256": spec.sha256,
        "spec_path": str(spec.path),
        "order_seed": spec.order_seed,
        "shard_index": spec.shard_index,
        "shard_count": spec.shard_count,
        "started_at": result.get("started_at"),
        "ended_at": result.get("ended_at"),
        "command": result.get("command") or [],
        "notes": result.get("notes") or [],
        "warm_cache_bytes": result.get("warm_cache_bytes"),
        "work_dir": result.get("work_dir"),
    }


def write_record(tmp_dir, record):
    """R-RUN-03: run.json is written last, into the `.tmp` directory."""
    path = Path(tmp_dir) / "run.json"
    if path.exists():
        raise FileExistsError(f"{path}: a run record is never overwritten")
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return path


def check_configuration_consistency(campaign_dir, run, parameters_json, fork_sha):
    """R-RUN-02: one configuration name means one parameter set and one fork.

    Returns a list of conflicts; a non-empty list means the run may not be
    executed.
    """
    conflicts = []
    for record in iter_records(campaign_dir):
        if record.get("configuration") != run.configuration:
            continue
        if record.get("method") != run.method:
            continue
        if record.get("fork_sha") != fork_sha:
            conflicts.append(
                f"{record['run_key']} ran fork {record.get('fork_sha')}, this run "
                f"would run {fork_sha}"
            )
        if record.get("parameters_json") != parameters_json:
            conflicts.append(
                f"{record['run_key']} used a different parameter set under the same "
                f"configuration name `{run.configuration}`"
            )
    return conflicts


def init_campaign(spec, fingerprint_data, deviations):
    """Write spec.yaml, fingerprint.json and deviations.json (R-STO-02).

    The specification is copied verbatim (R-EXP-02) and never overwritten: a
    campaign whose specification changed is a different campaign.
    """
    from . import fingerprint as fp

    campaign_dir = spec.campaign_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    spec_copy = campaign_dir / "spec.yaml"
    if spec_copy.exists():
        if spec_copy.read_text() != spec.text:
            raise FileExistsError(
                f"{spec_copy} differs from {spec.path}. A campaign is bound to one "
                "specification; start a new campaign rather than editing this one."
            )
    else:
        spec_copy.write_text(spec.text)

    bound = fp.bind(campaign_dir, fingerprint_data, spec.declared_fingerprint)
    (campaign_dir / "deviations.json").write_text(
        json.dumps(deviations, indent=2, sort_keys=True, default=str) + "\n"
    )
    return bound


def status_report(campaign_dir, runs=None):
    """R-OBS-01: reads only the store."""
    campaign_dir = Path(campaign_dir)
    by_status = {}
    finished = set()
    for record in iter_records(campaign_dir):
        by_status[record.get("status")] = by_status.get(record.get("status"), 0) + 1
        finished.add(record.get("run_key"))
    in_flight = []
    for directory in sorted(campaign_dir.glob("*.tmp")):
        started = datetime.fromtimestamp(directory.stat().st_mtime, tz=timezone.utc)
        host = None
        marker = directory / "host.txt"
        if marker.exists():
            host = marker.read_text().strip()
        in_flight.append(
            {
                "run_key": directory.name[: -len(".tmp")],
                "elapsed_s": (datetime.now(timezone.utc) - started).total_seconds(),
                "hostname": host,
                "alive": marker_is_alive(host),
            }
        )
    report = {
        "campaign": campaign_dir.name,
        "by_status": by_status,
        "finished": len(finished),
        "in_flight": in_flight,
    }
    if runs is not None:
        report["remaining"] = [r.key for r in runs if r.key not in finished]
        report["planned"] = len(runs)
    return report


def marker_is_alive(marker):
    """Is the process that wrote this `host.txt` still running?

    True/False only when the marker names this host, because a pid means
    nothing on another one; None otherwise. A `.tmp` whose writer is gone is
    the residue of a crash or a kill, and R-OBS-01 is useless if it cannot be
    told apart from a run in flight. Pid reuse can say True for a stranger; the
    marker keeps the pid so the operator can check.
    """
    if not marker:
        return None
    host, _, tail = marker.partition(" pid=")
    if host.strip() != socket.gethostname() or not tail.strip().isdigit():
        return None
    import psutil

    return psutil.pid_exists(int(tail.strip()))


def connect(campaign_dir):
    """A DuckDB connection with `runs` and `telemetry` over the store."""
    import duckdb

    campaign_dir = Path(campaign_dir)
    directories = finished_run_dirs(campaign_dir)
    if not directories:
        raise FileNotFoundError(f"{campaign_dir}: no finished runs")
    con = duckdb.connect()
    records = [str(d / "run.json") for d in directories]
    # A view definition cannot carry prepared parameters, so the file list is
    # inlined as a SQL list literal with quotes escaped.
    con.execute(
        f"CREATE VIEW runs AS SELECT * FROM "
        f"read_json_auto({_sql_list(records)}, union_by_name=true)"
    )
    telemetry = [
        str(d / "telemetry.parquet") for d in directories if (d / "telemetry.parquet").exists()
    ]
    if telemetry:
        con.execute(
            f"CREATE VIEW telemetry AS SELECT * FROM "
            f"read_parquet({_sql_list(telemetry)}, filename=true)"
        )
    return con


def _sql_list(paths):
    return "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def format_table(con, statement):
    """Render a query as text without depending on pandas."""
    cursor = con.execute(statement)
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    widths = [
        max(len(columns[i]), *(len(_cell(r[i])) for r in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]
    lines = [" | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(_cell(v).ljust(widths[i]) for i, v in enumerate(row)))
    return "\n".join(lines) + f"\n({len(rows)} rows)"


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_host_marker(tmp_dir):
    """Who is executing an in-flight run (R-OBS-01)."""
    Path(tmp_dir, "host.txt").write_text(f"{socket.gethostname()} pid={os.getpid()}\n")
