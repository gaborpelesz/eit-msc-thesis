"""`bench` -- run a campaign and query its result store.

    bench plan   <spec.yaml> [--pilot DIR|FILE]
    bench run    <spec.yaml> [--only KEY] [--dry-run] [--rerun]
    bench fingerprint [--image TAG]
    bench status <campaign_dir> [--spec spec.yaml]
    bench query  <campaign_dir> [--sql SQL | --report NAME]
    bench phases <run_dir>
"""

import argparse
import json
import signal
import sys
import traceback
from pathlib import Path

from deviations import manifest as mf
from deviations import verify as vf

from . import evaluate as ev
from . import fingerprint as fp
from . import phases as ph
from . import runner as rn
from . import spec as sp
from . import store as st


def _add_manifest(parser):
    parser.add_argument(
        "--manifest",
        default=str(mf.default_manifest_path()),
        help="path to methods.yaml (default: benchmark/methods/methods.yaml)",
    )


def _load(spec_path, manifest_path):
    manifest = mf.load(manifest_path)
    return sp.load(spec_path, manifest), manifest


def _entry(manifest, name):
    return sp.method_entry(manifest, name)


def cmd_plan(args):
    spec, manifest = _load(args.spec, args.manifest)
    runs = sp.expand(spec)
    for index, run in enumerate(runs, start=1):
        print(f"{index:4d}  {run.key}")
    print(f"\n{len(runs)} runs  (campaign {spec.campaign}, order seed {spec.order_seed}, "
          f"shard {spec.shard_index + 1}/{spec.shard_count}, timeout {spec.timeout_s}s)")
    if args.pilot:
        estimates, unknown = _project(runs, args.pilot)
        print(
            f"projected {estimates / 3600:.1f} h from the pilot"
            + (f"; {unknown} runs have no pilot datum and are not counted" if unknown else "")
        )
    return 0


def _project(runs, pilot):
    """Projected seconds from a pilot campaign directory or a JSON mapping."""
    pilot = Path(pilot)
    table = {}
    if pilot.is_dir():
        for record in st.iter_records(pilot):
            key = (record["method"], record["configuration"], record["width"])
            table.setdefault(key, []).append(record.get("wall_time_with_preprocess_s") or 0.0)
        table = {k: sum(v) / len(v) for k, v in table.items()}
    else:
        raw = json.loads(pilot.read_text())
        table = {tuple(k.split("|")) if "|" in k else k: v for k, v in raw.items()}

    total = 0.0
    unknown = 0
    for run in runs:
        value = table.get((run.method, run.configuration, run.width))
        if value is None:
            value = table.get((run.method, run.configuration, str(run.width)))
        if value is None:
            value = table.get(run.method)
        if value is None:
            unknown += 1
            continue
        total += float(value)
    return total, unknown


def cmd_fingerprint(args):
    root = mf.repo_root(Path(args.manifest).parent)
    data = fp.collect(args.image, gpu_index=args.gpu, repo_root=root, probe_image=not args.no_probe)
    print(json.dumps(data, indent=2, sort_keys=True))
    problems, evidence = fp.check_gpu_state(args.gpu)
    print("\nR-ENV-02 environment check:", file=sys.stderr)
    print(json.dumps(evidence, indent=2, sort_keys=True), file=sys.stderr)
    for problem in problems:
        print(f"  NOT READY: {problem}", file=sys.stderr)
    return 1 if problems else 0


def cmd_status(args):
    runs = None
    if args.spec:
        spec, _ = _load(args.spec, args.manifest)
        runs = sp.expand(spec)
    report = st.status_report(args.campaign_dir, runs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_query(args):
    con = st.connect(args.campaign_dir)
    if args.sql:
        print(st.format_table(con, args.sql))
        return 0
    for name in ([args.report] if args.report else list(st.CANNED_REPORTS)):
        print(f"\n== {name} ==")
        print(st.format_table(con, st.CANNED_REPORTS[name]))
    return 0


def cmd_phases(args):
    run_dir = Path(args.run_dir)
    trace = ph.parse_file(run_dir / "phases.txt")
    harness = ph.parse_file(run_dir / "phases.harness.txt")
    print(f"== {run_dir.name}: method phases ==")
    print(ph.format_tree(trace, max_depth=args.depth) or "(no in-process phase trace)")
    print("\n== totals (total_s includes nested spans, self_s does not) ==")
    header = f"{'phase':<34} {'total_s':>10} {'self_s':>10} {'count':>7} {'unclosed':>9}"
    print(header)
    print("-" * len(header))
    for row in ph.totals(trace) + ph.totals(harness):
        print(
            f"{row['name']:<34} {row['total_s']:>10.3f} {row['self_s']:>10.3f} "
            f"{row['count']:>7} {row['unclosed']:>9}"
        )
    by_pass = ph.totals_by_pass(trace)
    if by_pass:
        print("\n== by pass (the method's own vocabulary; mapped in methods.yaml) ==")
        for row in by_pass:
            print(f"{row['name']:<28} {row['pass']:<20} {row['total_s']:>10.3f} {row['count']:>7}")
    for error in trace.errors[:20] + harness.errors[:20]:
        print(f"  ! {error}", file=sys.stderr)
    return 0


_STOP = {"requested": False}


def _interrupted():
    return _STOP["requested"]


def _install_stop_handlers():
    """R-RES-01: the first signal pauses between runs, the second aborts.

    Pausing between runs means the run in flight still gets a record, so a
    stopped campaign loses no measurement.
    """

    def handler(signum, frame):
        if _STOP["requested"]:
            raise KeyboardInterrupt
        _STOP["requested"] = True
        print(
            "\nstop requested: the run in flight will be finished and recorded, "
            "then the campaign exits. Signal again to abort it.",
            file=sys.stderr,
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


def cmd_run(args):
    spec, manifest = _load(args.spec, args.manifest)
    root = mf.repo_root(Path(args.manifest).parent)
    runs = sp.expand(spec)
    if args.only:
        runs = [r for r in runs if r.key in set(args.only)]
        if not runs:
            print(f"no run in this specification matches {args.only}", file=sys.stderr)
            return 2

    if args.dry_run:
        for run in runs:
            entry = _entry(manifest, run.method)
            plan = rn.plan_commands(spec, entry, run, docker=args.docker)
            print(f"\n=== {run.key} ===")
            print(f"work dir     : {plan['work_dir']}")
            print(f"run dir      : {spec.campaign_dir / run.key}")
            print(f"preprocess   : {' '.join(plan['preprocess']['docker'])}")
            print(f"measured     : {' '.join(plan['measured']['docker'])}")
            print(f"evaluation   : {' '.join(plan['evaluation']['docker'])}")
        print(
            f"\n{len(runs)} runs planned; nothing was executed and "
            "`deviations verify` was not run (no measurement is produced by a dry run)."
        )
        return 0

    _install_stop_handlers()

    # R-ENV-01: the deviation verifier gates every benchmark run.
    if vf.verify(args.manifest) != 0:
        print(
            "\n`deviations verify` is failing; no measurement may be produced "
            "from code whose deviations are not recorded (R-ENV-01).",
            file=sys.stderr,
        )
        return 1

    # R-ENV-02: locked clocks, persistence mode, no display.
    problems, evidence = fp.check_gpu_state(spec.gpu_index)
    if problems:
        print("the benchmark GPU is not in a measurement state (R-ENV-02):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(json.dumps(evidence, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    fingerprint = fp.collect(spec.image, gpu_index=spec.gpu_index, repo_root=root)
    deviations = st.deviations_document(manifest, root)
    try:
        # R-ENV-06 / R-RES-04: a campaign is bound to one machine.
        fingerprint = st.init_campaign(spec, fingerprint, deviations)
    except fp.FingerprintMismatch as exc:
        print(str(exc), file=sys.stderr)
        return 1

    executed = 0
    failures = 0
    for run in runs:
        if _interrupted():
            return 130
        if st.is_finished(spec.campaign_dir, run.key) and not args.rerun:
            print(f"skip     {run.key} (finished)")
            continue
        # R-ENV-02 is a precondition of every run, not only of the campaign:
        # a clock lock can be lost between runs.
        problems, evidence = fp.check_gpu_state(spec.gpu_index)
        if problems:
            print(f"stop before {run.key}: the GPU left its measurement state:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        entry = _entry(manifest, run.method)
        try:
            tmp_dir, moved = rn.open_run_dir(spec.campaign_dir, run.key, rerun=args.rerun)
        except rn.RunnerError as exc:
            print(f"skip     {run.key}: {exc}")
            continue
        for path in moved:
            print(f"aside    {path}")
        st.write_host_marker(tmp_dir)

        parameters_json = json.dumps(
            st.parameters(spec, manifest, entry, run, root), default=str
        )
        fork_sha = next(
            (m["fork_sha"] for m in deviations["methods"] if m["name"] == entry["name"]), None
        )
        conflicts = st.check_configuration_consistency(
            spec.campaign_dir, run, parameters_json, fork_sha
        )
        if conflicts:
            print(f"refuse   {run.key}: R-RUN-02 conflict", file=sys.stderr)
            for conflict in conflicts:
                print(f"  - {conflict}", file=sys.stderr)
            rn.move_aside(tmp_dir, "aborted")
            return 1

        print(f"run      {run.key}")
        try:
            result = rn.execute(spec, entry, run, tmp_dir, docker=args.docker)
            result = rn.collect_point_cloud(spec, entry, run, result, tmp_dir)
            result = rn.run_evaluation(spec, run, result, tmp_dir, docker=args.docker)
            result = rn.discard_intermediates(spec, result)
            record = st.build_record(
                spec, manifest, entry, run, result, fingerprint, deviations, root
            )
            st.write_record(tmp_dir, record)
            final = rn.commit_run_dir(tmp_dir)
        except KeyboardInterrupt:
            print(
                f"\ninterrupted; {tmp_dir} is left in place and will be renamed aside "
                "on the next attempt (R-RES-01, R-RES-03).",
                file=sys.stderr,
            )
            return 130
        except Exception:
            # A harness error is not a run outcome and must not become a
            # record; the attempt is kept aside as evidence and the campaign
            # continues (R-FAIL-02).
            (tmp_dir / "harness_error.txt").write_text(traceback.format_exc())
            aside = rn.move_aside(tmp_dir, "aborted")
            print(f"ERROR    {run.key}: harness failure, attempt kept at {aside}", file=sys.stderr)
            traceback.print_exc()
            failures += 1
            continue

        executed += 1
        if record["status"] != "ok":
            failures += 1
        print(
            f"  -> {record['status']:<12} wall {record['wall_time_s'] or float('nan'):.1f}s"
            f"  f1 {record['f1_primary'] if record['f1_primary'] is not None else 'n/a'}"
            f"  {final}"
        )

    print(f"\n{executed} runs executed, {failures} not ok")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="print the run list and a projection")
    plan.add_argument("spec")
    plan.add_argument("--pilot", help="a pilot campaign directory or a JSON seconds map")
    _add_manifest(plan)
    plan.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="execute the campaign")
    run.add_argument("spec")
    run.add_argument("--only", nargs="+", help="execute only these run keys")
    run.add_argument("--dry-run", action="store_true", help="print the exact docker commands")
    run.add_argument(
        "--rerun",
        action="store_true",
        help="re-execute finished runs, moving the existing record aside (R-FAIL-04)",
    )
    run.add_argument("--docker", default="docker")
    _add_manifest(run)
    run.set_defaults(func=cmd_run)

    fingerprint = subparsers.add_parser("fingerprint", help="print this machine's fingerprint")
    fingerprint.add_argument("--image", default=sp.DEFAULT_IMAGE)
    fingerprint.add_argument("--gpu", type=int, default=0)
    fingerprint.add_argument("--no-probe", action="store_true", help="do not start a container")
    _add_manifest(fingerprint)
    fingerprint.set_defaults(func=cmd_fingerprint)

    status = subparsers.add_parser("status", help="counts by status, in flight, remaining")
    status.add_argument("campaign_dir")
    status.add_argument("--spec", help="the specification, to report what remains")
    _add_manifest(status)
    status.set_defaults(func=cmd_status)

    query = subparsers.add_parser("query", help="DuckDB over the result store")
    query.add_argument("campaign_dir")
    query.add_argument("--sql")
    query.add_argument("--report", choices=sorted(st.CANNED_REPORTS))
    query.set_defaults(func=cmd_query)

    phases = subparsers.add_parser("phases", help="pretty-print a run's phase trace")
    phases.add_argument("run_dir")
    phases.add_argument("--depth", type=int, default=None)
    phases.set_defaults(func=cmd_phases)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (sp.SpecError, rn.RunnerError, ev.EvaluationFailed, FileExistsError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2
    except fp.FingerprintMismatch as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
