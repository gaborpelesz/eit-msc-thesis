"""Execution of one run: containers, timeout, status classification, atomicity.

R-RUN-03, R-RUN-05, R-RUN-06, R-RUN-07, R-EXP-04, R-EXP-06, R-EXP-07,
R-TIM-02, R-TIM-03, R-TIM-04, R-IO-03, R-FAIL-01..R-FAIL-04, R-RES-02,
R-RES-03, R-STO-05, R-OBS-02, R-ART-02, R-ART-03.

Two containers are created per run: one for the preprocessing step, timed as
the harness phase `preprocess.convert`, and one fresh container for the
measured process itself, so that nothing but the method is inside the measured
region. Both are removed afterwards.

Nothing under a campaign directory is ever deleted here. A leftover `.tmp`
directory and a superseded finished run are renamed aside with a timestamp;
the only thing this module removes is the scratch working directory it created
itself outside the store (R-ART-02).
"""

import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from deviations import manifest as mf

from . import evaluate as ev
from . import fingerprint as fpr
from . import phases as ph
from . import sampler as smp

STATUSES = (
    "ok",
    "oom_gpu",
    "oom_host",
    "segfault",
    "timeout",
    "nonzero_exit",
    "no_output",
    "eval_failed",
)

CUDA_OOM_MARKERS = (
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudamalloc",
)
HOST_OOM_MARKERS = ("std::bad_alloc", "cannot allocate memory", "out of memory: killed")
SEGFAULT_MARKERS = ("segmentation fault", "sigsegv")

# The shared flag of R-EXP-11, as methods.yaml spells it in seven invocations.
DEBUG_OUTPUT_FLAG = "--no-debug-output"


class RunnerError(Exception):
    pass


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def move_aside(path, tag):
    """Rename `path` out of the way. Never deletes; returns the new path."""
    path = Path(path)
    target = path.with_name(f"{path.name}.{utc_stamp()}.{tag}")
    suffix = 1
    while target.exists():
        suffix += 1
        target = path.with_name(f"{path.name}.{utc_stamp()}.{tag}.{suffix}")
    path.rename(target)
    return target


def is_finished(campaign_dir, key):
    """R-RUN-03: a run is finished iff `<key>/run.json` exists."""
    return (Path(campaign_dir) / key / "run.json").exists()


def open_run_dir(campaign_dir, key, rerun=False):
    """Prepare `<key>.tmp` for a fresh attempt.

    A leftover `.tmp` from an interrupted attempt is renamed aside rather than
    discarded: R-RES-03 asks for those runs to be re-executed from the
    beginning, and repository policy forbids deleting anything in the store, so
    the aborted attempt is kept under `<key>.tmp.<timestamp>.aborted`.
    """
    campaign_dir = Path(campaign_dir)
    final = campaign_dir / key
    tmp = campaign_dir / f"{key}.tmp"
    moved = []
    if final.exists():
        if not rerun:
            raise RunnerError(f"{key}: already finished; pass --rerun to supersede it")
        moved.append(str(move_aside(final, "superseded")))
    if tmp.exists():
        moved.append(str(move_aside(tmp, "aborted")))
    tmp.mkdir(parents=True)
    return tmp, moved


def commit_run_dir(tmp_dir):
    """R-RUN-03: rename `<key>.tmp` to `<key>` once run.json is written."""
    tmp_dir = Path(tmp_dir)
    final = tmp_dir.with_name(tmp_dir.name[: -len(".tmp")])
    if not (tmp_dir / "run.json").exists():
        raise RunnerError(f"{tmp_dir}: refusing to commit a run without run.json")
    if final.exists():
        raise RunnerError(f"{final}: exists; refusing to overwrite a finished run")
    tmp_dir.rename(final)
    return final


def container_method_root(spec, entry):
    """The image mirrors each fork root at <method_root>/<fork directory>."""
    return f"{spec.container['method_root']}/{Path(entry['path']).name}"


def placeholder_values(spec, entry, run):
    root = container_method_root(spec, entry)
    dataset_dir = f"{spec.container['work']}/prepared"
    output_ply = entry.get("output_ply") or ""
    output_dir = f"{dataset_dir}/{Path(output_ply).parent}" if output_ply else dataset_dir
    return {
        "executable": f"{root}/{entry['executable']}",
        "initializer": f"{root}/{entry['initializer']}" if entry.get("initializer") else "",
        "dataset_dir": dataset_dir,
        "raw_scene_dir": spec.container["data"],
        "output_dir": output_dir,
        # The container is given exactly one device, so the method's own device
        # index is always 0; the host index is recorded in the fingerprint.
        "gpu_index": "0",
        "config": entry.get("config_path") or "",
        # A method whose RNG seed is a required argument has no default to fall
        # back on, and a campaign's repeats must not all draw the same one:
        # PatchMatch is randomized and R-STA-01's repeats exist to measure that
        # spread. The repeat index is the only per-run number the harness has
        # that is stable across a re-run of the same specification, so it is
        # what such a method's seed is written in terms of.
        "repeat": str(run.repeat),
    }


def debug_output_state(spec, entry):
    """R-EXP-11 for one run: was the debug-artefact normalization applied?

    Seven of the ten campaign methods take `--no-debug-output`; ACMH and ACMM
    write no debug artefacts and CUMVS's OpenCV CommandLineParser rejects the
    unknown key, so for those three the setting is inapplicable rather than
    reversed, and the record says which of the two it is.
    """
    has_flag = DEBUG_OUTPUT_FLAG in (entry.get("invocation") or [])
    passed = has_flag and spec.debug_output == "off"
    if not has_flag:
        reason = (
            f"{entry.get('name')}'s invocation in methods.yaml carries no "
            f"{DEBUG_OUTPUT_FLAG}, so `debug_output` changes nothing for it "
            "and R-EXP-11 needs no action"
        )
    elif passed:
        reason = f"{DEBUG_OUTPUT_FLAG} is in the executed argv; the debug writes are skipped"
    else:
        reason = (
            f"`debug_output: upstream` removed {DEBUG_OUTPUT_FLAG} from the "
            "invocation: this run writes the debug artefacts its authors shipped "
            "it writing, and the R-EXP-11 normalization is NOT in force"
        )
    return {
        "setting": spec.debug_output,
        "method_has_flag": has_flag,
        "flag_passed": passed,
        "normalization_in_force": passed,
        "reason": reason,
    }


def resolve_invocation(spec, entry, run):
    values = placeholder_values(spec, entry, run)
    argv = []
    tokens = list(entry["invocation"])
    if spec.debug_output == "upstream":
        # The token, not a value: DPE-MVS's DEBUG_COMPLEX block is compiled in
        # and guarded by this flag at run time, so dropping it restores the
        # released behaviour of that guard too (methods.yaml, DPE-MVS).
        tokens = [t for t in tokens if t != DEBUG_OUTPUT_FLAG]
    for token in tokens:
        rendered = token.format(**values)
        if rendered == "":
            raise RunnerError(
                f"{run.key}: invocation token {token!r} has no value in methods.yaml"
            )
        argv.append(rendered)
    return argv


def preprocess_argv(spec, entry, run):
    """The converter (or CUMVS's initialiser) as a container argv.

    Under `author` the fork's own converter is used (R-EXP-06); under any other
    configuration the shared converter, with the configuration's arguments
    (R-EXP-08). A method declaring `converter_source: shared` ships no
    converter of its own, so the shared one prepares its input in every
    configuration, `author` included.
    """
    values = placeholder_values(spec, entry, run)
    shared_converter = mf.converter_source(entry) == "shared"
    if entry.get("converter") is None and not shared_converter:
        if not entry.get("initializer_invocation"):
            raise RunnerError(f"{entry['name']}: no converter and no initializer in methods.yaml")
        argv = [t.format(**values) for t in entry["initializer_invocation"]]
        return argv + list(spec.initializer_args.get(run.configuration, []))

    if run.configuration == "author" and not shared_converter:
        converter = f"{container_method_root(spec, entry)}/{entry['converter']}"
        extra = []
    else:
        converter = spec.container.get(
            "shared_converter", "/sota/.venv/bin/colmap2mvsnet_acm_perf"
        )
        extra = list(spec.converter_args.get(run.configuration, []))
        if spec.padding == "all":
            extra.append("--padding")

    if converter.endswith(".py"):
        head = [spec.container["python"], converter]
    else:
        head = [converter]
    return head + [
        "--dense_folder",
        values["raw_scene_dir"],
        "--save_folder",
        values["dataset_dir"],
    ] + extra


# The ACM-lineage converters read the calibration from `<dense_folder>/sparse`
# (a COLMAP dense workspace); APD-MVS, DPE-MVS and CUMVS read ETH3D's
# `dslr_calibration_undistorted` directly. Both layouts are served by binding
# the calibration directory under both names, so no fork's converter is edited
# and no files are copied (R-EXP-06).
def scene_mounts(spec, run):
    raw = spec.raw_scene_dir(run.scene, run.width)
    data = spec.container["data"]
    pairs = [
        ("images", "images"),
        ("dslr_calibration_undistorted", "dslr_calibration_undistorted"),
        ("dslr_calibration_undistorted", "sparse"),
    ]
    mounts = []
    for host_sub, container_sub in pairs:
        mounts += ["-v", f"{raw / host_sub}:{data}/{container_sub}:ro"]
    return mounts


def docker_run_argv(spec, run, argv, work_dir, detach, name=None, docker="docker", env=None):
    entrypoint = argv[0]
    mounts = scene_mounts(spec, run) + [
        "-v",
        f"{work_dir / 'prepared'}:{spec.container['work']}/prepared",
        "-v",
        f"{work_dir / 'out'}:{spec.container['out']}",
    ]
    command = [docker, "run"]
    command += ["-d"] if detach else ["--rm"]
    if name:
        command += ["--name", name]
    command += ["--gpus", f"device={spec.gpu_index}"]
    for key, value in (env or {}).items():
        command += ["-e", f"{key}={value}"]
    command += mounts
    command += ["--entrypoint", entrypoint, spec.image, *argv[1:]]
    return command


def method_env(spec):
    """Environment of the measured process.

    The phase timer is gated by `MVS_BENCH_PHASES`; a specification with
    `phase_timer: false` leaves it unset, which is the uninstrumented half of
    the overhead pair R-TIM-09 asks for. `method_env` carries anything else a
    campaign needs to set, `MVS_BENCH_SYNC` for CUMVS above all (R-TIM-08).
    """
    env = {}
    if spec.phase_timer:
        env["MVS_BENCH_PHASES"] = "1"
        env["MVS_BENCH_FILE"] = f"{spec.container['out']}/phases.txt"
    env.update({str(k): str(v) for k, v in spec.method_env.items()})
    return env


def plan_commands(spec, entry, run, docker="docker"):
    """Everything the runner would execute, for `--dry-run` and the record."""
    work_dir = spec.work_dir(run)
    method_argv = resolve_invocation(spec, entry, run)
    convert_argv = preprocess_argv(spec, entry, run)
    env = method_env(spec)
    return {
        "work_dir": str(work_dir),
        "preprocess": {
            "argv": convert_argv,
            "docker": docker_run_argv(
                spec, run, convert_argv, work_dir, detach=False, docker=docker
            ),
        },
        "measured": {
            "argv": method_argv,
            "env": env,
            "debug_output": debug_output_state(spec, entry),
            "docker": docker_run_argv(
                spec,
                run,
                method_argv,
                work_dir,
                detach=True,
                name=f"bench_{run.key}",
                docker=docker,
                env=env,
            ),
        },
        "evaluation": {
            "docker": ev.evaluation_argv(
                spec,
                Path(spec.artifact_path(run)),
                run.scene,
                run.width,
                docker=docker,
            )
        },
    }


def _docker(argv, timeout=120):
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RunnerError(f"`{' '.join(argv)}` failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _inspect(cid, template, docker="docker"):
    return _docker([docker, "inspect", "-f", template, cid])


def _container_pid(cid, docker="docker", deadline_s=15.0):
    """Host PID of the container's process, once the runtime has published it."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            value = int(_inspect(cid, "{{.State.Pid}}", docker=docker) or 0)
        except (RunnerError, ValueError):
            value = 0
        if value:
            return value
        time.sleep(0.02)
    return None


def classify(exit_code, oom_killed, log_text, timed_out, telemetry):
    """R-FAIL-03: the run status and the evidence it was classified from.

    Precedence, most specific first: an operator-visible timeout; a runtime
    OOM kill; a fatal signal; an out-of-memory message in stderr, split into
    device and host by the marker that matched; any other non-zero exit. A
    zero exit is `ok` here -- `no_output` and `eval_failed` are decided by the
    caller, which is the only place that knows whether a point cloud appeared.
    """
    haystack = (log_text or "").lower()
    evidence = {
        "exit_code": exit_code,
        "oom_killed": bool(oom_killed),
        "timed_out": bool(timed_out),
        "signal": (exit_code - 128) if isinstance(exit_code, int) and exit_code > 128 else None,
        "log_markers": [],
        "last_device_mem_bytes": (telemetry or {}).get("last_device_mem_bytes"),
        "device_mem_total_bytes": (telemetry or {}).get("device_mem_total_bytes"),
        "peak_host_rss_bytes": (telemetry or {}).get("peak_host_rss_bytes"),
    }
    if timed_out:
        return "timeout", evidence
    if oom_killed:
        evidence["log_markers"].append("docker OOMKilled")
        return "oom_host", evidence

    cuda_hits = [m for m in CUDA_OOM_MARKERS if m in haystack]
    host_hits = [m for m in HOST_OOM_MARKERS if m in haystack]
    generic_oom = "out of memory" in haystack
    segfault_hits = [m for m in SEGFAULT_MARKERS if m in haystack]
    evidence["log_markers"] = cuda_hits + host_hits + segfault_hits + (
        ["out of memory"] if generic_oom and not (cuda_hits or host_hits) else []
    )

    if exit_code == 139 or (segfault_hits and exit_code != 0):
        return "segfault", evidence
    if exit_code != 0:
        if cuda_hits or (generic_oom and not host_hits):
            return "oom_gpu", evidence
        if host_hits:
            return "oom_host", evidence
        return "nonzero_exit", evidence
    return "ok", evidence


def _write_harness_phase(path, name, begin_ns, end_ns, attrs=None):
    """Harness-emitted phases, in the same line format as `bench_timer.h`."""
    tail = "".join(f" {k}={v}" for k, v in (attrs or {}).items())
    with open(path, "a") as f:
        f.write(f"PHASE {name} BEGIN {begin_ns}{tail}\n")
        f.write(f"PHASE {name} END {end_ns}{tail}\n")


def _container_duration_s(started_at, finished_at):
    """Container start to exit, from the runtime's own timestamps.

    R-TIM-02 wants the measured process timed from a monotonic clock, which is
    what `wall_time_s` is; it includes the container's start-up. These two
    wall-clock timestamps bound the process itself, so their difference is the
    same interval without the start-up, and the pair is what D10 asks to be
    able to separate.
    """
    if not started_at or not finished_at:
        return None
    try:
        # Docker prints nanoseconds; datetime parses at most microseconds.
        def _parse(value):
            head, _, tail = value.partition(".")
            if tail:
                fraction, zone = tail[:-1], tail[-1]
                value = f"{head}.{fraction[:6]}{zone}"
            return datetime.fromisoformat(value.replace("Z", "+00:00"))

        delta = (_parse(finished_at) - _parse(started_at)).total_seconds()
    except ValueError:
        return None
    return delta if delta >= 0 else None


def _sha256_or_none(path):
    try:
        return ev.sha256_file(path)
    except OSError:
        return None


def execute(
    spec,
    entry,
    run,
    tmp_dir,
    docker="docker",
    sample_interval_s=smp.DEFAULT_INTERVAL_S,
    gpu_state_evidence=None,
):
    """Run one run to completion and return everything the record needs.

    Never raises for a failure of the measured process: a failure is a result
    (R-FAIL-01, R-FAIL-02).
    """
    tmp_dir = Path(tmp_dir)
    work_dir = spec.work_dir(run)
    result = {
        "started_at": utc_now(),
        "work_dir": str(work_dir),
        "notes": [],
        "warnings": [],
        "gpu_state_evidence": gpu_state_evidence or {},
    }
    # A scratch directory left by an interrupted attempt would let its point
    # cloud or its .dmb files be mistaken for this run's output, so it is moved
    # out of the way rather than reused or deleted.
    if work_dir.exists():
        result["notes"].append(f"stale scratch moved aside: {move_aside(work_dir, 'stale')}")
    (work_dir / "prepared").mkdir(parents=True)
    (work_dir / "out").mkdir(parents=True)
    harness_phase_file = tmp_dir / "phases.harness.txt"

    # R-IO-03: warm the page cache from the run's own inputs; caches are never
    # dropped between runs.
    warm_t0 = time.monotonic_ns()
    warmed = smp.warm_cache(spec.raw_scene_dir(run.scene, run.width))
    result["warm_cache_bytes"] = warmed
    _write_harness_phase(
        harness_phase_file, "preprocess.warm_cache", warm_t0, time.monotonic_ns(),
        {"bytes": warmed},
    )

    # preprocess.convert (R-TIM-03).
    convert_argv = preprocess_argv(spec, entry, run)
    convert_cmd = docker_run_argv(
        spec, run, convert_argv, work_dir, detach=False, docker=docker
    )
    convert_t0 = time.monotonic_ns()
    convert = subprocess.run(convert_cmd, capture_output=True, text=True)
    convert_t1 = time.monotonic_ns()
    (tmp_dir / "convert.stdout.log").write_text(convert.stdout)
    (tmp_dir / "convert.stderr.log").write_text(convert.stderr)
    _write_harness_phase(harness_phase_file, "preprocess.convert", convert_t0, convert_t1)
    result["preprocess"] = {
        "argv": convert_argv,
        "exit_code": convert.returncode,
        "duration_s": (convert_t1 - convert_t0) / 1e9,
    }
    if convert.returncode != 0:
        result.update(
            status="nonzero_exit",
            status_evidence={
                "stage": "preprocess.convert",
                "exit_code": convert.returncode,
                "stderr_tail": convert.stderr[-4000:],
            },
            wall_time_s=None,
            ended_at=utc_now(),
        )
        return result

    # The neighbour list the method will consume (R-EXP-07).
    prepared = work_dir / "prepared"
    neighbour_file = next(
        (p for p in (prepared / "pair.txt", prepared / "view_id_sets.json") if p.exists()),
        None,
    )
    result["neighbour_list"] = {
        "path": str(neighbour_file) if neighbour_file else None,
        "sha256": _sha256_or_none(neighbour_file) if neighbour_file else None,
    }

    # The measured process, in a fresh container (R-RUN-05).
    env = method_env(spec)
    method_argv = resolve_invocation(spec, entry, run)
    cname = f"bench_{run.key}"
    subprocess.run([docker, "rm", "-f", cname], capture_output=True, text=True)
    run_cmd = docker_run_argv(
        spec, run, method_argv, work_dir, detach=True, name=cname, docker=docker, env=env
    )
    result["command"] = run_cmd

    sampler = smp.Sampler(gpu_index=spec.gpu_index, interval_s=sample_interval_s)
    stdout_log = open(tmp_dir / "stdout.log", "wb")
    stderr_log = open(tmp_dir / "stderr.log", "wb")
    timed_out = False
    log_proc = None
    cid = None
    t0 = time.monotonic_ns()
    try:
        # Sampling starts before the container so that the device-memory
        # baseline is taken with the GPU idle.
        sampler.start()
        cid = _docker(run_cmd)
        sampler.set_root_pid(_container_pid(cid, docker=docker))
        sampler.set_cgroup(smp.cgroup_for_container(cid))
        # R-OBS-02: logs stream to disk while the run is in flight.
        log_proc = subprocess.Popen(
            [docker, "logs", "-f", cid], stdout=stdout_log, stderr=stderr_log
        )
        try:
            exit_code = int(
                subprocess.run(
                    [docker, "wait", cid],
                    capture_output=True,
                    text=True,
                    timeout=spec.timeout_s,
                ).stdout.strip()
                or -1
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            # R-RUN-07: SIGKILL, keep everything sampled so far.
            subprocess.run([docker, "kill", "--signal=KILL", cid], capture_output=True)
            try:
                exit_code = int(
                    subprocess.run(
                        [docker, "wait", cid], capture_output=True, text=True, timeout=120
                    ).stdout.strip()
                    or -1
                )
            except subprocess.SubprocessError:
                exit_code = -1
    except BaseException:
        # An operator's second signal, or a harness fault, must not leave a
        # detached container holding the benchmark GPU: whatever runs next
        # would be measured against it (R-RUN-05, R-ENV-02).
        if cid:
            subprocess.run([docker, "rm", "-f", cid], capture_output=True)
        raise
    finally:
        t1 = time.monotonic_ns()
        sampler.stop()
        if sampler.ident is not None:
            sampler.join(timeout=5)
        if log_proc is not None:
            try:
                log_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log_proc.kill()
        stdout_log.close()
        stderr_log.close()

    inspected = {}
    if cid:
        for field, template in (
            ("oom_killed", "{{.State.OOMKilled}}"),
            ("started_at", "{{.State.StartedAt}}"),
            ("finished_at", "{{.State.FinishedAt}}"),
            ("exit_code", "{{.State.ExitCode}}"),
        ):
            try:
                inspected[field] = _inspect(cid, template, docker=docker)
            except RunnerError:
                inspected[field] = None
        # R-RUN-05: the container is removed after the run.
        subprocess.run([docker, "rm", "-f", cid], capture_output=True)

    telemetry = sampler.summary()
    sampler.write_parquet(tmp_dir / "telemetry.parquet")

    # R-ENV-02 after the fact: the pre-run gate proved the clock was locked
    # before the run; the telemetry says whether it stayed there during it. A
    # run that lost the lock is recorded with `held: false`, never retried.
    clocks = (gpu_state_evidence or {}).get("clocks", {})
    hold = fpr.clock_hold(
        sampler.column("gpu_sm_clock_mhz"),
        fpr.expected_clock_mhz(gpu_state_evidence),
        method=clocks.get("locked_clocks_method"),
    )
    result["clock_hold"] = hold
    if hold["held"] is False:
        result.setdefault("warnings", []).append(
            f"the SM clock did not hold {hold['expected_mhz']} MHz during this run "
            f"({hold['fraction_at_expected']:.3f} of {hold['samples']} samples within "
            f"1%, below {fpr.CLOCK_HELD_MIN_FRACTION}; observed {hold['min_mhz']}-"
            f"{hold['max_mhz']} MHz); the run is recorded, and comparisons should "
            "exclude it (R-ENV-02, D24)"
        )
    elif hold["held"] is None:
        result.setdefault("warnings", []).append(
            "no expected SM clock was established before this run, so the clock "
            f"hold is unverified (observed {hold['min_mhz']}-{hold['max_mhz']} MHz)"
        )

    device_phase_file = work_dir / "out" / "phases.txt"
    if device_phase_file.exists():
        shutil.copy2(device_phase_file, tmp_dir / "phases.txt")
    trace = ph.parse_file(tmp_dir / "phases.txt")
    harness_trace = ph.parse_file(harness_phase_file)
    if not spec.phase_timer:
        # R-TIM-09's uninstrumented arm is given no MVS_BENCH_PHASES, so the
        # missing trace is the arm working, not a defect to record as one.
        trace.errors = [e for e in trace.errors if "no phase trace was written" not in e]

    # Several forks report a CUDA failure on stdout, so both streams are
    # scanned for the failure markers.
    log_text = (tmp_dir / "stderr.log").read_text(errors="replace") + "\n" + (
        tmp_dir / "stdout.log"
    ).read_text(errors="replace")
    status, evidence = classify(
        exit_code, str(inspected.get("oom_killed")).lower() == "true", log_text,
        timed_out, telemetry,
    )
    evidence["container"] = inspected

    container_wall_s = _container_duration_s(
        inspected.get("started_at"), inspected.get("finished_at")
    )
    result.update(
        status=status,
        status_evidence=evidence,
        exit_code=exit_code,
        wall_time_s=(t1 - t0) / 1e9,
        container_wall_time_s=container_wall_s,
        container_startup_s=(
            None if container_wall_s is None else (t1 - t0) / 1e9 - container_wall_s
        ),
        container_started_at=inspected.get("started_at"),
        container_finished_at=inspected.get("finished_at"),
        telemetry=telemetry,
        phases=[dict(row, source="method") for row in ph.totals(trace)]
        + [dict(row, source="harness") for row in ph.totals(harness_trace)],
        phases_by_pass=ph.totals_by_pass(trace),
        phase_trace_errors=trace.errors[:50],
        ended_at=utc_now(),
    )
    result["wall_time_with_preprocess_s"] = (
        result["wall_time_s"] + result["preprocess"]["duration_s"]
    )
    return result


def collect_point_cloud(spec, entry, run, result, tmp_dir):
    """Copy the point cloud out of the scratch tree (R-ART-03) and hash it.

    Copied rather than moved: the measured container runs as root, so the
    directory it wrote the cloud into is root-owned and the harness -- which
    runs as the operator -- may read from it but not unlink out of it.
    """
    work_dir = Path(result["work_dir"])
    produced = work_dir / "prepared" / entry["output_ply"]
    if not produced.exists():
        result["point_cloud"] = {"path": None, "size_bytes": None, "sha256": None}
        if result.get("status") == "ok":
            result["status"] = "no_output"
            result.setdefault("status_evidence", {})["missing_output"] = str(produced)
        return result

    target = Path(spec.artifact_path(run))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        result.setdefault("notes", []).append(
            f"previous point cloud moved aside: {move_aside(target, 'superseded')}"
        )
    shutil.copy2(str(produced), str(target))
    point_count = ev.ply_point_count(target)
    result["point_cloud"] = {
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": ev.sha256_file(target),
        "point_count": point_count,
    }
    # A method that fails on a half-written dataset can still exit 0 and write
    # a PLY header with no vertices in it (HPM-MVS does exactly this when its
    # converter died first). An empty cloud is not an output, and a run that
    # produced one is not `ok`.
    if not point_count and result.get("status") == "ok":
        result["status"] = "no_output"
        result.setdefault("status_evidence", {})["empty_output"] = {
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "point_count": point_count,
        }
    return result


def run_evaluation(spec, run, result, tmp_dir, docker="docker"):
    """R-QUA-01/04: scores, or `eval_failed` / `no_output` with the evidence.

    A cloud produced by a run that already failed is still evaluated -- the
    partial output of a timeout is evidence -- but the failure keeps the run's
    status: `eval_failed` only ever replaces `ok`.
    """
    cloud = result.get("point_cloud") or {}
    if not cloud.get("path"):
        result["quality"] = None
        return result
    try:
        result["quality"] = ev.evaluate(
            spec,
            cloud["path"],
            run.scene,
            run.width,
            log_dir=tmp_dir,
            docker=docker,
            timeout=int(spec.data.get("eval_timeout_seconds", 3600)),
        )
    except ev.EvaluationFailed as exc:
        result["quality"] = None
        if result.get("status") == "ok":
            result["status"] = "eval_failed"
        result.setdefault("status_evidence", {})["evaluation"] = {
            "message": str(exc),
            "stderr_tail": exc.stderr[-4000:],
            "argv": exc.argv,
        }
    return result


def reclaim_ownership(spec, work_dir, docker="docker"):
    """Chown the scratch tree back to the operator, from inside a container.

    Everything the measured process created belongs to root, because that is
    what a container is by default and the harness does not change what the
    method runs as. Handing the tree back with a throwaway root container is
    cheaper than running the harness itself with the privileges to remove it.
    """
    return subprocess.run(
        [
            docker, "run", "--rm",
            "-v", f"{work_dir}:/scratch",
            "--entrypoint", "/bin/chown",
            spec.image,
            "-R", f"{os.getuid()}:{os.getgid()}", "/scratch",
        ],
        capture_output=True,
        text=True,
    )


def discard_intermediates(spec, result, docker="docker"):
    """R-ART-02: drop the depth, normal and cost maps after evaluation.

    The only deletion the harness performs, and it is confined to the scratch
    working directory the harness itself created for this run, outside the
    result store. `keep_intermediates: true` in the specification turns it off.
    """
    work_dir = Path(result["work_dir"]).resolve()
    root = Path(spec.paths["work_root"]).resolve()
    # Measured before the tree goes: it is the run's whole on-disk footprint,
    # and under `debug_output: upstream` the difference against the same run
    # with the flag is what the debug artefacts cost in bytes (R-EXP-11).
    size = sum(p.stat().st_size for p in work_dir.rglob("*") if p.is_file())
    result["intermediates_bytes"] = size
    if spec.keep_intermediates:
        result.setdefault("notes", []).append(f"intermediates kept in {work_dir}")
        return result
    if root not in work_dir.parents:
        raise RunnerError(f"refusing to remove {work_dir}: it is not inside {root}")
    try:
        shutil.rmtree(work_dir)
    except PermissionError:
        reclaim = reclaim_ownership(spec, work_dir, docker=docker)
        if reclaim.returncode != 0:
            raise
        shutil.rmtree(work_dir)
        result.setdefault("notes", []).append(
            "the measured container's root-owned files were chowned back before removal"
        )
    result.setdefault("notes", []).append(
        f"discarded {size} bytes of intermediates in {work_dir} (R-ART-02)"
    )
    return result
