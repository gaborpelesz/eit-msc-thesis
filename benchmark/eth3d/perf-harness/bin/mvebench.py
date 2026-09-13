#!/usr/bin/env python3
"""Self-verifying performance harness for ETH3DMultiViewEvaluation.

Three subcommands:

  golden   run a build and record the reference outputs (stdout result lines +
           SHA-256 of every per-point visualization cloud) for a dataset
  gate     re-run a build and assert byte-identical results against the golden
  bench    interleaved, repeated timing of one or more builds, with median/p95,
           a bootstrap CI on the speedup ratio, and the isolation guards below

ISOLATION (the point of the exercise)
-------------------------------------
Timing a code path that is itself parallel, inside a pipeline that is also
parallel, produces numbers that mean nothing. This harness therefore:

  * pins the OpenMP thread count explicitly and scrubs every other thread-count
    environment variable, so nothing is inherited from the caller's pipeline;
  * measures exactly one process at a time and holds an exclusive lock file, so
    two harness invocations cannot overlap;
  * refuses to start on battery power, under load, or while another copy of the
    binary under test is running;
  * measures child CPU time alongside wall time for every single run and
    ASSERTS the resulting parallelism ratio against the regime being measured.
    A `threads=N` run whose cpu/wall is ~1.0 was accidentally serialized and is
    reported as a hard failure, not silently averaged in. A `threads=1` run whose
    cpu/wall is well above 1.0 means something is still parallel underneath and
    the single-thread regime is not isolated.
  * interleaves variants round-robin in a seeded-random order within each round,
    so thermal drift hits every variant equally instead of penalising whichever
    one ran last.

Per-phase wall times come from timestamping the program's own progress lines on
stdout. That works on an unmodified binary, so every variant is measured the
same way with no instrumentation deviation.
"""

import argparse, hashlib, json, os, random, re, resource, shutil, statistics, subprocess
import sys, time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent.parent
LOCK_PATH = HARNESS_DIR / ".mvebench.lock"
EXE = "ETH3DMultiViewEvaluation"
TOLERANCES = "0.01,0.02,0.05,0.1,0.2,0.5"
RESULT_LINE_RE = re.compile(r"^(Tolerances|Completenesses|Accuracies|F1-scores):")
PHASE_MARKERS = [
    ("load", re.compile(r"^Loading reconstruction:")),
    ("completeness", re.compile(r"^Computing completeness$")),
    ("accuracy", re.compile(r"^Computing accuracy$")),
    ("report", re.compile(r"^Tolerances:")),
]

# Every thread-count knob we know of, so none of them leaks in from the caller.
THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "OMP_NESTED",
    "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_MAX_ACTIVE_LEVELS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "TBB_NUM_THREADS",
    "KMP_AFFINITY", "KMP_BLOCKTIME",
]


# --------------------------------------------------------------------- guards

class IsolationError(RuntimeError):
    pass


class ExclusiveLock:
    def __init__(self, path):
        self.path, self.fd = Path(path), None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise IsolationError(
                f"another mvebench run holds {self.path}; measurements must not overlap. "
                f"Remove the file if it is stale."
            )
        os.write(self.fd, f"{os.getpid()} {time.time()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def loadavg_1m():
    return os.getloadavg()[0]


def on_ac_power():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                             timeout=5).stdout
        return "AC Power" in out
    except Exception:
        return True          # not macOS, or pmset missing: do not block


def other_instances():
    # -x matches the executable NAME only. `pgrep -f` would match this harness's
    # own argv (it carries the binary path in --variant) and report a phantom.
    try:
        out = subprocess.run(["pgrep", "-x", EXE], capture_output=True, text=True).stdout
        return [p for p in out.split() if p and int(p) != os.getpid()]
    except Exception:
        return []


def cpu_idle_pct():
    """Instantaneous system-wide idle %. The right signal for 'is the machine quiet'.

    The 1-minute load average is not: it decays over ~60 s, so it is still
    reporting the harness's OWN previous invocation minutes after that run
    finished, and would block back-to-back measurements for no reason.
    """
    try:
        out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"],
                             capture_output=True, text=True, timeout=20).stdout
        vals = re.findall(r"CPU usage:.*?([\d.]+)% idle", out)
        if vals:
            return float(vals[-1])          # second sample: the first is since-boot
    except Exception:
        pass
    return 100.0


def wait_until_quiet(min_idle, timeout_s):
    """Let the machine settle instead of failing on our own decaying tail."""
    t0 = time.perf_counter()
    idle = cpu_idle_pct()
    while idle < min_idle and (time.perf_counter() - t0) < timeout_s:
        print(f"  waiting for the machine to settle: idle {idle:.0f}% "
              f"< {min_idle:.0f}%", file=sys.stderr)
        time.sleep(5)
        idle = cpu_idle_pct()
    return idle


def preflight(min_idle, settle_timeout, allow_dirty):
    problems = []
    if not on_ac_power():
        problems.append("running on battery: macOS throttles and timings are not comparable")
    inst = other_instances()
    if inst:
        problems.append(f"another {EXE} is already running (pids {','.join(inst)})")
    idle = wait_until_quiet(min_idle, settle_timeout)
    if idle < min_idle:
        problems.append(f"machine still busy after {settle_timeout}s: "
                        f"{idle:.0f}% idle (need {min_idle:.0f}%)")
    if problems and not allow_dirty:
        raise IsolationError("preflight failed:\n  - " + "\n  - ".join(problems)
                             + "\n(pass --allow-dirty to override, and say so in the report)")
    return {"cpu_idle_pct": idle, "loadavg_1m": loadavg_1m(),
            "ac_power": on_ac_power(), "warnings": problems}


def clean_env(threads, wait_policy="PASSIVE"):
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV_KEYS}
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"            # no runtime thread-count surprises
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"      # no nested parallelism
    # PASSIVE, not the libomp default: with spin-waiting, threads blocked on the
    # `omp critical` in ComputeCompleteness burn CPU doing nothing, and cpu/wall
    # reads ~6 even when the region is fully serialized -- which would defeat the
    # isolation guard below. PASSIVE makes cpu/wall mean actual concurrent work.
    env["OMP_WAIT_POLICY"] = wait_policy
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    return env


# ------------------------------------------------------------------ one run

MAXRSS_UNIT_MB = (1.0 / (1024 * 1024)) if sys.platform == "darwin" else (1.0 / 1024)


def _spawn_and_capture(cmd, env, timeout):
    """posix_spawn + wait4: exact user/sys CPU and peak RSS for THIS child only.

    Deliberately not subprocess + a polling sampler: polling `ps` would fork
    hundreds of short-lived children per run, and their CPU would land in
    RUSAGE_CHILDREN and silently inflate every cpu/wall figure the isolation
    guard depends on. wait4() gives the measured process's own rusage and
    perturbs nothing.
    """
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    actions = [(os.POSIX_SPAWN_CLOSE, out_r), (os.POSIX_SPAWN_CLOSE, err_r),
               (os.POSIX_SPAWN_DUP2, out_w, 1), (os.POSIX_SPAWN_DUP2, err_w, 2),
               (os.POSIX_SPAWN_CLOSE, out_w), (os.POSIX_SPAWN_CLOSE, err_w)]
    t0 = time.perf_counter()
    pid = os.posix_spawn(cmd[0], cmd, env, file_actions=actions)
    os.close(out_w); os.close(err_w)

    import select
    stamps, err_buf, pending = [], bytearray(), bytearray()
    fds = {out_r, err_r}
    deadline = t0 + timeout
    while fds:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            os.kill(pid, 9)
            break
        ready, _, _ = select.select(list(fds), [], [], min(remaining, 1.0))
        for fd in ready:
            chunk = os.read(fd, 65536)
            if not chunk:
                fds.discard(fd)
                os.close(fd)
                continue
            now = time.perf_counter() - t0
            if fd == err_r:
                err_buf += chunk
            else:
                pending += chunk
                while b"\n" in pending:
                    line, _, pending = pending.partition(b"\n")
                    stamps.append((now, line.decode("utf-8", "replace")))
    if pending:
        stamps.append((time.perf_counter() - t0, pending.decode("utf-8", "replace")))
    for fd in list(fds):
        os.close(fd)
    _, status, ru = os.wait4(pid, 0)
    wall = time.perf_counter() - t0
    rc = os.waitstatus_to_exitcode(status)
    return {
        "wall_s": wall,
        "cpu_s": ru.ru_utime + ru.ru_stime,
        "user_s": ru.ru_utime, "sys_s": ru.ru_stime,
        "peak_rss_mb": ru.ru_maxrss * MAXRSS_UNIT_MB,
        "returncode": rc,
        "stamps": stamps,
        "stderr": err_buf.decode("utf-8", "replace").strip(),
    }


def run_once(binary, dataset, threads, *, accuracy_out=None, completeness_out=None,
             timeout=3600, wait_policy="PASSIVE"):
    """One measured invocation. Returns a dict; never raises on a nonzero exit."""
    ds = Path(dataset)
    cmd = [str(Path(binary).resolve()),
           "--tolerances", TOLERANCES,
           "--reconstruction_ply_path", str(ds / "reconstruction.ply"),
           "--ground_truth_mlp_path", str(ds / "scan_alignment.mlp")]
    if accuracy_out:
        cmd += ["--accuracy_cloud_output_path", str(accuracy_out)]
    if completeness_out:
        cmd += ["--completeness_cloud_output_path", str(completeness_out)]

    env = clean_env(threads, wait_policy)
    la_before = loadavg_1m()
    raw = _spawn_and_capture(cmd, env, timeout)

    out_lines = [l for _, l in raw["stamps"]]
    marks = {}
    for t, line in raw["stamps"]:
        for name, rx in PHASE_MARKERS:
            if name not in marks and rx.match(line):
                marks[name] = t
    phases, order = {}, [n for n, _ in PHASE_MARKERS]
    for i, name in enumerate(order):
        if name not in marks:
            continue
        nxt = order[i + 1] if i + 1 < len(order) else None
        end = marks[nxt] if (nxt and nxt in marks) else raw["wall_s"]
        phases[name] = end - marks[name]

    return {
        "returncode": raw["returncode"],
        "wall_s": raw["wall_s"],
        "cpu_s": raw["cpu_s"], "user_s": raw["user_s"], "sys_s": raw["sys_s"],
        "parallelism": (raw["cpu_s"] / raw["wall_s"]) if raw["wall_s"] > 0 else 0.0,
        "peak_rss_mb": raw["peak_rss_mb"],
        "phases_s": phases,
        "result_lines": [l for l in out_lines if RESULT_LINE_RE.match(l)],
        "stderr": raw["stderr"],
        "loadavg_before": la_before,
        "loadavg_after": loadavg_1m(),
        "threads": threads,
    }


def assert_isolation(run, threads, min_parallelism, max_serial_parallelism):
    """The guard against the exact failure mode of measuring a serialized run."""
    p = run["parallelism"]
    if run["returncode"] != 0:
        raise IsolationError(f"binary exited {run['returncode']}: {run['stderr'][:400]}")
    if threads == 1:
        if p > max_serial_parallelism:
            raise IsolationError(
                f"threads=1 run had cpu/wall={p:.2f} (>{max_serial_parallelism}): the "
                f"single-thread regime is not isolated -- something below the binary is "
                f"still using multiple cores.")
    else:
        if p < min_parallelism:
            raise IsolationError(
                f"threads={threads} run had cpu/wall={p:.2f} (<{min_parallelism}): this "
                f"run was effectively SERIALIZED. Timing it as a parallel run would be "
                f"misleading -- check OMP_NUM_THREADS leakage, an outer pipeline lock, "
                f"or oversubscription.")
        if p > threads + 0.5:
            raise IsolationError(
                f"threads={threads} run had cpu/wall={p:.2f}, above the {threads} threads "
                f"requested: something outside the binary under test is also running, or "
                f"an outer pipeline leaked its own parallelism in.")


# ---------------------------------------------------------------- statistics

def p95(xs):
    """Nearest-rank 95th percentile: for small n this is the honest choice."""
    s = sorted(xs)
    k = max(1, int(-(-95 * len(s) // 100)))
    return s[k - 1]


def rel_mad(xs):
    med = statistics.median(xs)
    if med == 0:
        return 0.0
    return statistics.median(abs(x - med) for x in xs) / med


def summarize(samples):
    w = [s["wall_s"] for s in samples]
    out = {
        "n": len(w), "median_s": statistics.median(w), "p95_s": p95(w),
        "min_s": min(w), "max_s": max(w),
        "mean_s": statistics.fmean(w),
        "stdev_s": statistics.stdev(w) if len(w) > 1 else 0.0,
        "rel_mad": rel_mad(w),
        "spread_pct": 100.0 * (max(w) - min(w)) / statistics.median(w),
        "median_cpu_s": statistics.median([s["cpu_s"] for s in samples]),
        "median_parallelism": statistics.median([s["parallelism"] for s in samples]),
        "median_peak_rss_mb": statistics.median([s["peak_rss_mb"] for s in samples]),
        "samples_s": w,
    }
    for ph in ("load", "completeness", "accuracy", "report"):
        vals = [s["phases_s"].get(ph) for s in samples if ph in s["phases_s"]]
        if vals:
            out[f"median_{ph}_s"] = statistics.median(vals)
            out[f"p95_{ph}_s"] = p95(vals)
    return out


def bootstrap_speedup_ci(base, var, iters=20000, seed=12345, alpha=0.05):
    """95% CI for median(base)/median(var), resampling each arm independently."""
    rng = random.Random(seed)
    nb, nv = len(base), len(var)
    ratios = []
    for _ in range(iters):
        b = statistics.median([base[rng.randrange(nb)] for _ in range(nb)])
        v = statistics.median([var[rng.randrange(nv)] for _ in range(nv)])
        ratios.append(b / v if v > 0 else float("inf"))
    ratios.sort()
    lo = ratios[int(alpha / 2 * iters)]
    hi = ratios[int((1 - alpha / 2) * iters)]
    return lo, hi


def mann_whitney_u_p(a, b):
    """Two-sided p via the normal approximation with tie correction. n>=8 per arm."""
    n1, n2 = len(a), len(b)
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = [0.0] * len(comb), 0
    tie_terms = 0.0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_terms += t ** 3 - t
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, comb) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_terms / (n * (n - 1)))
    if sigma2 <= 0:
        return 1.0
    z = (u - mu + 0.5) / (sigma2 ** 0.5)
    # two-sided normal tail
    import math
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2))))


# ------------------------------------------------------------------- golden

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_outputs(binary, dataset, workdir, threads):
    """Run with per-point visualization enabled; return the full result fingerprint."""
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    run = run_once(binary, dataset, threads,
                   accuracy_out=workdir / "acc", completeness_out=workdir / "cmp")
    if run["returncode"] != 0:
        raise RuntimeError(f"binary exited {run['returncode']}: {run['stderr'][:500]}")
    clouds = {p.name: sha256(p) for p in sorted(workdir.rglob("*.ply"))}
    if not clouds:
        raise RuntimeError("no visualization clouds were written; the gate would be vacuous")
    return {"result_lines": run["result_lines"], "cloud_sha256": clouds,
            "n_clouds": len(clouds)}


def cmd_golden(args):
    fp = capture_outputs(args.binary, args.dataset, args.workdir, args.threads)
    fp["dataset"] = str(Path(args.dataset).resolve())
    fp["binary"] = str(Path(args.binary).resolve())
    fp["tolerances"] = TOLERANCES
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fp, indent=2, sort_keys=True))
    print(f"golden written: {args.out}  ({fp['n_clouds']} clouds)")
    for l in fp["result_lines"]:
        print("   ", l)


def cmd_gate(args):
    golden = json.loads(Path(args.golden).read_text())
    fp = capture_outputs(args.binary, args.dataset, args.workdir, args.threads)
    diffs = []
    if fp["result_lines"] != golden["result_lines"]:
        diffs.append("stdout result lines differ:")
        for g, n in zip(golden["result_lines"], fp["result_lines"]):
            if g != n:
                diffs.append(f"    golden: {g}\n    actual: {n}")
        if len(golden["result_lines"]) != len(fp["result_lines"]):
            diffs.append(f"    line count {len(golden['result_lines'])} -> {len(fp['result_lines'])}")
    gk, nk = set(golden["cloud_sha256"]), set(fp["cloud_sha256"])
    if gk != nk:
        diffs.append(f"cloud set differs: missing={sorted(gk - nk)} extra={sorted(nk - gk)}")
    for k in sorted(gk & nk):
        if golden["cloud_sha256"][k] != fp["cloud_sha256"][k]:
            diffs.append(f"cloud {k}: sha256 {golden['cloud_sha256'][k][:16]}"
                         f" -> {fp['cloud_sha256'][k][:16]}")
    ok = not diffs
    print(("PASS" if ok else "FAIL") + f"  byte-exactness gate  [{Path(args.dataset).name}]"
          f"  ({fp['n_clouds']} per-point clouds + 4 result lines)")
    for d in diffs:
        print("  " + d)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"pass": ok, "diffs": diffs, "actual": fp}, indent=2, sort_keys=True))
    return 0 if ok else 1


# -------------------------------------------------------------------- bench

def cmd_bench(args):
    variants = []
    for spec in args.variant:
        name, _, path = spec.partition("=")
        variants.append((name, Path(path or name)))
    for name, p in variants:
        if not p.exists():
            raise SystemExit(f"binary not found for variant {name}: {p}")
    thread_counts = [int(t) for t in str(args.threads).split(",") if t.strip()]

    with ExclusiveLock(LOCK_PATH):
        pre = preflight(args.min_idle, args.settle_timeout, args.allow_dirty)
        report = {
            "dataset": str(Path(args.dataset).resolve()),
            "thread_counts": thread_counts, "warmup": args.warmup,
            "repeats": args.repeats, "seed": args.seed, "preflight": pre,
            "wait_policy": args.wait_policy, "host": host_info(),
            "baseline": args.baseline, "by_threads": {},
        }
        all_failures = []

        for threads in thread_counts:
            rng = random.Random(args.seed + 1000 * threads)
            samples = {n: [] for n, _ in variants}
            failures = []
            print(f"\n== threads={threads} ==", file=sys.stderr)
            for rnd in range(args.warmup + args.repeats):
                order = list(variants)
                rng.shuffle(order)              # defeat thermal / ordering bias
                tag = "warmup" if rnd < args.warmup else f"rep {rnd - args.warmup + 1}"
                for name, path in order:
                    r = run_once(path, args.dataset, threads,
                                 wait_policy=args.wait_policy)
                    try:
                        assert_isolation(r, threads, args.min_parallelism,
                                         args.max_serial_parallelism)
                    except IsolationError as e:
                        failures.append({"round": rnd, "threads": threads,
                                         "variant": name, "error": str(e)})
                        print(f"  [{tag}] {name:<22} ISOLATION FAILURE: {e}",
                              file=sys.stderr)
                        continue
                    if rnd >= args.warmup:
                        samples[name].append(r)
                    print(f"  [{tag}] {name:<22} {r['wall_s']:7.3f}s  "
                          f"cpu/wall={r['parallelism']:5.2f}  "
                          f"rss={r['peak_rss_mb']:6.0f}MB  "
                          + "  ".join(f"{k}={v:.3f}" for k, v in r["phases_s"].items()),
                          file=sys.stderr)
            all_failures += failures

            ref, mismatch = None, []
            for name, _ in variants:
                if not samples[name]:
                    continue
                lines = samples[name][0]["result_lines"]
                if ref is None:
                    ref = (name, lines)
                elif lines != ref[1]:
                    mismatch.append(f"{name} vs {ref[0]}")

            block = {"variants": {n: summarize(samples[n]) for n, _ in variants
                                  if samples[n]},
                     "metric_mismatch": mismatch,
                     "result_lines": ref[1] if ref else [],
                     "isolation_failures": failures}
            base = args.baseline
            if base in block["variants"]:
                bw = block["variants"][base]["samples_s"]
                for n, st in block["variants"].items():
                    if n == base:
                        continue
                    lo, hi = bootstrap_speedup_ci(bw, st["samples_s"], seed=args.seed)
                    st["speedup_median"] = block["variants"][base]["median_s"] / st["median_s"]
                    st["speedup_ci95"] = [lo, hi]
                    st["mannwhitney_p"] = mann_whitney_u_p(bw, st["samples_s"])
                    st["significant"] = ((lo > 1.0 or hi < 1.0)
                                         and st["mannwhitney_p"] < 0.05)
            report["by_threads"][str(threads)] = block

        report["isolation_failures"] = all_failures
        report["thread_scaling"] = compute_scaling(report)

        if all_failures and not args.allow_dirty:
            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(f"\n{len(all_failures)} run(s) failed the isolation guard; "
                             f"refusing to report timings.")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
        print_report(report)
    return 0


def compute_scaling(report):
    """Per-phase speedup from the lowest thread count to each higher one.

    This is the guard that cpu/wall cannot provide. A thread pool can be fully
    occupied -- cpu/wall pinned at the thread count -- while every thread spins
    on the same lock and the phase gets SLOWER. Anything below 1.0 here means
    adding threads costs time, i.e. the code path serializes itself.
    """
    tcs = sorted(int(t) for t in report["by_threads"])
    if len(tcs) < 2:
        return {}
    base_t = str(tcs[0])
    out = {}
    for t in tcs[1:]:
        blk, ref = report["by_threads"][str(t)], report["by_threads"][base_t]
        per_variant = {}
        for name, st in blk["variants"].items():
            if name not in ref["variants"]:
                continue
            r = ref["variants"][name]
            row = {"total": r["median_s"] / st["median_s"]}
            for ph in ("load", "completeness", "accuracy"):
                a, b = r.get(f"median_{ph}_s"), st.get(f"median_{ph}_s")
                if a and b:
                    row[ph] = a / b
            row["regressions"] = sorted(k for k, v in row.items()
                                        if k != "regressions" and v < 1.0)
            per_variant[name] = row
        out[f"{base_t}->{t}"] = per_variant
    return out


def host_info():
    def sc(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return ""
    return {"cpu": sc("machdep.cpu.brand_string"), "ncpu": sc("hw.ncpu"),
            "perf_cores": sc("hw.perflevel0.logicalcpu"),
            "eff_cores": sc("hw.perflevel1.logicalcpu"),
            "os": subprocess.run(["uname", "-sr"], capture_output=True,
                                 text=True).stdout.strip()}


def print_report(rep):
    base = rep["baseline"]
    print()
    print(f"dataset   {rep['dataset']}")
    print(f"host      {rep['host']['cpu']}  {rep['host']['ncpu']} cpu "
          f"({rep['host']['perf_cores']}P+{rep['host']['eff_cores']}E)  {rep['host']['os']}")
    print(f"protocol  {rep['repeats']} repeats (+{rep['warmup']} warmup), variants "
          f"interleaved round-robin in seeded-random order (seed {rep['seed']}), "
          f"OMP_WAIT_POLICY={rep['wait_policy']}")
    print(f"preflight {rep['preflight']['cpu_idle_pct']:.0f}% CPU idle, "
          f"loadavg {rep['preflight']['loadavg_1m']:.2f}, "
          f"AC power {rep['preflight']['ac_power']}"
          + (f", WARNINGS {rep['preflight']['warnings']}"
             if rep['preflight']['warnings'] else ""))

    for t in sorted(rep["by_threads"], key=int):
        blk = rep["by_threads"][t]
        if not blk["variants"]:
            continue
        print(f"\n--- threads = {t} " + "-" * 58)
        if blk["metric_mismatch"]:
            print(f"!! METRIC MISMATCH between variants: {blk['metric_mismatch']}")
        hdr = (f"{'variant':<22}{'median':>9}{'p95':>9}{'min':>9}{'max':>9}"
               f"{'relMAD':>8}{'cpu/wl':>8}{'RSS MB':>8}{'speedup':>9}"
               f"{'95% CI':>16}{'sig':>5}")
        print(hdr); print("-" * len(hdr))
        for n, st in sorted(blk["variants"].items(), key=lambda kv: kv[1]["median_s"]):
            sp, ci = st.get("speedup_median"), st.get("speedup_ci95")
            print(f"{n:<22}{st['median_s']:9.3f}{st['p95_s']:9.3f}{st['min_s']:9.3f}"
                  f"{st['max_s']:9.3f}{100*st['rel_mad']:7.2f}%"
                  f"{st['median_parallelism']:8.2f}{st['median_peak_rss_mb']:8.0f}"
                  + (f"{sp:8.2f}x" if sp else f"{'-':>9}")
                  + (f"  [{ci[0]:4.2f},{ci[1]:5.2f}]" if ci else f"{'-':>16}")
                  + (f"{'yes' if st.get('significant') else 'NO':>5}" if sp
                     else f"{'-':>5}"))
        print(f"\n{'  phase medians (s)':<22}" + "".join(f"{p:>14}" for p in
              ("load", "completeness", "accuracy")))
        for n, st in sorted(blk["variants"].items(), key=lambda kv: kv[1]["median_s"]):
            print(f"  {n:<20}" + "".join(
                f"{st.get(f'median_{p}_s', float('nan')):14.3f}"
                for p in ("load", "completeness", "accuracy")))

    if rep.get("thread_scaling"):
        print(f"\n--- thread scaling (>1.0 = faster with more threads) " + "-" * 24)
        for key, per in rep["thread_scaling"].items():
            print(f"  {key} threads")
            print(f"    {'variant':<20}{'total':>10}{'load':>10}"
                  f"{'completeness':>14}{'accuracy':>10}   note")
            for n, row in sorted(per.items()):
                note = ("SELF-SERIALIZING: " + ",".join(row["regressions"])
                        if row["regressions"] else "")
                print(f"    {n:<20}{row['total']:10.2f}{row.get('load', float('nan')):10.2f}"
                      f"{row.get('completeness', float('nan')):14.2f}"
                      f"{row.get('accuracy', float('nan')):10.2f}   {note}")

    for t in sorted(rep["by_threads"], key=int):
        lines = rep["by_threads"][t]["result_lines"]
        if lines:
            print()
            for l in lines:
                print("   ", l)
            break


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("golden")
    g.add_argument("--binary", required=True)
    g.add_argument("--dataset", required=True)
    g.add_argument("--workdir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--threads", type=int, default=1)
    g.set_defaults(fn=cmd_golden)

    t = sub.add_parser("gate")
    t.add_argument("--binary", required=True)
    t.add_argument("--dataset", required=True)
    t.add_argument("--workdir", required=True)
    t.add_argument("--golden", required=True)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--json")
    t.set_defaults(fn=cmd_gate)

    b = sub.add_parser("bench")
    b.add_argument("--variant", action="append", required=True,
                   metavar="NAME=PATH", help="repeatable; NAME=path/to/binary")
    b.add_argument("--dataset", required=True)
    b.add_argument("--threads", default="1",
                   help="comma-separated thread counts to measure, e.g. 1,12")
    b.add_argument("--repeats", type=int, default=11)
    b.add_argument("--warmup", type=int, default=2)
    b.add_argument("--seed", type=int, default=20260913)
    b.add_argument("--baseline", default="baseline")
    b.add_argument("--min-idle", type=float, default=70.0,
                   help="require this much system-wide CPU idle before measuring")
    b.add_argument("--settle-timeout", type=float, default=240.0)
    b.add_argument("--min-parallelism", type=float, default=1.5,
                   help="threads>1: cpu/wall below this means the run was serialized")
    b.add_argument("--max-serial-parallelism", type=float, default=1.35,
                   help="threads=1: cpu/wall above this means something else is parallel")
    b.add_argument("--wait-policy", default="PASSIVE", choices=["PASSIVE", "ACTIVE"],
                   help="OMP_WAIT_POLICY; PASSIVE keeps cpu/wall meaningful (see clean_env)")
    b.add_argument("--allow-dirty", action="store_true")
    b.add_argument("--json")
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args()
    try:
        sys.exit(args.fn(args) or 0)
    except IsolationError as e:
        print(f"\nISOLATION ERROR: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
