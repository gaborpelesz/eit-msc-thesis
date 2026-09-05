"""System fingerprint: capture, bind a campaign to it, and check the GPU state.

R-ENV-02, R-ENV-03, R-ENV-04, R-ENV-06, R-EXP-03, R-RES-04, R-STO-01.

The fingerprint is captured once per campaign start, written to
`fingerprint.json` in the campaign directory and repeated in every run record.
Four of its fields are load-bearing (`LOAD_BEARING`); a machine that differs in
any of them may not start or continue the campaign.
"""

import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

# R-ENV-06.
LOAD_BEARING = ("gpu_model", "driver_version", "cuda_version", "image_digest")


class FingerprintMismatch(Exception):
    """The campaign is bound to a different machine (R-ENV-06 / D13)."""


class EnvironmentNotReady(Exception):
    """GPU clocks, persistence mode or display state fail R-ENV-02."""


def _run(argv, timeout=60):
    """Return stdout, or None when the command is missing or fails."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def nvidia_smi_query(fields, gpu_index=0):
    out = _run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader",
        ]
    )
    if out is None:
        return None
    values = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    return dict(zip(fields, values))


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def os_release():
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def memory_total_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def block_devices():
    out = _run(["lsblk", "-dno", "NAME,MODEL,SIZE"])
    if out is None:
        return []
    return [" ".join(line.split()) for line in out.strip().splitlines() if line.strip()]


def driver_cuda_version():
    """The CUDA version the *driver* reports, not the toolkit in the image."""
    out = _run(["nvidia-smi", "-q"])
    if out is None:
        return None
    for line in out.splitlines():
        if line.strip().startswith("CUDA Version"):
            return line.split(":", 1)[1].strip()
    return None


def image_identity(image, docker="docker"):
    """Image id and repo digests. The digest is the campaign-binding field."""
    out = _run([docker, "image", "inspect", "--format", "{{json .}}", image])
    if out is None:
        return {"image": image, "image_id": None, "image_repo_digests": [], "image_digest": None}
    info = json.loads(out.strip().splitlines()[0])
    digests = info.get("RepoDigests") or []
    return {
        "image": image,
        "image_id": info.get("Id"),
        "image_repo_digests": digests,
        # A locally built image has no repo digest; its content-addressed Id is
        # the only identity it has, and it is what binds the campaign.
        "image_digest": digests[0] if digests else info.get("Id"),
    }


def probe_image_toolchain(image, docker="docker"):
    """Compiler identity as it exists inside the image (R-ENV-03)."""
    probes = {
        "host_cxx": ["clang++-18", "--version"],
        "nvcc": ["/usr/local/cuda-12.8/bin/nvcc", "--version"],
    }
    result = {}
    for name, argv in probes.items():
        out = _run([docker, "run", "--rm", "--entrypoint", argv[0], image, *argv[1:]], timeout=180)
        result[name] = out.strip().splitlines()[0] if out else None
    return result


def submodule_shas(repo_root):
    """Every submodule path in the superproject's .gitmodules -> its HEAD.

    A list of records rather than a mapping, because the fingerprint is
    embedded in every run record and DuckDB unifies a list of structs across
    runs but not a struct whose keys differ between machines (R-STO-06). A
    private submodule that is not initialised on this machine records null.
    """
    repo_root = Path(repo_root)
    out = _run(
        ["git", "-C", str(repo_root), "config", "--file", ".gitmodules", "--get-regexp", r"path$"]
    )
    shas = []
    for line in (out or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        path = parts[1].strip()
        head = _run(["git", "-C", str(repo_root / path), "rev-parse", "HEAD"])
        shas.append({"path": path, "sha": head.strip() if head else None})
    return sorted(shas, key=lambda r: r["path"])


def collect(image, gpu_index=0, repo_root=None, docker="docker", probe_image=True):
    gpu = nvidia_smi_query(
        [
            "name",
            "driver_version",
            "uuid",
            "memory.total",
            "pci.bus_id",
            "persistence_mode",
            "display_active",
        ],
        gpu_index=gpu_index,
    ) or {}
    fp = {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "gpu_index": gpu_index,
        "gpu_model": gpu.get("name"),
        "gpu_uuid": gpu.get("uuid"),
        "gpu_memory_total": gpu.get("memory.total"),
        "gpu_pci_bus_id": gpu.get("pci.bus_id"),
        "driver_version": gpu.get("driver_version"),
        "cuda_version": driver_cuda_version(),
        "ram_total_bytes": memory_total_bytes(),
        # RAM part numbers need dmidecode and root; recorded as unavailable
        # rather than guessed.
        "ram_model": None,
        "block_devices": block_devices(),
        "os": os_release(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }
    fp.update(image_identity(image, docker=docker))
    fp["toolchain"] = probe_image_toolchain(image, docker=docker) if probe_image else None
    fp["submodules"] = submodule_shas(repo_root) if repo_root else []
    return fp


def _differences(a, b, fields=LOAD_BEARING):
    return {f: (a.get(f), b.get(f)) for f in fields if a.get(f) != b.get(f)}


def bind(campaign_dir, fingerprint, declared=None):
    """Bind the campaign to this machine, or refuse to continue on another.

    Returns the fingerprint written or already on disk. Raises
    FingerprintMismatch (the caller exits non-zero) when a load-bearing field
    differs from the campaign's `fingerprint.json` or from the fields the
    specification declared (R-ENV-06, R-EXP-03).
    """
    campaign_dir = Path(campaign_dir)
    if declared:
        declared_diff = _differences(
            declared, fingerprint, [f for f in LOAD_BEARING if f in declared]
        )
        if declared_diff:
            raise FingerprintMismatch(
                "this machine does not match the fingerprint the specification "
                "declares:\n"
                + "\n".join(
                    f"  {f}: spec={want!r} machine={got!r}"
                    for f, (want, got) in declared_diff.items()
                )
            )

    path = campaign_dir / "fingerprint.json"
    if path.exists():
        existing = json.loads(path.read_text())
        diff = _differences(existing, fingerprint)
        if diff:
            raise FingerprintMismatch(
                f"campaign {campaign_dir.name} is bound to a different machine; "
                "its measurements would not be comparable (R-ENV-06, D13):\n"
                + "\n".join(
                    f"  {f}: campaign={was!r} machine={now!r}"
                    for f, (was, now) in diff.items()
                )
                + "\nRun the remaining runs on the original machine, or start a "
                "new campaign."
            )
        return existing

    campaign_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    return fingerprint


# R-ENV-02, third detection route. An idle GPU whose clocks are not locked
# drops to the lowest supported graphics clock (P8, ~300 MHz on consumer
# parts); one that holds a higher clock while idle is holding it because
# something applied a lock. The inference is only valid at idle, so a busy GPU
# is reported unknown rather than assumed locked.
IDLE_CLOCK_SAMPLES = 10
IDLE_CLOCK_INTERVAL_S = 0.1
IDLE_UTILIZATION_MAX_PCT = 10


def _mhz(value):
    """`1800 MHz` -> 1800; `N/A`, `[N/A]` and anything else -> None."""
    try:
        return int(str(value).strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return None


def supported_graphics_clocks(gpu_index=0):
    """Every supported graphics clock in MHz, ascending, or None.

    The smallest entry is the idle floor the driver falls back to, so it is
    read from the device rather than assumed to be 300 MHz.
    """
    out = _run(["nvidia-smi", f"--id={gpu_index}", "-q", "-d", "SUPPORTED_CLOCKS"])
    if out is None:
        return None
    values = []
    for line in out.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "Graphics":
            mhz = _mhz(value)
            if mhz is not None:
                values.append(mhz)
    return sorted(set(values)) or None


def sample_clocks(
    gpu_index=0, samples=IDLE_CLOCK_SAMPLES, interval_s=IDLE_CLOCK_INTERVAL_S, sleep=time.sleep
):
    """`samples` readings of the SM and graphics clock, over the idle window."""
    rows = []
    for index in range(samples):
        if index:
            sleep(interval_s)
        query = nvidia_smi_query(
            ["clocks.sm", "clocks.gr", "clocks.max.sm", "utilization.gpu"], gpu_index
        )
        if query is None:
            break
        rows.append(
            {
                "sm_mhz": _mhz(query.get("clocks.sm")),
                "gr_mhz": _mhz(query.get("clocks.gr")),
                "max_sm_mhz": _mhz(query.get("clocks.max.sm")),
                "utilization_pct": _mhz(query.get("utilization.gpu")),
            }
        )
    return rows


def infer_locked_clocks(samples, floor_mhz=None, idle_utilization_max_pct=IDLE_UTILIZATION_MAX_PCT):
    """(locked, evidence) from a window of idle clock samples.

    True only when every sample carries the same SM clock, that clock is above
    the device's idle floor, and the GPU was idle throughout. Anything else --
    a varying clock, a clock sitting on the floor, a busy GPU, an unreadable
    floor -- is None: the state is unknown and R-ENV-02 refuses the run.
    """
    evidence = {
        "samples": len(samples),
        "sm_mhz": sorted({s.get("sm_mhz") for s in samples}, key=lambda v: (v is None, v)),
        "floor_mhz": floor_mhz,
        "max_sm_mhz": next((s.get("max_sm_mhz") for s in samples if s.get("max_sm_mhz")), None),
        "utilization_max_pct": max(
            (s.get("utilization_pct") for s in samples if s.get("utilization_pct") is not None),
            default=None,
        ),
    }

    def unknown(reason):
        evidence["verdict"] = reason
        return None, evidence

    if len(samples) < 3:
        return unknown("fewer than three clock samples were read")
    values = [s.get("sm_mhz") for s in samples]
    if any(v is None for v in values):
        return unknown("the SM clock was not readable in every sample")
    if len(set(values)) != 1:
        return unknown(f"the SM clock varied over the window: {sorted(set(values))}")
    value = values[0]
    utilization = evidence["utilization_max_pct"]
    if utilization is None:
        return unknown("GPU utilization was not readable, so the window is not known to be idle")
    if utilization > idle_utilization_max_pct:
        return unknown(
            f"the GPU was busy during the window ({utilization}% > "
            f"{idle_utilization_max_pct}%); a steady clock under load is not evidence of a lock"
        )
    if floor_mhz is None:
        return unknown(
            "the device's supported clocks are unreadable, so a held clock cannot "
            "be told apart from the idle floor"
        )
    if value <= floor_mhz:
        return unknown(
            f"the SM clock sits on the idle floor ({value} MHz <= {floor_mhz} MHz); "
            "that is what an unlocked GPU does at idle"
        )
    evidence["verdict"] = "held above the idle floor across the whole idle window"
    return True, evidence


def locked_clocks_state(gpu_index=0, sample=None, floor=None):
    """(locked, evidence) for the benchmark GPU.

    Two mechanisms exist to lock a GPU and only one of them is readable on a
    given part: `nvidia-smi -ac` (applications clocks, datacentre parts) and
    `nvidia-smi -lgc` (locked clocks). NVML exposes a getter for the second
    only from recent drivers -- driver 575 on the RTX 2080 Ti has neither --
    so both are probed first, and only when neither answers is the state
    inferred from the idle clock itself. It is never assumed.
    """
    evidence = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            getter = getattr(pynvml, "nvmlDeviceGetGpuLockedClocks", None)
            if getter is not None:
                try:
                    low, high = getter(handle)
                    evidence["locked_clocks_mhz"] = [low, high]
                    if low and high:
                        return True, evidence
                except pynvml.NVMLError as exc:
                    evidence["nvmlDeviceGetGpuLockedClocks"] = str(exc)
            else:
                evidence["nvmlDeviceGetGpuLockedClocks"] = "not available in this NVML"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # pynvml missing, no driver, no device
        evidence["pynvml"] = str(exc)

    applications = nvidia_smi_query(
        ["clocks.applications.graphics", "clocks.applications.memory"], gpu_index
    )
    if applications:
        evidence.update(applications)
        graphics = applications.get("clocks.applications.graphics", "")
        if graphics and not graphics.startswith("[") and graphics != "N/A":
            evidence["locked_clocks_method"] = "applications-clocks"
            return True, evidence

    # A getter that answered with a zero range is a definite "not locked"; only
    # a device that answered nothing at all is a candidate for the inference.
    if "locked_clocks_mhz" in evidence:
        return False, evidence

    samples = (sample or sample_clocks)(gpu_index)
    floor_clocks = (floor or supported_graphics_clocks)(gpu_index)
    locked, inference = infer_locked_clocks(
        samples, floor_mhz=floor_clocks[0] if floor_clocks else None
    )
    evidence["idle_clock_inference"] = inference
    if locked:
        evidence["locked_clocks_method"] = "idle-clock-inference"
        evidence["locked_clocks_inferred_mhz"] = samples[0]["sm_mhz"]
        return True, evidence
    return None, evidence


def expected_clock_mhz(evidence):
    """The SM clock the pre-run gate established, for the post-run check.

    None when the gate proved a lock without pinning it to one value (an
    applications-clock pair, or a locked range whose ends differ): the run is
    then recorded with its observed range and no verdict, rather than with a
    verdict invented here.
    """
    clocks = (evidence or {}).get("clocks", evidence) or {}
    inferred = clocks.get("locked_clocks_inferred_mhz")
    if inferred:
        return int(inferred)
    low_high = clocks.get("locked_clocks_mhz")
    if isinstance(low_high, (list, tuple)) and len(low_high) == 2 and low_high[0] == low_high[1]:
        return int(low_high[0]) or None
    return _mhz(clocks.get("clocks.applications.graphics"))


# D24.2. `-lgc <v>,<v>` is a request the hardware can still miss for a few
# samples (CUMVS dipped to 1635 MHz for ~7 of 120 samples in a 12 s run), so a
# strict "every sample at the locked value" verdict would be false for nearly
# every run of a multi-hour campaign and could exclude nothing. The duty cycle
# is what analysis uses; `held` is a convenience flag over it.
CLOCK_TOLERANCE_FRACTION = 0.01
CLOCK_HELD_MIN_FRACTION = 0.95


def clock_hold(values, expected_mhz, method=None):
    """R-ENV-02 after the fact: did the SM clock hold for the whole run?

    `fraction_at_expected` is the share of 100 ms samples within +-1 % of the
    clock the pre-run gate established, and `held` is that fraction against the
    D24.2 threshold. A run whose clock did not hold is still a run -- it is
    recorded with `held: false` so that a query can exclude it, never discarded
    or retried.
    """
    readings = [int(v) for v in values if v is not None]
    low = min(readings) if readings else None
    high = max(readings) if readings else None
    fraction = None
    held = None
    if expected_mhz is not None and readings:
        tolerance = abs(expected_mhz) * CLOCK_TOLERANCE_FRACTION
        at_expected = sum(1 for v in readings if abs(v - expected_mhz) <= tolerance)
        fraction = at_expected / len(readings)
        held = fraction >= CLOCK_HELD_MIN_FRACTION
    return {
        "expected_mhz": expected_mhz,
        "min_mhz": low,
        "max_mhz": high,
        "samples": len(readings),
        "fraction_at_expected": fraction,
        "held": held,
        "method": method,
    }


def clock_hold_from_table(table, expected_mhz, method=None, column="gpu_sm_clock_mhz"):
    """`clock_hold` over a telemetry table (anything with `.column(name)`)."""
    try:
        values = table.column(column).to_pylist()
    except (KeyError, AttributeError):
        values = []
    return clock_hold(values, expected_mhz, method=method)


def check_gpu_state(gpu_index=0):
    """R-ENV-02: locked clocks, persistence mode, no display attached.

    Returns (problems, evidence). A non-empty `problems` list means the run may
    not be executed. An indeterminate clock state counts as a problem: the
    conservative reading of R-ENV-02 is that an unverified environment is not a
    measurement environment.
    """
    problems = []
    evidence = {}

    query = nvidia_smi_query(["persistence_mode", "display_active", "display_mode"], gpu_index)
    if query is None:
        return (
            ["nvidia-smi did not answer; the benchmark GPU state cannot be verified"],
            evidence,
        )
    evidence.update(query)
    if query.get("persistence_mode") != "Enabled":
        problems.append(
            f"persistence mode is {query.get('persistence_mode')!r}; "
            f"enable it with `nvidia-smi -i {gpu_index} -pm 1`"
        )
    display = query.get("display_active")
    if display not in ("Disabled", "[N/A]", "N/A"):
        problems.append(f"a display is attached to GPU {gpu_index} (display_active={display!r})")

    locked, clock_evidence = locked_clocks_state(gpu_index)
    evidence["clocks"] = clock_evidence
    if locked is not True:
        problems.append(
            "GPU clocks are not verifiably locked "
            f"(state={'unknown' if locked is None else 'unlocked'}); "
            f"lock them with `nvidia-smi -i {gpu_index} -lgc <min>,<max>`"
        )
    return problems, evidence
