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


def locked_clocks_state(gpu_index=0):
    """(locked, evidence) for the benchmark GPU.

    Two mechanisms exist and only one of them is readable on a given GPU:
    `nvidia-smi -ac` (applications clocks, datacentre parts) and
    `nvidia-smi -lgc` (locked clocks). NVML exposes a getter for the second
    only from recent drivers, so both are probed and the state is reported
    unknown -- never assumed locked -- when neither answers.
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
            return True, evidence
    return None if "locked_clocks_mhz" not in evidence else False, evidence


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
