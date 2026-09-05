"""Host-side telemetry sampler.

R-MEM-02, R-MEM-03, R-MEM-04, R-PWR-01, R-PWR-02, R-IO-01, R-IO-02,
R-RUN-06, R-STO-04.

The sampler runs in the harness process on the host, outside the container, so
it survives the measured process and keeps the last readings before an OOM. It
attributes samples to the measured process tree, found from the container's
host PID.

Missed ticks are recorded, never hidden: every sample carries the interval that
actually elapsed and a `late` flag, and the summary reports how many ticks the
loop failed to keep.
"""

import threading
import time
from pathlib import Path

import psutil

try:  # nvidia-ml-py
    import pynvml
except ImportError:  # the sampler still records host metrics without a GPU
    pynvml = None

DEFAULT_INTERVAL_S = 0.1

COLUMNS = (
    "t_mono_ns",
    "t_unix_s",
    "elapsed_s",
    "interval_ms",
    "late",
    "proc_count",
    "host_rss_bytes",
    "host_uss_bytes",
    "gpu_mem_proc_bytes",
    "gpu_mem_device_bytes",
    "gpu_util_pct",
    "gpu_mem_util_pct",
    "gpu_power_w",
    "gpu_temp_c",
    "gpu_sm_clock_mhz",
    "gpu_mem_clock_mhz",
    "io_read_bytes",
    "io_write_bytes",
)


def _proc_io(pid):
    """(read_bytes, write_bytes) from /proc/<pid>/io, or None."""
    try:
        with open(f"/proc/{pid}/io") as f:
            values = {}
            for line in f:
                key, _, value = line.partition(":")
                if key in ("read_bytes", "write_bytes"):
                    values[key] = int(value)
        return values.get("read_bytes", 0), values.get("write_bytes", 0)
    except (OSError, ValueError):
        return None


class Sampler(threading.Thread):
    def __init__(
        self,
        gpu_index=0,
        interval_s=DEFAULT_INTERVAL_S,
        collect_uss=True,
        tree_refresh_s=1.0,
    ):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.tree_refresh_s = tree_refresh_s
        self.collect_uss = collect_uss
        self.root_pid = None
        self._stop = threading.Event()
        self._rows = {name: [] for name in COLUMNS}
        self.missed_ticks = 0
        self.late_samples = 0
        self.errors = []
        self.uss_disabled_reason = None
        self.device_mem_attribution = "device"
        self.device_mem_baseline_bytes = None
        self.gpu_memory_total_bytes = None
        self._nvml = None
        self._io_by_pid = {}
        self._tree_cache = None
        self._tree_refreshed = 0.0
        self._t0 = None

    def set_root_pid(self, pid):
        """The container's host PID; the measured tree is rooted at it."""
        self.root_pid = pid

    def stop(self):
        self._stop.set()

    def _nvml_init(self):
        if pynvml is None:
            self.errors.append("pynvml is not installed; no GPU telemetry was recorded")
            return
        try:
            pynvml.nvmlInit()
            self._nvml = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml)
            self.gpu_memory_total_bytes = int(info.total)
            self.device_mem_baseline_bytes = int(info.used)
        except Exception as exc:
            self._nvml = None
            self.errors.append(f"NVML unavailable: {exc}")

    def _gpu_sample(self, pids):
        if self._nvml is None:
            return {}
        sample = {}
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml)
            sample["gpu_mem_device_bytes"] = int(info.used)
        except Exception:
            pass
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._nvml)
            used = [
                int(p.usedGpuMemory)
                for p in procs
                if p.pid in pids and p.usedGpuMemory is not None
            ]
            if used:
                sample["gpu_mem_proc_bytes"] = sum(used)
                self.device_mem_attribution = "process"
        except Exception:
            pass
        for key, call in (
            ("gpu_power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(self._nvml) / 1000.0),
            ("gpu_temp_c", lambda: pynvml.nvmlDeviceGetTemperature(self._nvml, 0)),
            ("gpu_sm_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(self._nvml, 1)),
            ("gpu_mem_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(self._nvml, 2)),
        ):
            try:
                sample[key] = call()
            except Exception:
                pass
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml)
            sample["gpu_util_pct"] = int(rates.gpu)
            sample["gpu_mem_util_pct"] = int(rates.memory)
        except Exception:
            pass
        return sample

    def _tree(self, now):
        """The measured process tree, re-enumerated at `tree_refresh_s`.

        Enumerating descendants scans every process on the host and costs about
        25 ms here -- a quarter of the sampling interval -- so the list is
        cached and refreshed on a slower clock, and immediately whenever a
        member has exited. A child that both appears and exits inside one
        refresh window is missed; the window is recorded next to the peaks.
        """
        if self.root_pid is None:
            return []
        stale = (
            self._tree_cache is None
            or now - self._tree_refreshed > self.tree_refresh_s
            or not all(p.is_running() for p in self._tree_cache)
        )
        if not stale:
            return self._tree_cache
        self._tree_refreshed = now
        try:
            root = psutil.Process(self.root_pid)
        except psutil.Error:
            self._tree_cache = []
            return self._tree_cache
        try:
            self._tree_cache = [root, *root.children(recursive=True)]
        except psutil.Error:
            self._tree_cache = [root]
        return self._tree_cache

    def _host_sample(self, procs):
        rss = 0
        uss = 0
        uss_ok = self.collect_uss
        for proc in procs:
            try:
                rss += proc.memory_info().rss
            except psutil.Error:
                continue
            if uss_ok:
                try:
                    uss += proc.memory_full_info().uss
                except (psutil.Error, OSError) as exc:
                    uss_ok = False
                    if self.uss_disabled_reason is None:
                        self.uss_disabled_reason = f"{type(exc).__name__}: {exc}"
                    self.collect_uss = False
        return rss, (uss if uss_ok else None)

    def _io_sample(self, procs):
        """Sum monotonic per-pid counters, keeping the last value of dead pids."""
        for proc in procs:
            values = _proc_io(proc.pid)
            if values is not None:
                previous = self._io_by_pid.get(proc.pid, (0, 0))
                self._io_by_pid[proc.pid] = (
                    max(previous[0], values[0]),
                    max(previous[1], values[1]),
                )
        reads = sum(v[0] for v in self._io_by_pid.values())
        writes = sum(v[1] for v in self._io_by_pid.values())
        return reads, writes

    def _append(self, row):
        for name in COLUMNS:
            self._rows[name].append(row.get(name))

    def run(self):
        self._nvml_init()
        self._t0 = time.monotonic_ns()
        previous_ns = None
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now_ns = time.monotonic_ns()
            procs = self._tree(now_ns / 1e9)
            pids = {p.pid for p in procs}
            rss, uss = self._host_sample(procs)
            reads, writes = self._io_sample(procs)
            row = {
                "t_mono_ns": now_ns,
                "t_unix_s": time.time(),
                "elapsed_s": (now_ns - self._t0) / 1e9,
                "interval_ms": None if previous_ns is None else (now_ns - previous_ns) / 1e6,
                "late": False,
                "proc_count": len(procs),
                "host_rss_bytes": rss,
                "host_uss_bytes": uss,
                "io_read_bytes": reads,
                "io_write_bytes": writes,
            }
            row.update(self._gpu_sample(pids))
            if row["interval_ms"] is not None and row["interval_ms"] > 1.5 * self.interval_s * 1000:
                row["late"] = True
                self.late_samples += 1
                self.missed_ticks += int(row["interval_ms"] / (self.interval_s * 1000)) - 1
            self._append(row)
            previous_ns = now_ns

            next_tick += self.interval_s
            delay = next_tick - time.monotonic()
            if delay < 0:
                # The loop fell behind; skip the ticks it cannot take rather
                # than sampling in a burst, and count them.
                skipped = int(-delay / self.interval_s) + 1
                next_tick += skipped * self.interval_s
                delay = next_tick - time.monotonic()
            self._stop.wait(max(0.0, delay))

        if self._nvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @property
    def sample_count(self):
        return len(self._rows["t_mono_ns"])

    def table(self):
        import pyarrow as pa

        schema = pa.schema(
            [
                ("t_mono_ns", pa.int64()),
                ("t_unix_s", pa.float64()),
                ("elapsed_s", pa.float64()),
                ("interval_ms", pa.float64()),
                ("late", pa.bool_()),
                ("proc_count", pa.int32()),
                ("host_rss_bytes", pa.int64()),
                ("host_uss_bytes", pa.int64()),
                ("gpu_mem_proc_bytes", pa.int64()),
                ("gpu_mem_device_bytes", pa.int64()),
                ("gpu_util_pct", pa.int32()),
                ("gpu_mem_util_pct", pa.int32()),
                ("gpu_power_w", pa.float64()),
                ("gpu_temp_c", pa.int32()),
                ("gpu_sm_clock_mhz", pa.int32()),
                ("gpu_mem_clock_mhz", pa.int32()),
                ("io_read_bytes", pa.int64()),
                ("io_write_bytes", pa.int64()),
            ]
        )
        return pa.table({name: self._rows[name] for name in COLUMNS}, schema=schema)

    def write_parquet(self, path):
        import pyarrow.parquet as pq

        path = Path(path)
        pq.write_table(self.table(), path)
        return path

    def summary(self):
        """Scalars for run.json. Every peak is stated with its interval."""

        def peak(name):
            values = [v for v in self._rows[name] if v is not None]
            return max(values) if values else None

        def last(name):
            values = [v for v in self._rows[name] if v is not None]
            return values[-1] if values else None

        power = self._rows["gpu_power_w"]
        times = self._rows["t_mono_ns"]
        energy = 0.0
        seen = False
        for i in range(1, len(times)):
            p0, p1 = power[i - 1], power[i]
            if p0 is None or p1 is None:
                continue
            energy += 0.5 * (p0 + p1) * (times[i] - times[i - 1]) / 1e9
            seen = True

        device_peak = peak("gpu_mem_proc_bytes")
        if device_peak is None:
            device_peak = peak("gpu_mem_device_bytes")
        return {
            "sample_interval_ms": self.interval_s * 1000,
            "tree_refresh_s": self.tree_refresh_s,
            "sample_count": self.sample_count,
            "late_samples": self.late_samples,
            "missed_ticks": self.missed_ticks,
            "peak_host_rss_bytes": peak("host_rss_bytes"),
            "peak_host_uss_bytes": peak("host_uss_bytes"),
            "host_uss_unavailable_reason": self.uss_disabled_reason,
            "peak_device_mem_bytes": device_peak,
            "device_mem_attribution": self.device_mem_attribution,
            "device_mem_baseline_bytes": self.device_mem_baseline_bytes,
            "device_mem_total_bytes": self.gpu_memory_total_bytes,
            "last_device_mem_bytes": last("gpu_mem_proc_bytes") or last("gpu_mem_device_bytes"),
            "peak_gpu_temp_c": peak("gpu_temp_c"),
            "mean_gpu_power_w": (
                sum(p for p in power if p is not None) / max(1, sum(1 for p in power if p is not None))
                if any(p is not None for p in power)
                else None
            ),
            "energy_j": energy if seen else None,
            "io_read_bytes": peak("io_read_bytes"),
            "io_write_bytes": peak("io_write_bytes"),
            "sampler_errors": list(self.errors),
        }


def warm_cache(path, max_bytes=None):
    """R-IO-03: read the input dataset once so every run starts warm.

    Returns the number of bytes read. Caches are never dropped.
    """
    total = 0
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for file in files:
        try:
            with open(file, "rb") as f:
                while True:
                    chunk = f.read(4 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total >= max_bytes:
                        return total
        except OSError:
            continue
    return total
