import os
import time

import pyarrow.parquet as pq

from bench import sampler as smp


def test_sampler_collects_rows_and_writes_parquet(tmp_path):
    sampler = smp.Sampler(interval_s=0.02)
    sampler.set_root_pid(os.getpid())
    sampler.start()
    time.sleep(0.25)
    sampler.stop()
    sampler.join(timeout=5)

    assert sampler.sample_count >= 5
    path = sampler.write_parquet(tmp_path / "telemetry.parquet")
    table = pq.read_table(path)
    assert table.num_rows == sampler.sample_count
    assert set(table.column_names) == set(smp.COLUMNS)

    intervals = [v for v in table.column("interval_ms").to_pylist() if v is not None]
    assert intervals and min(intervals) > 0

    summary = sampler.summary()
    assert summary["sample_interval_ms"] == 20.0
    assert summary["peak_host_rss_bytes"] > 0
    assert summary["sample_count"] == sampler.sample_count
    assert summary["missed_ticks"] >= 0


def test_sampler_without_a_pid_still_produces_a_series(tmp_path):
    sampler = smp.Sampler(interval_s=0.02)
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    sampler.join(timeout=5)
    assert sampler.sample_count >= 1
    assert sampler.summary()["peak_host_rss_bytes"] == 0


def test_warm_cache_reads_every_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x" * 1000)
    (tmp_path / "sub" / "b.jpg").write_bytes(b"y" * 2000)
    assert smp.warm_cache(tmp_path) == 3000


def test_cgroup_io_sums_every_device(tmp_path):
    (tmp_path / "io.stat").write_text(
        "259:2 rbytes=8192 wbytes=1063014400 rios=2 wios=1221 dbytes=0 dios=0\n"
        "8:0 rbytes=1000 wbytes=2000 rios=1 wios=1 dbytes=0 dios=0\n"
    )
    assert smp._cgroup_io(tmp_path) == (9192, 1063016400)


def test_cgroup_for_container_needs_a_readable_io_stat(tmp_path):
    scope = tmp_path / "docker-abc.scope"
    scope.mkdir()
    roots = (str(tmp_path / "docker-{cid}.scope"),)
    assert smp.cgroup_for_container("abc", roots) is None
    (scope / "io.stat").write_text("")
    assert smp.cgroup_for_container("abc", roots) == scope
    assert smp.cgroup_for_container(None, roots) is None


def test_unreadable_io_counters_are_null_not_zero(tmp_path, monkeypatch):
    """`/proc/<pid>/io` of a root-owned container process is not readable by
    the harness. A recorded 0 would read as `this method performed no I/O`."""
    monkeypatch.setattr(smp, "_proc_io", lambda pid: None)
    sampler = smp.Sampler(interval_s=0.02)
    sampler.set_root_pid(os.getpid())
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    sampler.join(timeout=5)

    summary = sampler.summary()
    assert summary["io_read_bytes"] is None
    assert summary["io_write_bytes"] is None
    assert summary["io_attribution"] is None
    assert "readable" in summary["io_unavailable_reason"]


def test_the_cgroup_is_preferred_over_per_process_counters(tmp_path, monkeypatch):
    (tmp_path / "io.stat").write_text("259:2 rbytes=7 wbytes=11 rios=1 wios=1\n")
    monkeypatch.setattr(smp, "_proc_io", lambda pid: (10**9, 10**9))
    sampler = smp.Sampler(interval_s=0.02)
    sampler.set_root_pid(os.getpid())
    sampler.set_cgroup(tmp_path)
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    sampler.join(timeout=5)

    summary = sampler.summary()
    assert (summary["io_read_bytes"], summary["io_write_bytes"]) == (7, 11)
    assert summary["io_attribution"] == "cgroup"
