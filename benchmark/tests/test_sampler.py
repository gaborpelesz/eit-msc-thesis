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
