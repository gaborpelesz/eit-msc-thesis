import json
import os
import socket

from bench import runner as rn
from bench import sampler as smp
from bench import spec as sp
from bench import store as st
from deviations import manifest as mf

ROOT = mf.repo_root(mf.default_manifest_path().parent)


def _result(status="ok", wall=12.5, f1=0.71):
    return {
        "status": status,
        "exit_code": 0,
        "wall_time_s": wall,
        "wall_time_with_preprocess_s": wall + 3.0,
        "started_at": "2026-09-05T18:00:00+00:00",
        "ended_at": "2026-09-05T18:00:12+00:00",
        "work_dir": "/scratch/bench/x",
        "preprocess": {"argv": ["convert"], "exit_code": 0, "duration_s": 3.0},
        "status_evidence": {"exit_code": 0, "stderr_markers": [], "container": {}},
        "telemetry": {
            "sample_interval_ms": 100.0,
            "sample_count": 125,
            "missed_ticks": 0,
            "peak_host_rss_bytes": 4 << 30,
            "peak_host_uss_bytes": None,
            "peak_device_mem_bytes": 6 << 30,
            "device_mem_attribution": "process",
            "device_mem_total_bytes": 12 << 30,
            "energy_j": 4321.0,
            "mean_gpu_power_w": 210.0,
            "peak_gpu_temp_c": 63,
            "io_read_bytes": 1 << 30,
            "io_write_bytes": 1 << 28,
        },
        "phases": [
            {"name": "patchmatch", "total_s": 8.0, "self_s": 8.0, "count": 9, "unclosed": 0,
             "source": "method"},
            {"name": "preprocess.convert", "total_s": 3.0, "self_s": 3.0, "count": 1,
             "unclosed": 0, "source": "harness"},
        ],
        "phases_by_pass": [{"name": "image_pass", "pass": "photometric", "total_s": 4.0,
                            "count": 3}],
        "phase_trace_errors": [],
        "neighbour_list": {"path": "/scratch/pair.txt", "sha256": "d" * 64},
        "point_cloud": {
            "path": "/archive/x.ply",
            "size_bytes": 1234,
            "sha256": "a" * 64,
            "point_count": 999,
        },
        "quality": [
            {"tolerance": 0.02, "accuracy": 0.8, "completeness": 0.6, "f1": f1},
            {"tolerance": 0.05, "accuracy": 0.9, "completeness": 0.7, "f1": f1 + 0.1},
        ],
        "command": ["docker", "run"],
        "notes": [],
        "warm_cache_bytes": 100,
    }


def _record(spec, manifest, run, result):
    entry = sp.method_entry(manifest, run.method)
    deviations = {"methods": [{"name": entry["name"], "fork_sha": "f" * 40}], "global": []}
    fingerprint = {"gpu_model": "RTX 5070", "image_digest": "sha256:x", "submodules": []}
    return st.build_record(spec, manifest, entry, run, result, fingerprint, deviations, ROOT)


def test_record_round_trips_through_duckdb(write_spec, spec_dict, manifest, tmp_path):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)

    for index, run in enumerate(sp.expand(spec)[:3]):
        tmp_dir, _ = rn.open_run_dir(campaign, run.key)
        record = _record(spec, manifest, run, _result(status="ok" if index else "timeout"))
        st.write_record(tmp_dir, record)
        smp.Sampler().write_parquet(tmp_dir / "telemetry.parquet")
        rn.commit_run_dir(tmp_dir)

    con = st.connect(campaign)
    rows = con.execute(
        "SELECT run_key, method, width, configuration, repeat, status, wall_time_s, "
        "f1_primary, peak_device_mem_bytes, sample_interval_ms, provenance FROM runs "
        "ORDER BY run_key"
    ).fetchall()
    assert len(rows) == 3
    assert all(row[10] == "published-reference" for row in rows)
    assert {row[5] for row in rows} == {"ok", "timeout"}

    # every canned report parses and runs
    for statement in st.CANNED_REPORTS.values():
        con.execute(statement).fetchall()

    phases = con.execute(
        "SELECT p.name, p.total_s FROM runs, UNNEST(phases) AS t(p) "
        "WHERE p.source = 'method' LIMIT 1"
    ).fetchall()
    assert phases[0][0] == "patchmatch"

    telemetry = con.execute("SELECT count(*) FROM telemetry").fetchone()
    assert telemetry[0] == 0


def test_superseded_and_tmp_directories_stay_out_of_queries(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)
    run = sp.expand(spec)[0]

    tmp_dir, _ = rn.open_run_dir(campaign, run.key)
    st.write_record(tmp_dir, _record(spec, manifest, run, _result()))
    rn.commit_run_dir(tmp_dir)
    tmp_dir2, moved = rn.open_run_dir(campaign, run.key, rerun=True)
    st.write_record(tmp_dir2, _record(spec, manifest, run, _result(wall=99.0)))
    rn.commit_run_dir(tmp_dir2)

    assert len(st.finished_run_dirs(campaign)) == 1
    con = st.connect(campaign)
    assert con.execute("SELECT wall_time_s FROM runs").fetchall() == [(99.0,)]
    assert (campaign / moved[0]).exists()


def test_record_never_overwritten(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)
    run = sp.expand(spec)[0]
    tmp_dir, _ = rn.open_run_dir(campaign, run.key)
    st.write_record(tmp_dir, _record(spec, manifest, run, _result()))
    try:
        st.write_record(tmp_dir, _record(spec, manifest, run, _result(wall=1.0)))
    except FileExistsError:
        pass
    else:
        raise AssertionError("a run record must never be overwritten")


def test_configuration_consistency_is_enforced(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)
    run = sp.expand(spec)[0]
    tmp_dir, _ = rn.open_run_dir(campaign, run.key)
    record = _record(spec, manifest, run, _result())
    st.write_record(tmp_dir, record)
    rn.commit_run_dir(tmp_dir)

    same = st.check_configuration_consistency(
        campaign, run, record["parameters_json"], record["fork_sha"]
    )
    assert same == []

    conflicts = st.check_configuration_consistency(
        campaign, run, json.dumps({"neighbours": 3}), "0" * 40
    )
    assert len(conflicts) == 2


def test_normalizations_depend_on_the_configuration(write_spec, spec_dict, manifest):
    spec_dict["methods"] = ["APD-MVS"]
    spec_dict["configurations"] = ["author", "norm10"]
    spec_dict["padding"] = "all"
    spec = sp.load(write_spec(spec_dict), manifest)
    entry = sp.method_entry(manifest, "APD-MVS")
    author = next(r for r in sp.expand(spec) if r.configuration == "author")
    norm = next(r for r in sp.expand(spec) if r.configuration == "norm10")

    author_norms = st.active_normalizations(manifest, entry, spec, author)
    norm_norms = st.active_normalizations(manifest, entry, spec, norm)
    assert len(norm_norms) > len(author_norms)
    assert not any(
        "shared" in n["description"] and n["scope"] == "global" for n in author_norms
    )
    # --padding reaches the shared converter, so it is in force under norm10
    # and not under author, where the fork runs its own converter.
    assert any(n.get("reversible") == "flag --padding" for n in norm_norms)
    assert not any(n.get("reversible") == "flag --padding" for n in author_norms)
    assert any(n.get("reversible") == "flag --neighbours=20" for n in norm_norms)


def test_only_normalizations_the_invocation_carries_are_in_force(
    write_spec, spec_dict, manifest
):
    """R-REP-01: the list says what ran, not what the manifest declares."""
    spec_dict["methods"] = ["ACMH", "CUMVS"]
    spec_dict["configurations"] = ["norm10"]
    spec = sp.load(write_spec(spec_dict), manifest)

    acmh_entry = sp.method_entry(manifest, "ACMH")
    acmh = next(r for r in sp.expand(spec) if r.method == "ACMH")
    declared = st.declared_normalizations(manifest, acmh_entry, spec, acmh)
    # No fork implements --no-debug-output yet and the harness passes none, so
    # the global flag is declared but not in force.
    debug = [d for d in declared if d.get("reversible") == "flag --no-debug-output"]
    assert debug and not any(d["in_force"] for d in debug)
    # ACMH's own "needs no action" entry is likewise not something in force.
    assert not any(
        d["in_force"] for d in declared if "needs no action" in d["description"]
    )
    assert any(d["in_force"] and d["scope"] == "global" for d in declared)

    cumvs_entry = sp.method_entry(manifest, "CUMVS")
    cumvs = next(r for r in sp.expand(spec) if r.method == "CUMVS")
    cumvs_declared = st.declared_normalizations(manifest, cumvs_entry, spec, cumvs)
    # CUMVS preprocesses with its own initializer, so the shared-converter
    # normalization is declared globally but does not apply to it.
    shared = [d for d in cumvs_declared if d.get("reversible") == "configuration author"]
    assert shared and not any(d["in_force"] for d in shared)
    assert "did not preprocess with the shared converter" in shared[0]["in_force_reason"]
    # Its neighbour count does reach it, through the initializer's argv.
    assert any(
        d["in_force"] and d.get("reversible") == "flag --max-neighbors=20"
        for d in cumvs_declared
    )


def test_the_cumvs_initializer_is_identified_by_its_source(write_spec, spec_dict, manifest):
    spec_dict["methods"] = ["CUMVS"]
    spec = sp.load(write_spec(spec_dict), manifest)
    entry = sp.method_entry(manifest, "CUMVS")
    run = sp.expand(spec)[0]
    record = st.converter_record(spec, entry, run, ROOT)
    assert record["kind"] == "initializer"
    assert record["identity"] == "source"
    # The binary exists only inside the image, so no host hash is invented.
    assert record["sha256"] is None
    assert record["path"].endswith("app_initialize_ETH3D")
    assert record["source_path"].endswith("samples/app_initialize_ETH3D.cpp")
    assert len(record["source_sha256"]) == 64


def test_init_campaign_copies_the_spec_and_refuses_a_changed_one(
    write_spec, spec_dict, manifest
):
    spec = sp.load(write_spec(spec_dict), manifest)
    fingerprint = {"gpu_model": "g", "driver_version": "d", "cuda_version": "c",
                   "image_digest": "i"}
    st.init_campaign(spec, fingerprint, {"methods": []})
    assert (spec.campaign_dir / "spec.yaml").read_text() == spec.text
    assert (spec.campaign_dir / "deviations.json").exists()

    spec_dict["repeats"] = 5
    changed = sp.load(write_spec(spec_dict, "changed.yaml"), manifest)
    changed.campaign = spec.campaign
    try:
        st.init_campaign(changed, fingerprint, {"methods": []})
    except FileExistsError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("a campaign's specification must not be silently replaced")


def test_status_report_counts_in_flight_and_remaining(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)
    runs = sp.expand(spec)

    tmp_dir, _ = rn.open_run_dir(campaign, runs[0].key)
    st.write_record(tmp_dir, _record(spec, manifest, runs[0], _result()))
    rn.commit_run_dir(tmp_dir)
    in_flight, _ = rn.open_run_dir(campaign, runs[1].key)
    st.write_host_marker(in_flight)

    report = st.status_report(campaign, runs)
    assert report["by_status"] == {"ok": 1}
    assert report["in_flight"][0]["run_key"] == runs[1].key
    assert report["in_flight"][0]["hostname"]
    assert len(report["remaining"]) == len(runs) - 1


def test_status_says_whether_an_in_flight_run_is_still_alive(tmp_path):
    """A `.tmp` left by a crash looks exactly like a run in flight (the first
    real campaign left one for four minutes), so the report says which."""
    campaign = tmp_path / "campaign"
    for key, marker in (
        ("A__s__w1__author__r1", f"{socket.gethostname()} pid={os.getpid()}"),
        ("B__s__w1__author__r1", f"{socket.gethostname()} pid=999999999"),
        ("C__s__w1__author__r1", "another-host pid=17"),
    ):
        directory = campaign / f"{key}.tmp"
        directory.mkdir(parents=True)
        (directory / "host.txt").write_text(marker + "\n")

    alive = {row["run_key"]: row["alive"] for row in st.status_report(campaign)["in_flight"]}
    assert alive == {
        "A__s__w1__author__r1": True,
        "B__s__w1__author__r1": False,
        "C__s__w1__author__r1": None,
    }


def test_debug_output_off_puts_the_r_exp_11_normalization_in_force(
    write_spec, spec_dict, manifest
):
    spec_dict["methods"] = ["APD-MVS"]
    spec = sp.load(write_spec(spec_dict), manifest)
    entry = sp.method_entry(manifest, "APD-MVS")
    run = sp.expand(spec)[0]

    declared = st.declared_normalizations(manifest, entry, spec, run)
    debug = [d for d in declared if d.get("reversible") == "flag --no-debug-output"]
    assert debug and all(d["in_force"] for d in debug)
    assert all("is in the executed argv" in d["in_force_reason"] for d in debug)


def test_debug_output_upstream_records_the_normalization_as_not_in_force(
    write_spec, spec_dict, manifest
):
    """R-EXP-11: the arm that measures the released behaviour has to say so."""
    spec_dict["methods"] = ["APD-MVS"]
    spec_dict["debug_output"] = "upstream"
    spec = sp.load(write_spec(spec_dict), manifest)
    entry = sp.method_entry(manifest, "APD-MVS")
    run = sp.expand(spec)[0]

    declared = st.declared_normalizations(manifest, entry, spec, run)
    debug = [d for d in declared if d.get("reversible") == "flag --no-debug-output"]
    assert debug and not any(d["in_force"] for d in debug)
    reason = debug[0]["in_force_reason"]
    assert "removed from this run's argv" in reason
    assert "debug_output: upstream" in reason

    record = _record(spec, manifest, run, _result())
    assert record["debug_output"]["setting"] == "upstream"
    assert record["debug_output"]["method_has_flag"] is True
    assert record["debug_output"]["normalization_in_force"] is False
    assert not any("--no-debug-output" in n for n in record["normalizations"])
    assert json.loads(record["parameters_json"])["debug_output"] == "upstream"


def test_a_method_with_no_debug_flag_says_the_setting_does_not_apply(
    write_spec, spec_dict, manifest
):
    spec_dict["methods"] = ["ACMM"]
    spec_dict["debug_output"] = "upstream"
    spec = sp.load(write_spec(spec_dict), manifest)
    run = sp.expand(spec)[0]
    record = _record(spec, manifest, run, _result())
    assert record["debug_output"]["method_has_flag"] is False
    assert record["debug_output"]["normalization_in_force"] is False
    assert "carries no --no-debug-output" in record["debug_output"]["reason"]


def test_the_record_flags_an_uninstrumented_run(write_spec, spec_dict, manifest):
    """R-TIM-09: the off half of the phase-timer pair is marked, not inferred."""
    spec_dict["methods"] = ["ACMM"]
    spec_dict["phase_timer"] = False
    spec = sp.load(write_spec(spec_dict), manifest)
    run = sp.expand(spec)[0]
    result = _result()
    result["phases"] = [
        {"name": "preprocess.convert", "total_s": 3.0, "self_s": 3.0, "count": 1,
         "unclosed": 0, "source": "harness"}
    ]
    result["phases_by_pass"] = []
    record = _record(spec, manifest, run, result)
    assert record["phase_timer"] is False
    assert record["instrumented"] is False
    assert record["method_env_json"] == "{}"
    assert not any(p["source"] == "method" for p in record["phases"])
    assert json.loads(record["parameters_json"])["phase_timer"] is False


def test_the_record_flags_mvs_bench_sync(write_spec, spec_dict, manifest):
    """R-TIM-08: the SYNC pair must be visible in the record, not only the spec."""
    spec_dict["methods"] = ["CUMVS"]
    spec_dict["method_env"] = {"MVS_BENCH_SYNC": "1"}
    spec = sp.load(write_spec(spec_dict), manifest)
    run = sp.expand(spec)[0]
    record = _record(spec, manifest, run, _result())
    assert record["mvs_bench_sync"] == "1"
    env = json.loads(record["method_env_json"])
    assert env["MVS_BENCH_SYNC"] == "1"
    assert env["MVS_BENCH_PHASES"] == "1"
    assert json.loads(record["parameters_json"])["method_env"] == {"MVS_BENCH_SYNC": "1"}

    spec_dict["method_env"] = {}
    off = sp.load(write_spec(spec_dict, "off.yaml"), manifest)
    off_record = _record(off, manifest, sp.expand(off)[0], _result())
    assert off_record["mvs_bench_sync"] is None


def test_the_record_carries_the_runs_on_disk_footprint(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    run = sp.expand(spec)[0]
    result = _result()
    result["intermediates_bytes"] = 4096
    assert _record(spec, manifest, run, result)["intermediates_bytes"] == 4096
    assert _record(spec, manifest, run, _result())["intermediates_bytes"] is None
