"""The runner's container path, exercised end to end against a fake docker."""

import json
import stat
from pathlib import Path

import pytest

from bench import runner as rn
from bench import spec as sp
from bench import store as st
from deviations import manifest as mf

FAKE_DOCKER = Path(__file__).parent / "fake_docker.py"
ROOT = mf.repo_root(mf.default_manifest_path().parent)


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    FAKE_DOCKER.chmod(FAKE_DOCKER.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(tmp_path / "docker-state"))
    monkeypatch.setenv("FAKE_DOCKER_OUTPUT_PLY", "ACMM/ACMM_model.ply")
    return str(FAKE_DOCKER)


@pytest.fixture
def prepared_spec(write_spec, spec_dict, manifest, tmp_path):
    spec_dict["methods"] = ["ACMM"]
    spec_dict["widths"] = [3200]
    spec_dict["repeats"] = 1
    spec = sp.load(write_spec(spec_dict), manifest)
    scene = spec.raw_scene_dir("courtyard", 3200)
    (scene / "images").mkdir(parents=True)
    (scene / "images" / "one.jpg").write_bytes(b"\xff" * 4096)
    spec.ground_truth_mlp("courtyard", 3200).parent.mkdir(parents=True, exist_ok=True)
    spec.ground_truth_mlp("courtyard", 3200).write_text("<MeshLabProject/>")
    spec.campaign_dir.mkdir(parents=True)
    return spec


def _run_once(spec, manifest, fake_docker):
    run = sp.expand(spec)[0]
    entry = sp.method_entry(manifest, run.method)
    tmp_dir, _ = rn.open_run_dir(spec.campaign_dir, run.key)
    result = rn.execute(spec, entry, run, tmp_dir, docker=fake_docker)
    result = rn.collect_point_cloud(spec, entry, run, result, tmp_dir)
    result = rn.run_evaluation(spec, run, result, tmp_dir, docker=fake_docker)
    result = rn.discard_intermediates(spec, result)
    record = st.build_record(
        spec,
        manifest,
        entry,
        run,
        result,
        {"gpu_model": "fake", "image_digest": "sha256:fake", "submodules": []},
        {"methods": [{"name": entry["name"], "fork_sha": "f" * 40}]},
        ROOT,
    )
    st.write_record(tmp_dir, record)
    return run, rn.commit_run_dir(tmp_dir), record


def test_successful_run_produces_a_complete_record(prepared_spec, manifest, fake_docker):
    run, final, record = _run_once(prepared_spec, manifest, fake_docker)

    assert record["status"] == "ok"
    assert record["run_key"] == run.key
    assert record["wall_time_s"] > 0
    assert record["preprocess_convert_s"] > 0
    assert record["preprocess_argv"][-4:] == [
        "--dense_folder", "/data", "--save_folder", "/work/prepared"
    ]
    assert record["container_wall_time_s"] == pytest.approx(12.1, abs=0.01)
    assert record["f1_primary"] == 0.55
    assert len(record["quality"]) == 5
    assert record["point_count"] == 3
    assert record["point_cloud_sha256"]
    assert Path(record["point_cloud_path"]).exists()
    assert record["neighbour_list"]["sha256"]
    assert record["warm_cache_bytes"] == 4096
    assert record["sample_interval_ms"] == 100.0

    phases = {p["name"]: p for p in record["phases"]}
    assert phases["patchmatch"]["total_s"] == 0.8
    assert phases["preprocess.convert"]["source"] == "harness"
    assert phases["image_pass"]["source"] == "method"
    assert record["phases_by_pass"][0]["pass"] == "photometric"

    for name in ("run.json", "phases.txt", "phases.harness.txt", "stdout.log",
                 "stderr.log", "telemetry.parquet", "eval.stdout.log"):
        assert (final / name).exists(), name
    assert "fake method stdout" in (final / "stdout.log").read_text()
    # R-ART-02: the scratch tree is gone, the store and the cloud are not.
    assert not Path(record["work_dir"]).exists()


def test_failure_is_recorded_as_a_result(prepared_spec, manifest, fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_EXIT", "1")
    monkeypatch.setenv("FAKE_DOCKER_STDERR", "CUDA error: out of memory\n")
    monkeypatch.setenv("FAKE_DOCKER_OUTPUT_PLY", "")

    _, final, record = _run_once(prepared_spec, manifest, fake_docker)
    assert record["status"] == "oom_gpu"
    assert record["exit_code"] == 1
    assert record["quality"] == []
    assert record["point_cloud_path"] is None
    evidence = json.loads(record["status_evidence_json"])
    assert "cuda error: out of memory" in evidence["log_markers"]
    assert (final / "run.json").exists()


def test_successful_exit_without_a_cloud_is_no_output(
    prepared_spec, manifest, fake_docker, monkeypatch
):
    monkeypatch.setenv("FAKE_DOCKER_OUTPUT_PLY", "")
    _, _, record = _run_once(prepared_spec, manifest, fake_docker)
    assert record["status"] == "no_output"
    assert record["f1_primary"] is None


def test_truncated_phase_trace_does_not_break_the_record(
    prepared_spec, manifest, fake_docker, monkeypatch
):
    monkeypatch.setenv("FAKE_DOCKER_NO_PHASES", "1")
    _, _, record = _run_once(prepared_spec, manifest, fake_docker)
    assert record["status"] == "ok"
    assert [p["name"] for p in record["phases"]] == [
        "preprocess.convert",
        "preprocess.warm_cache",
    ]
    assert record["phase_trace_errors"]


def test_stale_scratch_is_moved_aside_not_reused(
    prepared_spec, manifest, fake_docker
):
    run = sp.expand(prepared_spec)[0]
    stale = prepared_spec.work_dir(run) / "prepared" / "ACMM"
    stale.mkdir(parents=True)
    (stale / "ACMM_model.ply").write_text("a stale cloud from an earlier attempt")

    _, _, record = _run_once(prepared_spec, manifest, fake_docker)
    assert record["point_count"] == 3
    assert any("stale scratch moved aside" in note for note in record["notes"])
    aside = list(Path(prepared_spec.paths["work_root"], "unit-test").glob("*.stale"))
    assert aside and (aside[0] / "prepared" / "ACMM" / "ACMM_model.ply").exists()


def test_keep_intermediates_leaves_the_scratch_tree(prepared_spec, manifest, fake_docker):
    prepared_spec.keep_intermediates = True
    _, _, record = _run_once(prepared_spec, manifest, fake_docker)
    assert Path(record["work_dir"]).exists()
    assert any("intermediates kept" in note for note in record["notes"])


def test_a_failed_converter_stops_the_run_before_the_measured_stage(
    prepared_spec, manifest, fake_docker, monkeypatch, tmp_path
):
    """HPM-MVS, 2026-09-06: `np.asscalar` killed the fork's own converter and
    the method binary then ran on the half-written dataset, exited 0 and wrote
    a header-only PLY. The measured stage must not start at all."""
    monkeypatch.setenv("FAKE_DOCKER_CONVERT_EXIT", "1")
    _, final, record = _run_once(prepared_spec, manifest, fake_docker)

    assert record["status"] == "nonzero_exit"
    evidence = json.loads(record["status_evidence_json"])
    assert evidence["stage"] == "preprocess.convert"
    assert "fake converter died" in evidence["stderr_tail"]
    assert record["wall_time_s"] is None
    assert record["command"] == []
    assert record["point_cloud_path"] is None
    # No measured container was created, so no container state was written.
    assert not (tmp_path / "docker-state" / "state.json").exists()
    assert not (final / "stdout.log").exists()


def test_a_header_only_point_cloud_is_no_output_not_ok(
    prepared_spec, manifest, fake_docker, monkeypatch
):
    monkeypatch.setenv("FAKE_DOCKER_EMPTY_PLY", "1")
    _, _, record = _run_once(prepared_spec, manifest, fake_docker)

    assert record["status"] == "no_output"
    assert record["point_count"] == 0
    # The cloud is kept as evidence; it is its emptiness that is the result.
    assert Path(record["point_cloud_path"]).exists()
    assert json.loads(record["status_evidence_json"])["empty_output"]["point_count"] == 0


def test_the_point_cloud_is_copied_out_of_a_directory_we_cannot_write(
    prepared_spec, manifest, tmp_path
):
    """The measured container runs as root, so the directory it writes the
    cloud into is root-owned: the harness may read it but not unlink out of
    it. Reproduced here with a read-only directory."""
    run = sp.expand(prepared_spec)[0]
    entry = sp.method_entry(manifest, run.method)
    produced = prepared_spec.work_dir(run) / "prepared" / entry["output_ply"]
    produced.parent.mkdir(parents=True)
    produced.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n1 2 3\n")
    produced.parent.chmod(0o555)
    try:
        result = rn.collect_point_cloud(
            prepared_spec,
            entry,
            run,
            {"status": "ok", "work_dir": str(prepared_spec.work_dir(run))},
            tmp_path,
        )
        assert result["status"] == "ok"
        assert result["point_cloud"]["point_count"] == 1
        assert Path(result["point_cloud"]["path"]).exists()
        assert produced.exists()
    finally:
        produced.parent.chmod(0o755)


def test_root_owned_intermediates_are_reclaimed_before_removal(
    prepared_spec, manifest, monkeypatch
):
    import subprocess

    run = sp.expand(prepared_spec)[0]
    work = prepared_spec.work_dir(run)
    locked = work / "prepared" / "ACMM"
    locked.mkdir(parents=True)
    (locked / "depths.dmb").write_bytes(b"\x00" * 16)
    locked.chmod(0o555)
    calls = []

    def fake_reclaim(spec, work_dir, docker="docker"):
        calls.append(str(work_dir))
        locked.chmod(0o755)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(rn, "reclaim_ownership", fake_reclaim)
    try:
        result = rn.discard_intermediates(prepared_spec, {"work_dir": str(work)})
    finally:
        if locked.exists():
            locked.chmod(0o755)
    assert calls == [str(work)]
    assert not work.exists()
    assert any("chowned back" in note for note in result["notes"])


def test_an_interrupt_does_not_leave_the_container_holding_the_gpu(
    prepared_spec, manifest, fake_docker, monkeypatch, tmp_path
):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(rn, "_container_pid", interrupt)
    run = sp.expand(prepared_spec)[0]
    entry = sp.method_entry(manifest, run.method)
    tmp_dir, _ = rn.open_run_dir(prepared_spec.campaign_dir, run.key)
    with pytest.raises(KeyboardInterrupt):
        rn.execute(prepared_spec, entry, run, tmp_dir, docker=fake_docker)

    calls = (tmp_path / "docker-state" / "calls.log").read_text().splitlines()
    assert any(call.startswith("rm -f 0123456789abcdef") for call in calls)


def test_an_uninstrumented_run_writes_no_trace_and_is_still_ok(
    write_spec, spec_dict, manifest, tmp_path, fake_docker
):
    """R-TIM-09: the off half of the timer pair is a normal, complete run."""
    spec_dict["methods"] = ["ACMM"]
    spec_dict["widths"] = [3200]
    spec_dict["repeats"] = 1
    spec_dict["phase_timer"] = False
    spec = sp.load(write_spec(spec_dict, "notimer.yaml"), manifest)
    scene = spec.raw_scene_dir("courtyard", 3200)
    (scene / "images").mkdir(parents=True)
    (scene / "images" / "one.jpg").write_bytes(b"\xff" * 4096)
    spec.ground_truth_mlp("courtyard", 3200).parent.mkdir(parents=True, exist_ok=True)
    spec.ground_truth_mlp("courtyard", 3200).write_text("<MeshLabProject/>")
    spec.campaign_dir.mkdir(parents=True)

    _, final, record = _run_once(spec, manifest, fake_docker)

    assert record["status"] == "ok"
    assert record["instrumented"] is False
    assert not (final / "phases.txt").exists()
    assert not any(p["source"] == "method" for p in record["phases"])
    # The harness's own spans are unaffected: preprocessing is still timed.
    assert record["preprocess_convert_s"] > 0
    assert not any("MVS_BENCH" in token for token in record["command"])


def test_debug_output_upstream_reaches_the_measured_container(
    write_spec, spec_dict, manifest, tmp_path, fake_docker, monkeypatch
):
    monkeypatch.setenv("FAKE_DOCKER_OUTPUT_PLY", "APD/APD.ply")
    spec_dict["methods"] = ["APD-MVS"]
    spec_dict["widths"] = [3200]
    spec_dict["repeats"] = 1
    spec_dict["debug_output"] = "upstream"
    spec = sp.load(write_spec(spec_dict, "upstream.yaml"), manifest)
    scene = spec.raw_scene_dir("courtyard", 3200)
    (scene / "images").mkdir(parents=True)
    (scene / "images" / "one.jpg").write_bytes(b"\xff" * 4096)
    spec.ground_truth_mlp("courtyard", 3200).parent.mkdir(parents=True, exist_ok=True)
    spec.ground_truth_mlp("courtyard", 3200).write_text("<MeshLabProject/>")
    spec.campaign_dir.mkdir(parents=True)

    _, _, record = _run_once(spec, manifest, fake_docker)
    assert record["status"] == "ok"
    assert "--no-debug-output" not in record["command"]
    assert record["debug_output"]["normalization_in_force"] is False
    # R-ART-02 measures the footprint before the scratch tree goes, which is
    # what the debug arm is compared on.
    assert record["intermediates_bytes"] > 0
