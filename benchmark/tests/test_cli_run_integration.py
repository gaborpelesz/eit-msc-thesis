"""`bench run` end to end through the CLI, against a fake docker.

`test_execute_integration` drives `runner.execute` directly, so nothing there
exercises the wiring in `cmd_run` -- the campaign gate, the skip/rerun
decision, the record it assembles. The first real campaign died in that
wiring, so it has a test of its own.
"""

import json
import stat
from pathlib import Path

import pytest

from bench import cli
from bench import spec as sp
from deviations import manifest as mf

ROOT = mf.repo_root(mf.default_manifest_path().parent)

FAKE_DOCKER = Path(__file__).parent / "fake_docker.py"

GPU_EVIDENCE = {
    "persistence_mode": "Enabled",
    "display_active": "Disabled",
    "clocks": {
        "locked_clocks_method": "idle-clock-inference",
        "locked_clocks_inferred_mhz": 1800,
    },
}


@pytest.fixture
def campaign(write_spec, spec_dict, manifest, monkeypatch, tmp_path):
    FAKE_DOCKER.chmod(FAKE_DOCKER.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(tmp_path / "docker-state"))
    monkeypatch.setenv("FAKE_DOCKER_OUTPUT_PLY", "ACMM/ACMM_model.ply")

    spec_dict["methods"] = ["ACMM"]
    spec_dict["widths"] = [3200]
    spec_dict["repeats"] = 1
    path = write_spec(spec_dict)
    spec = sp.load(path, manifest)
    scene = spec.raw_scene_dir("courtyard", 3200)
    (scene / "images").mkdir(parents=True)
    (scene / "images" / "one.jpg").write_bytes(b"\xff" * 4096)
    spec.ground_truth_mlp("courtyard", 3200).parent.mkdir(parents=True, exist_ok=True)
    spec.ground_truth_mlp("courtyard", 3200).write_text("<MeshLabProject/>")

    # The two gates and the fingerprint probe are the only parts of `cmd_run`
    # that need the real machine; everything between them is under test.
    monkeypatch.setattr(cli.vf, "verify", lambda manifest_path: 0)
    monkeypatch.setattr(cli.fp, "check_gpu_state", lambda gpu_index=0: ([], GPU_EVIDENCE))
    # The image the campaign would run is stood in for by the working tree's
    # own build manifest, so the D24.1 gate sees an image built from it.
    image_manifest = cli.imf.compute(ROOT, manifest)
    monkeypatch.setattr(
        cli.fp,
        "collect",
        lambda *a, **k: {
            "gpu_model": "fake",
            "driver_version": "1",
            "cuda_version": "12.8",
            "image_digest": "sha256:fake",
            "image_id": "sha256:fake",
            "submodules": [],
            "image_manifest": image_manifest,
            "image_manifest_sha256": cli.imf.digest(image_manifest),
        },
    )
    return spec


def _record(spec, key):
    return json.loads((spec.campaign_dir / key / "run.json").read_text())


def test_bench_run_executes_a_campaign_and_records_it(campaign, capsys):
    key = sp.expand(campaign)[0].key
    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 0
    out = capsys.readouterr().out
    assert "1 runs executed, 0 not ok" in out

    record = _record(campaign, key)
    assert record["status"] == "ok"
    assert record["f1_primary"] == 0.55
    assert (campaign.campaign_dir / key / "telemetry.parquet").exists()
    assert (campaign.campaign_dir / "fingerprint.json").exists()
    assert (campaign.campaign_dir / "spec.yaml").exists()


def test_bench_run_carries_the_clock_gate_into_the_record(campaign):
    key = sp.expand(campaign)[0].key
    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 0
    record = _record(campaign, key)
    assert record["clock_hold"]["expected_mhz"] == 1800
    assert record["clock_hold"]["method"] == "idle-clock-inference"
    assert json.loads(record["gpu_state_evidence_json"])["persistence_mode"] == "Enabled"


def test_bench_run_skips_a_finished_run_and_reruns_only_on_request(campaign, capsys):
    key = sp.expand(campaign)[0].key
    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 0
    capsys.readouterr()

    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 0
    assert f"skip     {key} (finished)" in capsys.readouterr().out

    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER), "--rerun"]) == 0
    superseded = list(campaign.campaign_dir.glob(f"{key}.*.superseded"))
    assert len(superseded) == 1
    assert (superseded[0] / "run.json").exists()
    assert _record(campaign, key)["status"] == "ok"


def test_bench_run_refuses_an_image_whose_manifest_differs(campaign, monkeypatch, capsys):
    """D24.1: a stale image would make every record name code that did not run."""
    stale = cli.imf.compute(ROOT, mf.load(mf.default_manifest_path()))
    stale["methods"][0]["fork_sha"] = "0" * 40
    stale["dockerfile"]["sha256"] = "1" * 64
    fingerprint = dict(
        cli.fp.collect(), image_manifest=stale, image_manifest_sha256=cli.imf.digest(stale)
    )
    monkeypatch.setattr(cli.fp, "collect", lambda *a, **k: fingerprint)

    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 1
    err = capsys.readouterr().err
    assert "was not built from this working tree" in err
    assert "dockerfile.sha256" in err
    assert f"methods[{stale['methods'][0]['name']}].fork_sha" in err
    # Nothing was executed, so the campaign directory was never created.
    assert not campaign.campaign_dir.exists()


def test_bench_run_refuses_an_image_without_a_manifest(campaign, monkeypatch, capsys):
    fingerprint = dict(cli.fp.collect(), image_manifest=None, image_manifest_sha256=None)
    monkeypatch.setattr(cli.fp, "collect", lambda *a, **k: fingerprint)

    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 1
    assert "carries no /sota/image-manifest.json" in capsys.readouterr().err


def test_bench_run_reports_the_image_and_the_manifest_check(campaign, capsys):
    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 0
    out = capsys.readouterr().out
    assert "image    sota-deps:latest id=sha256:fake" in out
    assert "manifest ok" in out


def test_the_record_references_the_image_manifest(campaign):
    key = sp.expand(campaign)[0].key
    assert cli.main(["run", str(campaign.path), "--docker", str(FAKE_DOCKER)]) == 0
    record = _record(campaign, key)
    assert record["image_manifest_sha256"] == cli.imf.digest(
        record["fingerprint"]["image_manifest"]
    )
