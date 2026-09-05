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
    monkeypatch.setattr(
        cli.fp,
        "collect",
        lambda *a, **k: {
            "gpu_model": "fake",
            "driver_version": "1",
            "cuda_version": "12.8",
            "image_digest": "sha256:fake",
            "submodules": [],
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
