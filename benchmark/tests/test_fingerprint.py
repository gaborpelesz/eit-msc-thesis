import json

import pytest

from bench import fingerprint as fp

BASE = {
    "gpu_model": "NVIDIA GeForce RTX 5070",
    "driver_version": "580.00",
    "cuda_version": "13.0",
    "image_digest": "sha256:aaaa",
    "hostname": "fleet-01",
    "cpu_model": "some cpu",
}


def test_bind_writes_the_fingerprint_once(tmp_path):
    bound = fp.bind(tmp_path, BASE)
    assert bound == BASE
    written = json.loads((tmp_path / "fingerprint.json").read_text())
    assert written["gpu_model"] == BASE["gpu_model"]

    # A second machine that matches on every load-bearing field continues.
    same_class = dict(BASE, hostname="fleet-02", cpu_model="another cpu")
    assert fp.bind(tmp_path, same_class)["hostname"] == "fleet-01"


@pytest.mark.parametrize("field", fp.LOAD_BEARING)
def test_bind_hard_stops_on_a_load_bearing_mismatch(tmp_path, field):
    fp.bind(tmp_path, BASE)
    with pytest.raises(fp.FingerprintMismatch) as excinfo:
        fp.bind(tmp_path, dict(BASE, **{field: "different"}))
    assert field in str(excinfo.value)
    # The campaign's own fingerprint is left untouched.
    assert json.loads((tmp_path / "fingerprint.json").read_text())[field] == BASE[field]


def test_declared_fingerprint_is_checked_before_the_campaign_exists(tmp_path):
    declared = {"gpu_model": "NVIDIA GeForce RTX 5070", "driver_version": "999"}
    with pytest.raises(fp.FingerprintMismatch, match="driver_version"):
        fp.bind(tmp_path, BASE, declared)
    assert not (tmp_path / "fingerprint.json").exists()


def test_check_gpu_state_refuses_when_nvidia_smi_is_silent(monkeypatch):
    monkeypatch.setattr(fp, "nvidia_smi_query", lambda *a, **k: None)
    problems, _ = fp.check_gpu_state(0)
    assert problems and "cannot be verified" in problems[0]


def test_check_gpu_state_lists_every_unmet_precondition(monkeypatch):
    monkeypatch.setattr(
        fp,
        "nvidia_smi_query",
        lambda *a, **k: {
            "persistence_mode": "Disabled",
            "display_active": "Enabled",
            "display_mode": "Enabled",
        },
    )
    monkeypatch.setattr(fp, "locked_clocks_state", lambda gpu_index=0: (None, {}))
    problems, evidence = fp.check_gpu_state(0)
    joined = " ".join(problems)
    assert "persistence" in joined and "display" in joined and "clocks" in joined
    assert evidence["persistence_mode"] == "Disabled"


def test_check_gpu_state_passes_when_every_precondition_holds(monkeypatch):
    monkeypatch.setattr(
        fp,
        "nvidia_smi_query",
        lambda *a, **k: {
            "persistence_mode": "Enabled",
            "display_active": "Disabled",
            "display_mode": "Disabled",
        },
    )
    monkeypatch.setattr(
        fp, "locked_clocks_state", lambda gpu_index=0: (True, {"locked_clocks_mhz": [2000, 2000]})
    )
    problems, _ = fp.check_gpu_state(0)
    assert problems == []


def test_submodule_shas_is_a_list_of_records(monkeypatch, tmp_path):
    outputs = {
        "config": "submodule.a.path benchmark/methods/ACMH\nsubmodule.b.path resources/papers\n",
    }

    def fake_run(argv, timeout=60):
        if "config" in argv:
            return outputs["config"]
        if argv[-1] == "HEAD":
            return "abc123\n" if "ACMH" in argv[2] else None
        return None

    monkeypatch.setattr(fp, "_run", fake_run)
    shas = fp.submodule_shas(tmp_path)
    assert shas == [
        {"path": "benchmark/methods/ACMH", "sha": "abc123"},
        {"path": "resources/papers", "sha": None},
    ]
