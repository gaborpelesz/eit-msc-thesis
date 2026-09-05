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


# --- R-ENV-02, third detection route: inference from the idle clock ---------


def _samples(sm, count=10, utilization=0, max_sm=2175):
    values = sm if isinstance(sm, list) else [sm] * count
    return [
        {"sm_mhz": v, "gr_mhz": v, "max_sm_mhz": max_sm, "utilization_pct": utilization}
        for v in values
    ]


def test_inference_accepts_a_clock_held_above_the_idle_floor():
    locked, evidence = fp.infer_locked_clocks(_samples(1800), floor_mhz=300)
    assert locked is True
    assert evidence["sm_mhz"] == [1800]


def test_inference_refuses_a_varying_clock():
    locked, evidence = fp.infer_locked_clocks(
        _samples([1800] * 9 + [1755]), floor_mhz=300
    )
    assert locked is None
    assert "varied" in evidence["verdict"]


def test_inference_refuses_a_clock_sitting_on_the_idle_floor():
    locked, evidence = fp.infer_locked_clocks(_samples(300), floor_mhz=300)
    assert locked is None
    assert "idle floor" in evidence["verdict"]


def test_inference_refuses_a_busy_gpu():
    locked, evidence = fp.infer_locked_clocks(_samples(1800, utilization=80), floor_mhz=300)
    assert locked is None
    assert "busy" in evidence["verdict"]


def test_inference_refuses_when_the_floor_is_unreadable():
    locked, evidence = fp.infer_locked_clocks(_samples(1800), floor_mhz=None)
    assert locked is None
    assert "idle floor" in evidence["verdict"]


def test_inference_refuses_too_few_samples():
    locked, evidence = fp.infer_locked_clocks(_samples(1800, count=2), floor_mhz=300)
    assert locked is None
    assert "three" in evidence["verdict"]


def test_locked_clocks_state_falls_back_to_the_inference(monkeypatch):
    """Driver 575 answers neither NVML nor applications clocks; the third
    route is what makes the RTX 2080 Ti verifiable."""
    monkeypatch.setattr(fp, "nvidia_smi_query", lambda *a, **k: {
        "clocks.applications.graphics": "N/A",
        "clocks.applications.memory": "N/A",
    })
    locked, evidence = fp.locked_clocks_state(
        0, sample=lambda gpu_index: _samples(1800), floor=lambda gpu_index: [300, 1800, 2175]
    )
    assert locked is True
    assert evidence["locked_clocks_method"] == "idle-clock-inference"
    assert evidence["locked_clocks_inferred_mhz"] == 1800


def test_locked_clocks_state_stays_unknown_when_the_inference_refuses(monkeypatch):
    monkeypatch.setattr(fp, "nvidia_smi_query", lambda *a, **k: {
        "clocks.applications.graphics": "N/A",
    })
    locked, evidence = fp.locked_clocks_state(
        0, sample=lambda gpu_index: _samples(300), floor=lambda gpu_index: [300, 2175]
    )
    assert locked is None
    assert "locked_clocks_method" not in evidence


def test_supported_graphics_clocks_parses_the_nvidia_smi_report(monkeypatch):
    monkeypatch.setattr(fp, "_run", lambda argv, timeout=60: (
        "    Supported Clocks\n"
        "        Memory                            : 7000 MHz\n"
        "            Graphics                      : 2175 MHz\n"
        "            Graphics                      : 1800 MHz\n"
        "            Graphics                      : 300 MHz\n"
    ))
    assert fp.supported_graphics_clocks(0) == [300, 1800, 2175]


# --- R-ENV-02 after the fact: did the clock hold for the whole run? --------


def _telemetry_table(values):
    import pyarrow as pa

    return pa.table({"gpu_sm_clock_mhz": pa.array(values, type=pa.int32())})


def test_clock_hold_is_true_when_every_sample_sits_at_the_expected_clock():
    hold = fp.clock_hold_from_table(_telemetry_table([1800] * 50), 1800, method="x")
    assert hold == {
        "expected_mhz": 1800,
        "min_mhz": 1800,
        "max_mhz": 1800,
        "samples": 50,
        "held": True,
        "method": "x",
    }


def test_clock_hold_is_false_when_the_clock_left_the_lock():
    hold = fp.clock_hold_from_table(_telemetry_table([1800] * 40 + [1500] * 10), 1800)
    assert hold["held"] is False
    assert (hold["min_mhz"], hold["max_mhz"]) == (1500, 1800)


def test_clock_hold_is_unverified_without_an_expected_clock():
    hold = fp.clock_hold_from_table(_telemetry_table([1800, 1800]), None)
    assert hold["held"] is None
    assert hold["min_mhz"] == 1800


def test_clock_hold_tolerates_a_telemetry_table_without_the_column():
    import pyarrow as pa

    hold = fp.clock_hold_from_table(pa.table({"elapsed_s": pa.array([0.1])}), 1800)
    assert hold == {
        "expected_mhz": 1800,
        "min_mhz": None,
        "max_mhz": None,
        "samples": 0,
        "held": None,
        "method": None,
    }


def test_expected_clock_mhz_prefers_the_inferred_value_then_the_locked_range():
    assert fp.expected_clock_mhz({"clocks": {"locked_clocks_inferred_mhz": 1800}}) == 1800
    assert fp.expected_clock_mhz({"clocks": {"locked_clocks_mhz": [1500, 1500]}}) == 1500
    # A range is a proof of a lock but not of one value, so no verdict is invented.
    assert fp.expected_clock_mhz({"clocks": {"locked_clocks_mhz": [300, 2175]}}) is None
    assert fp.expected_clock_mhz({}) is None
