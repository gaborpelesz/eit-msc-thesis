import re

import pytest

from bench import spec as sp


def test_run_key_format(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    runs = sp.expand(spec)
    pattern = re.compile(r"^[\w.+-]+__[\w-]+__w\d+__[\w-]+__r\d+$")
    for run in runs:
        assert pattern.match(run.key), run.key
    assert "ACMH__courtyard__w1600__author__r1" in {r.key for r in runs}


def test_cartesian_product_and_exclusions(write_spec, spec_dict, manifest):
    spec = sp.load(write_spec(spec_dict), manifest)
    assert len(sp.expand(spec)) == 2 * 1 * 2 * 1 * 2

    spec_dict["exclude"] = [{"method": "ACMH", "width": 3200}]
    spec = sp.load(write_spec(spec_dict, "excl.yaml"), manifest)
    runs = sp.expand(spec)
    assert len(runs) == 6
    assert not any(r.method == "ACMH" and r.width == 3200 for r in runs)


def test_unknown_exclusion_field_is_refused(write_spec, spec_dict, manifest):
    spec_dict["exclude"] = [{"methods": "ACMH"}]
    spec = sp.load(write_spec(spec_dict), manifest)
    with pytest.raises(sp.SpecError, match="unknown field"):
        sp.expand(spec)


def test_order_is_randomised_but_reproducible(write_spec, spec_dict, manifest):
    spec_dict["repeats"] = 3
    spec = sp.load(write_spec(spec_dict), manifest)
    first = [r.key for r in sp.expand(spec)]
    second = [r.key for r in sp.expand(spec)]
    assert first == second

    ordered = sorted(first)
    assert first != ordered, "the run list must not be in Cartesian order (R-STA-01)"

    spec_dict["order_seed"] = 8
    other = [r.key for r in sp.expand(sp.load(write_spec(spec_dict, "b.yaml"), manifest))]
    assert sorted(other) == ordered
    assert other != first


def test_shards_are_disjoint_and_complete(write_spec, spec_dict, manifest):
    spec_dict["repeats"] = 3
    keys = {r.key for r in sp.expand(sp.load(write_spec(spec_dict), manifest))}
    collected = []
    for index in range(3):
        spec_dict["shard"] = {"index": index, "count": 3}
        shard = sp.load(write_spec(spec_dict, f"s{index}.yaml"), manifest)
        collected.append({r.key for r in sp.expand(shard)})
    assert set().union(*collected) == keys
    assert sum(len(s) for s in collected) == len(keys)


def test_excluded_method_is_refused(write_spec, spec_dict, manifest):
    spec_dict["methods"] = ["ACMH", "DVP-MVS"]
    with pytest.raises(sp.SpecError, match="excluded from the campaign"):
        sp.load(write_spec(spec_dict), manifest)


def test_unknown_method_and_configuration_are_refused(write_spec, spec_dict, manifest):
    spec_dict["methods"] = ["NOT-A-METHOD"]
    with pytest.raises(sp.SpecError, match="not in methods.yaml"):
        sp.load(write_spec(spec_dict), manifest)

    spec_dict["methods"] = ["ACMH"]
    spec_dict["configurations"] = ["nope"]
    with pytest.raises(sp.SpecError, match="not defined in methods.yaml"):
        sp.load(write_spec(spec_dict, "cfg.yaml"), manifest)


def test_schema_version_is_enforced(write_spec, spec_dict, manifest):
    spec_dict["version"] = 99
    with pytest.raises(sp.SpecError, match="not implemented"):
        sp.load(write_spec(spec_dict), manifest)


def test_padding_must_be_decided_for_the_whole_batch(write_spec, spec_dict, manifest):
    spec_dict["methods"] = ["ACMH", "APD-MVS"]
    spec_dict["configurations"] = ["author", "norm10"]
    with pytest.raises(sp.SpecError, match="R-EXP-09"):
        sp.load(write_spec(spec_dict), manifest)

    spec_dict["padding"] = "all"
    spec = sp.load(write_spec(spec_dict, "pad.yaml"), manifest)
    assert spec.padding == "all"


def test_spec_hash_and_timeout_default(write_spec, spec_dict, manifest):
    del spec_dict["timeout_seconds"]
    path = write_spec(spec_dict)
    spec = sp.load(path, manifest)
    assert spec.timeout_s == sp.DEFAULT_TIMEOUT_S
    assert spec.sha256 == sp.load(path, manifest).sha256
    assert len(spec.sha256) == 64


def _write_calibration(raw_scene_dir, cameras, image_camera_ids):
    cal = raw_scene_dir / "dslr_calibration_undistorted"
    cal.mkdir(parents=True)
    cal.joinpath("cameras.txt").write_text(
        "# cams\n" + "".join(f"{i} PINHOLE {w} {h} 1 1 1 1\n" for i, (w, h) in cameras.items())
    )
    rows = "".join(f"{n} 1 0 0 0 0 0 0 {cid} img{n}.JPG\n\n" for n, cid in enumerate(image_camera_ids))
    cal.joinpath("images.txt").write_text("# images\n" + rows)


def test_equal_size_check_reads_the_scene_on_disk(write_spec, spec_dict, manifest, tmp_path):
    spec_dict["methods"] = ["ACMH", "APD-MVS"]
    spec_dict["configurations"] = ["author", "norm10"]
    spec_dict["widths"] = [3200]
    # Uniform scene: no assertion needed, loads.
    uniform = tmp_path / "datasets" / "courtyard_3200" / "courtyard_dslr_undistorted" / "courtyard"
    _write_calibration(uniform, {"0": (6208, 4135)}, ["0"] * 4)
    assert sp.scene_image_sizes(uniform) == {(6208, 4135): 4}
    sp.load(write_spec(spec_dict), manifest)

    # Mixed scene: refused even if the operator asserts otherwise.
    spec_dict["scenes"] = ["electro"]
    spec_dict["scenes_have_equal_image_sizes"] = True
    mixed = tmp_path / "datasets" / "electro_3200" / "electro_dslr_undistorted" / "electro"
    _write_calibration(mixed, {"0": (6205, 4134), "1": (6203, 4134)}, ["0", "1", "0"])
    with pytest.raises(sp.SpecError, match="2 image sizes"):
        sp.load(write_spec(spec_dict, "mixed.yaml"), manifest)
