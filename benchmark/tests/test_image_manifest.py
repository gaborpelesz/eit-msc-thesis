"""D24.1: the image build manifest and the gate that compares it."""

import json

import pytest

from bench import cli
from bench import fingerprint as fp
from bench import image_manifest as imf
from deviations import manifest as mf

ROOT = mf.repo_root(mf.default_manifest_path().parent)


@pytest.fixture(scope="module")
def document(manifest):
    return imf.compute(ROOT, manifest)


def test_the_manifest_describes_every_method_in_the_yaml(document, manifest):
    names = [m["name"] for m in document["methods"]]
    assert names == sorted(m["name"] for m in manifest["methods"])
    excluded = [m for m in document["methods"] if m["status"] == "excluded"]
    assert [m["name"] for m in excluded] == ["DVP-MVS"]
    for method in document["methods"]:
        assert len(method["fork_sha"]) == 40
        assert method["dirty"] in (True, False)
        assert len(method["converter"]["sha256"]) == 64
    cumvs = next(m for m in document["methods"] if m["name"] == "CUMVS")
    # CUMVS has no converter script; the image builds its initializer from this
    # source, which is the only thing the host can hash.
    assert cumvs["converter"]["kind"] == "initializer_source"
    assert cumvs["converter"]["path"].endswith("samples/app_initialize_ETH3D.cpp")


def test_the_manifest_hashes_the_build_inputs(document):
    for key in ("dockerfile", "pyproject", "uv_lock", "shared_converter"):
        assert len(document[key]["sha256"]) == 64
    assert document["dockerfile"]["path"] == "benchmark/Dockerfile"
    assert document["shared_converter"]["path"].endswith("colmap2mvsnet_acm_perf.py")
    assert len(document["evaluator"]["sha"]) == 40


def test_generation_is_deterministic_apart_from_the_timestamp(manifest, document):
    again = imf.compute(ROOT, manifest)
    assert imf.canonical(again) == imf.canonical(document)
    assert imf.digest(again) == imf.digest(document)
    assert imf.compare(again, document) == []
    # Serialisation is stable too: the file is committed next to the Dockerfile.
    assert json.dumps(imf.canonical(again), sort_keys=True) == json.dumps(
        imf.canonical(document), sort_keys=True
    )


def test_a_later_generation_time_is_not_a_difference(document):
    later = dict(document, generated_at="2099-01-01T00:00:00+00:00")
    assert imf.compare(document, later) == []
    assert imf.digest(later) == imf.digest(document)


def test_compare_reports_a_changed_fork_sha(document):
    stale = json.loads(json.dumps(document))
    stale["methods"][2]["fork_sha"] = "0" * 40
    differences = imf.compare(stale, document)
    assert [d["key"] for d in differences] == [
        f"methods[{document['methods'][2]['name']}].fork_sha"
    ]
    assert differences[0]["image"] == "0" * 40
    assert differences[0]["working"] == document["methods"][2]["fork_sha"]


def test_compare_reports_a_changed_converter_and_dockerfile_hash(document):
    stale = json.loads(json.dumps(document))
    stale["methods"][0]["converter"]["sha256"] = "1" * 64
    stale["dockerfile"]["sha256"] = "2" * 64
    keys = [d["key"] for d in imf.compare(stale, document)]
    assert keys == [
        "dockerfile.sha256",
        f"methods[{document['methods'][0]['name']}].converter.sha256",
    ]


def test_compare_reports_a_method_the_image_does_not_have(document):
    stale = json.loads(json.dumps(document))
    dropped = stale["methods"].pop()
    keys = [d["key"] for d in imf.compare(stale, document)]
    assert keys and all(k.startswith(f"methods[{dropped['name']}]") for k in keys)


def test_write_and_read_back(tmp_path, document):
    path = tmp_path / "image-manifest.json"
    imf.write(path, document)
    assert imf.compare(json.loads(path.read_text()), document) == []


def test_read_image_manifest_from_the_image(monkeypatch, document):
    calls = []

    def fake_run(argv, timeout=60):
        calls.append(argv)
        return json.dumps(document)

    monkeypatch.setattr(fp, "_run", fake_run)
    assert fp.read_image_manifest("img:tag") == document
    assert calls[0][:5] == ["docker", "run", "--rm", "--entrypoint", "cat"]
    assert calls[0][-1] == "/sota/image-manifest.json"


@pytest.mark.parametrize("output", [None, "", "not json"])
def test_an_image_without_a_readable_manifest_reads_as_none(monkeypatch, output):
    monkeypatch.setattr(fp, "_run", lambda argv, timeout=60: output)
    assert fp.read_image_manifest("img:tag") is None


def test_collect_records_the_manifest_and_its_digest(monkeypatch, document):
    monkeypatch.setattr(fp, "nvidia_smi_query", lambda *a, **k: {})
    monkeypatch.setattr(fp, "driver_cuda_version", lambda: "12.8")
    monkeypatch.setattr(fp, "image_identity", lambda image, docker="docker": {})
    monkeypatch.setattr(fp, "read_image_manifest", lambda *a, **k: document)
    collected = fp.collect("img:tag", probe_image=True)
    assert collected["image_manifest"] == document
    assert collected["image_manifest_sha256"] == imf.digest(document)

    monkeypatch.setattr(fp, "read_image_manifest", lambda *a, **k: None)
    unprobed = fp.collect("img:tag", probe_image=False)
    assert unprobed["image_manifest"] is None
    assert unprobed["image_manifest_sha256"] is None


def test_image_manifest_command_writes_the_file(tmp_path, capsys):
    out = tmp_path / "image-manifest.json"
    assert cli.main(["image-manifest", "--out", str(out)]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out) == json.loads(out.read_text())
    assert f"wrote {out}" in printed.err


def test_dry_run_needs_no_image(write_spec, spec_dict, monkeypatch, capsys):
    """A dry run produces no measurement, so it probes nothing (D24.1)."""

    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not touch the image")

    monkeypatch.setattr(cli.fp, "collect", refuse)
    monkeypatch.setattr(cli.fp, "read_image_manifest", refuse)
    assert cli.main(["run", str(write_spec(spec_dict)), "--dry-run"]) == 0
    assert "nothing was executed" in capsys.readouterr().out
