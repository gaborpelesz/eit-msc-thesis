import json

import pytest

from bench import runner as rn

KEY = "ACMM__courtyard__w3200__author__r1"


def test_tmp_directory_is_created_and_committed(tmp_path):
    tmp_dir, moved = rn.open_run_dir(tmp_path, KEY)
    assert tmp_dir.name == f"{KEY}.tmp"
    assert moved == []
    (tmp_dir / "run.json").write_text("{}")
    final = rn.commit_run_dir(tmp_dir)
    assert final.name == KEY
    assert not tmp_dir.exists()
    assert rn.is_finished(tmp_path, KEY)


def test_commit_refuses_without_run_json(tmp_path):
    tmp_dir, _ = rn.open_run_dir(tmp_path, KEY)
    with pytest.raises(rn.RunnerError, match="without run.json"):
        rn.commit_run_dir(tmp_dir)


def test_leftover_tmp_is_moved_aside_never_deleted(tmp_path):
    tmp_dir, _ = rn.open_run_dir(tmp_path, KEY)
    (tmp_dir / "stdout.log").write_text("partial output")

    tmp_dir2, moved = rn.open_run_dir(tmp_path, KEY)
    assert len(moved) == 1
    aside = tmp_path / moved[0]
    assert aside.exists() and aside.name.endswith(".aborted")
    assert (aside / "stdout.log").read_text() == "partial output"
    assert tmp_dir2.exists()


def test_finished_run_is_not_overwritten(tmp_path):
    tmp_dir, _ = rn.open_run_dir(tmp_path, KEY)
    (tmp_dir / "run.json").write_text(json.dumps({"run_key": KEY}))
    rn.commit_run_dir(tmp_dir)

    with pytest.raises(rn.RunnerError, match="already finished"):
        rn.open_run_dir(tmp_path, KEY)


def test_rerun_supersedes_without_deleting(tmp_path):
    tmp_dir, _ = rn.open_run_dir(tmp_path, KEY)
    (tmp_dir / "run.json").write_text(json.dumps({"run_key": KEY, "status": "ok"}))
    rn.commit_run_dir(tmp_dir)

    tmp_dir2, moved = rn.open_run_dir(tmp_path, KEY, rerun=True)
    assert any(m.endswith(".superseded") for m in moved)
    kept = json.loads((tmp_path / moved[0] / "run.json").read_text())
    assert kept["status"] == "ok"

    (tmp_dir2 / "run.json").write_text(json.dumps({"run_key": KEY, "status": "timeout"}))
    rn.commit_run_dir(tmp_dir2)
    # both attempts survive on disk
    assert len(list(tmp_path.glob(f"{KEY}*"))) == 2


def test_commit_never_clobbers_an_existing_directory(tmp_path):
    tmp_dir, _ = rn.open_run_dir(tmp_path, KEY)
    (tmp_dir / "run.json").write_text("{}")
    (tmp_path / KEY).mkdir()
    with pytest.raises(rn.RunnerError, match="refusing to overwrite"):
        rn.commit_run_dir(tmp_dir)


def test_discard_intermediates_refuses_outside_the_work_root(tmp_path):
    class FakeSpec:
        paths = {"work_root": str(tmp_path / "work")}
        keep_intermediates = False

    (tmp_path / "work").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(rn.RunnerError, match="refusing to remove"):
        rn.discard_intermediates(FakeSpec(), {"work_dir": str(elsewhere)})
    assert elsewhere.exists()
