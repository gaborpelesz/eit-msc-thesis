import json

from bench import cli
from bench import runner as rn
from bench import sampler as smp
from bench import spec as sp
from bench import store as st

from test_store import _record, _result  # noqa: F401


def test_dry_run_prints_commands_and_writes_nothing(write_spec, spec_dict, capsys, tmp_path):
    path = write_spec(spec_dict)
    key = "ACMM__courtyard__w3200__author__r1"
    assert cli.main(["run", str(path), "--dry-run", "--only", key]) == 0
    out = capsys.readouterr().out
    assert key in out
    assert "docker run -d --name bench_" in out
    assert "MVS_BENCH_PHASES=1" in out
    assert "nothing was executed" in out
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "work").exists()


def test_dry_run_with_an_unknown_key_fails(write_spec, spec_dict):
    assert cli.main(["run", str(write_spec(spec_dict)), "--dry-run", "--only", "nope"]) == 2


def test_plan_projects_from_a_pilot_campaign(write_spec, spec_dict, manifest, capsys):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)
    for run in sp.expand(spec):
        tmp_dir, _ = rn.open_run_dir(campaign, run.key)
        st.write_record(tmp_dir, _record(spec, manifest, run, _result(wall=3600.0)))
        rn.commit_run_dir(tmp_dir)

    assert cli.main(["plan", str(spec.path), "--pilot", str(campaign)]) == 0
    out = capsys.readouterr().out
    assert "8 runs" in out
    # 8 runs x (3600 + 3 s preprocess)
    assert "projected 8.0 h" in out


def test_phases_command_prints_totals(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "phases.txt").write_text(
        "PHASE patchmatch BEGIN 0 pass=photometric\nPHASE patchmatch END 2000000000 pass=photometric\n"
    )
    (run_dir / "phases.harness.txt").write_text(
        "PHASE preprocess.convert BEGIN 0\nPHASE preprocess.convert END 1000000000\n"
    )
    assert cli.main(["phases", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "patchmatch" in out and "2.000" in out
    assert "preprocess.convert" in out and "1.000" in out
    assert "photometric" in out


def test_status_and_query_commands(write_spec, spec_dict, manifest, capsys):
    spec = sp.load(write_spec(spec_dict), manifest)
    campaign = spec.campaign_dir
    campaign.mkdir(parents=True)
    run = sp.expand(spec)[0]
    tmp_dir, _ = rn.open_run_dir(campaign, run.key)
    st.write_record(tmp_dir, _record(spec, manifest, run, _result()))
    smp.Sampler().write_parquet(tmp_dir / "telemetry.parquet")
    rn.commit_run_dir(tmp_dir)

    assert cli.main(["status", str(campaign), "--spec", str(spec.path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["finished"] == 1 and len(report["remaining"]) == 7

    assert cli.main(["query", str(campaign), "--report", "wall"]) == 0
    assert "repeats" in capsys.readouterr().out

    assert cli.main(["query", str(campaign), "--sql", "SELECT count(*) AS n FROM runs"]) == 0
    assert "1" in capsys.readouterr().out


def test_spec_error_exits_two(write_spec, spec_dict, capsys):
    spec_dict["methods"] = ["DVP-MVS"]
    assert cli.main(["plan", str(write_spec(spec_dict))]) == 2
    assert "excluded from the campaign" in capsys.readouterr().err
