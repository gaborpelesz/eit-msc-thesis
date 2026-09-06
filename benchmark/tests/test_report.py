"""The results -> manuscript generator, on a synthetic store and on the real one.

The synthetic campaign is deliberately awkward: two repeats of one method, a
run that crashed, a run whose evaluation failed, a truncated phase trace, and a
method whose provenance is `third-party-optimization`. Every one of those is a
case the manuscript must render honestly.
"""

import json
import shutil

import pytest
import yaml
from conftest import braces_balance
from report import cli as rc
from report import figures as fg
from report import load as ld
from report import tables as tb

PILOT_A = "/data/bench/results/pilot-a-dev"

TRACE = """PHASE run BEGIN 0
PHASE problem_list BEGIN 100000000
PHASE problem_list END 200000000
PHASE image_pass BEGIN 300000000 pass=photometric
PHASE input_load BEGIN 300000000
PHASE input_load END 800000000
PHASE patchmatch BEGIN 800000000
PHASE patchmatch.propagate BEGIN 900000000
PHASE patchmatch.propagate END 2900000000
PHASE patchmatch END 3000000000
PHASE image_pass END 3100000000 pass=photometric
PHASE fusion BEGIN 3100000000
PHASE fusion END 3900000000
PHASE run END 4000000000
"""

TRUNCATED_TRACE = """PHASE run BEGIN 0
PHASE input_load BEGIN 100000000
PHASE input_load END 600000000
"""


def _totals_from(text):
    from bench import phases as ph

    return [{**row, "source": "method"} for row in ph.totals(ph.parse(text))]


def make_record(method, provenance, **overrides):
    record = {
        "record_schema_version": 1,
        "campaign": "synthetic",
        "run_key": f"{method}__pipes__w1600__author__r1",
        "method": method,
        "scene": "pipes",
        "width": 1600,
        "configuration": "author",
        "repeat": 1,
        "provenance": provenance,
        "status": "ok",
        "wall_time_s": 10.0,
        "wall_time_with_preprocess_s": 12.0,
        "preprocess_convert_s": 2.0,
        "peak_device_mem_bytes": 512 * 1024 * 1024,
        "peak_host_rss_bytes": 1024 * 1024 * 1024,
        "device_mem_attribution": "process",
        "primary_tolerance": 0.02,
        "f1_primary": 0.5,
        "quality": [
            {"tolerance": 0.01, "f1": 0.4, "accuracy": 0.5, "completeness": 0.35},
            {"tolerance": 0.02, "f1": 0.5, "accuracy": 0.6, "completeness": 0.45},
        ],
        "phases": _totals_from(TRACE),
        "trace": TRACE,
    }
    record.update(overrides)
    return record


@pytest.fixture
def synthetic_store(tmp_path):
    """A campaign directory with the awkward cases, laid out as `bench run` does."""
    campaign = tmp_path / "synthetic"
    campaign.mkdir()
    (campaign / "spec.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "campaign": "synthetic",
                "image": "mvs-bench:test",
                "repeats": 2,
                "primary_tolerance": 0.02,
            }
        )
    )
    (campaign / "fingerprint.json").write_text(
        json.dumps(
            {
                "gpu_model": "NVIDIA GeForce RTX 2080 Ti",
                "driver_version": "575.51.03",
                "cuda_version": "12.9",
                "hostname": "mug",
                "image": "mvs-bench:test",
                "image_digest": "sha256:0123456789abcdef",
            }
        )
    )
    (campaign / "deviations.json").write_text(json.dumps({"methods": []}))

    records = [
        make_record("ACMH", "published-reference"),
        make_record(
            "ACMH",
            "published-reference",
            run_key="ACMH__pipes__w1600__author__r2",
            repeat=2,
            wall_time_s=12.0,
            f1_primary=0.6,
            quality=[
                {"tolerance": 0.01, "f1": 0.5, "accuracy": 0.6, "completeness": 0.45},
                {"tolerance": 0.02, "f1": 0.6, "accuracy": 0.7, "completeness": 0.55},
            ],
        ),
        make_record(
            "ACMM",
            "published-reference",
            status="nonzero_exit",
            wall_time_s=None,
            wall_time_with_preprocess_s=None,
            preprocess_convert_s=None,
            f1_primary=None,
            quality=[],
            peak_device_mem_bytes=None,
            phases=[],
            trace="",
        ),
        make_record(
            "ACMP",
            "published-reference",
            status="eval_failed",
            f1_primary=None,
            quality=[],
        ),
        make_record(
            "CUMVS",
            "third-party-optimization",
            phases=_totals_from(TRUNCATED_TRACE),
            trace=TRUNCATED_TRACE,
        ),
    ]
    for record in records:
        trace = record.pop("trace")
        directory = campaign / record["run_key"]
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True))
        if trace:
            (directory / "phases.txt").write_text(trace)
    # A superseded run must never re-enter an aggregate.
    shutil.copytree(
        campaign / "ACMH__pipes__w1600__author__r1",
        campaign / "ACMH__pipes__w1600__author__r1.20260101T000000Z.superseded",
    )
    return campaign


@pytest.fixture
def synthetic_manifest():
    return {
        "version": 1,
        "canonical_phases": [
            "problem_list",
            "input_load",
            "patchmatch",
            "patchmatch.propagate",
            "image_pass",
            "fusion",
            "preprocess.convert",
        ],
        "methods": [
            {"name": "ACMH", "provenance": "published-reference"},
            {"name": "ACMM", "provenance": "published-reference"},
            {"name": "ACMP", "provenance": "published-reference"},
            {"name": "CUMVS", "provenance": "third-party-optimization"},
        ],
    }


@pytest.fixture
def campaign(synthetic_store, synthetic_manifest):
    return ld.Campaign(synthetic_store, synthetic_manifest)


# -- refusals ---------------------------------------------------------------


def test_missing_provenance_is_refused(synthetic_store, synthetic_manifest):
    synthetic_manifest["methods"][0].pop("provenance")
    with pytest.raises(ld.ReportError, match="provenance"):
        ld.Campaign(synthetic_store, synthetic_manifest)


def test_unknown_method_is_refused(synthetic_store, synthetic_manifest):
    synthetic_manifest["methods"] = [
        m for m in synthetic_manifest["methods"] if m["name"] != "CUMVS"
    ]
    with pytest.raises(ld.ReportError, match="not in the manifest"):
        ld.Campaign(synthetic_store, synthetic_manifest)


def test_provenance_disagreement_is_refused(synthetic_store, synthetic_manifest):
    """A record made under a different manifest may not be silently relabelled."""
    for method in synthetic_manifest["methods"]:
        if method["name"] == "CUMVS":
            method["provenance"] = "own"
    with pytest.raises(ld.ReportError, match="recorded provenance"):
        ld.Campaign(synthetic_store, synthetic_manifest)


def test_empty_directory_is_refused(tmp_path, synthetic_manifest):
    with pytest.raises(ld.ReportError):
        ld.Campaign(tmp_path, synthetic_manifest)


def test_unmapped_canonical_phase_is_refused(campaign):
    campaign.manifest["canonical_phases"].append("brand_new_phase")
    with pytest.raises(ld.ReportError, match="brand_new_phase"):
        ld.phase_rows(campaign)


# -- aggregates -------------------------------------------------------------


def test_superseded_run_is_not_aggregated(campaign):
    assert len(campaign.records) == 5
    assert all(".superseded" not in r["run_key"] for r in campaign.records)


def test_statistics_use_ok_runs_only(campaign):
    rows = {(r["method"], r["configuration"]): r for r in campaign.summary_rows()}
    acmh = rows[("ACMH", "author")]
    assert acmh["n_ok"] == 2 and acmh["runs"] == 2
    assert acmh["wall_mean"] == pytest.approx(11.0)
    assert acmh["wall_sd"] == pytest.approx(2**0.5)
    # The crashed run contributes a count, not a number.
    acmm = rows[("ACMM", "author")]
    assert acmm["n_ok"] == 0 and acmm["runs"] == 1
    assert acmm["wall_mean"] is None
    # `eval_failed` produced a wall time but no quality; it is not averaged in.
    acmp = rows[("ACMP", "author")]
    assert acmp["n_ok"] == 0 and acmp["wall_mean"] is None


def test_no_failure_is_dropped(campaign):
    counts = campaign.status_counts()
    assert counts[("ACMM", "author")] == {"nonzero_exit": 1}
    assert counts[("ACMP", "author")] == {"eval_failed": 1}
    total = sum(sum(v.values()) for v in counts.values())
    assert total == len(campaign.records)


def test_single_run_has_no_standard_deviation(campaign):
    rows = {(r["method"], r["configuration"]): r for r in campaign.summary_rows()}
    assert rows[("CUMVS", "author")]["wall_sd"] is None
    assert "$\\pm$" not in tb.pm(10.0, None)


def test_tolerance_rows_skip_runs_with_no_point_cloud(campaign):
    rows = campaign.tolerance_rows()
    assert not [r for r in rows if r["method"] in ("ACMM", "ACMP")]
    acmh = {r["tolerance"]: r for r in rows if r["method"] == "ACMH"}
    assert acmh[0.02]["n"] == 2
    assert acmh[0.02]["f1_mean"] == pytest.approx(0.55)


# -- phases -----------------------------------------------------------------


def test_phase_shares_sum_to_one_hundred(campaign):
    rows, _ = ld.phase_rows(campaign)
    acmh = next(r for r in rows if r["method"] == "ACMH")
    assert acmh["base_is_run"]
    assert acmh["base_s"] == pytest.approx(4.0)
    assert sum(acmh["shares"].values()) + acmh["residual"] == pytest.approx(100.0)
    assert acmh["shares"]["patchmatch"] == pytest.approx(55.0)  # 2.2 s of 4.0 s
    # 0.3 s of the 4.0 s `run` span lies in no phase span at all.
    assert acmh["residual"] == pytest.approx(7.5)


def test_truncated_trace_falls_back_to_summed_time(campaign):
    rows, _ = ld.phase_rows(campaign)
    cumvs = next(r for r in rows if r["method"] == "CUMVS")
    assert not cumvs["base_is_run"]
    assert cumvs["base_s"] == pytest.approx(0.5)
    assert "$^{*}$" in tb.phases(campaign, rows, [])


def test_phase_row_uses_the_longest_ok_repeat(campaign):
    rows, _ = ld.phase_rows(campaign)
    acmh = next(r for r in rows if r["method"] == "ACMH")
    assert acmh["run_key"] == "ACMH__pipes__w1600__author__r2"


def test_methods_without_a_completed_run_are_named(campaign):
    rows, _ = ld.phase_rows(campaign)
    assert {r["method"] for r in rows} == {"ACMH", "CUMVS"}
    text = tb.phases(campaign, rows, [])
    assert "No completed run, hence no row: ACMM, ACMP." in text


def test_trace_disagreement_is_detected(campaign):
    record = next(r for r in campaign.records if r["method"] == "ACMH")
    record["phases"] = [
        {**row, "self_s": row["self_s"] + 1.0} if row["name"] == "fusion" else row
        for row in record["phases"]
    ]
    directory = campaign.directory / record["run_key"]
    assert "fusion" in ld.trace_disagreement(directory, record)


# -- rendered artefacts -----------------------------------------------------


def test_render_writes_every_artefact(campaign, tmp_path):
    paths, _ = rc.render_campaign(campaign, tmp_path / "out")
    names = {p.name for p in paths}
    assert names == {
        "results-synthetic-summary.tex",
        "results-synthetic-phases.tex",
        "results-synthetic-f1-tolerances.tex",
        "results-synthetic-phase-shares.tex",
        "results-synthetic-wall-f1.tex",
    }
    for path in paths:
        assert path.read_text().startswith("% GENERATED by")


def test_render_is_deterministic(campaign, tmp_path):
    first, _ = rc.render_campaign(campaign, tmp_path / "a")
    second, _ = rc.render_campaign(campaign, tmp_path / "b")
    for left, right in zip(first, second):
        assert left.read_text() == right.read_text()


def test_every_table_carries_a_label_and_the_campaign_footer(campaign):
    rows, disagreements = ld.phase_rows(campaign)
    for text, kind in (
        (tb.summary(campaign), "summary"),
        (tb.phases(campaign, rows, disagreements), "phases"),
        (tb.tolerances(campaign), "f1-tolerances"),
    ):
        assert f"\\label{{tab:results-synthetic-{kind}}}" in text
        assert "Campaign synthetic" in text
        assert "RTX 2080 Ti" in text
        assert "sha256:0123456789ab" in text
        assert "2 repeat(s) planned" in text


def test_tables_state_provenance_from_the_manifest(campaign):
    text = tb.summary(campaign)
    assert "third-party opt." in text
    assert "published ref." in text


def test_failures_appear_in_the_summary_table(campaign):
    text = tb.summary(campaign)
    assert "nonzero\\_exit~1" in text
    assert "eval\\_failed~1" in text


def test_every_method_that_ran_appears(campaign):
    text = tb.summary(campaign)
    for method in campaign.methods:
        assert method in text


def test_tex_escapes_specials_in_generated_tables(campaign):
    rows, _ = ld.phase_rows(campaign)
    for text in (tb.summary(campaign), tb.phases(campaign, rows, []), tb.tolerances(campaign)):
        for line in text.splitlines():
            if line.startswith("%"):
                continue
            for index, character in enumerate(line):
                if character in "_#":
                    assert index and line[index - 1] == "\\", f"{character} in {line!r}"


def test_figures_are_pgfplots_and_name_their_preamble(campaign):
    rows, _ = ld.phase_rows(campaign)
    for text in (fg.phase_shares(campaign, rows), fg.wall_vs_f1(campaign)):
        assert "\\usepackage{pgfplots}" in text
        assert "\\begin{axis}[" in text
        assert "\\label{fig:results-synthetic-" in text


def test_wall_vs_f1_names_methods_it_could_not_plot(campaign):
    text = fg.wall_vs_f1(campaign)
    assert "Not plotted, no completed run: ACMM, ACMP." in text


# -- the real store ---------------------------------------------------------


@pytest.mark.skipif(
    not __import__("pathlib").Path(PILOT_A).is_dir(), reason="pilot-a-dev is not present"
)
def test_pilot_a_renders(tmp_path, manifest):
    campaign = ld.Campaign(PILOT_A, manifest)
    paths, _ = rc.render_campaign(campaign, tmp_path)
    assert len(paths) == 5
    summary = (tmp_path / "results-pilot-a-dev-summary.tex").read_text()
    # CUMVS is prior art by a third party, never the thesis author's work.
    assert "CUMVS & third-party opt." in summary
    assert "HPM-MVS\\_plusplus" in summary


def test_generated_tex_has_balanced_braces(campaign):
    rows, _ = ld.phase_rows(campaign)
    for text in (
        tb.summary(campaign),
        tb.phases(campaign, rows, []),
        tb.tolerances(campaign),
        fg.phase_shares(campaign, rows),
        fg.wall_vs_f1(campaign),
    ):
        assert braces_balance(text)[0] == braces_balance(text)[1]
