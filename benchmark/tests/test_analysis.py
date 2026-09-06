"""The analysis package, on a synthetic store built by the real record writer."""

import math

import pytest

from bench import runner as rn
from bench import spec as sp
from bench import store as st
from bench.analysis import frames as fr
from bench.analysis import markdown as md
from bench.analysis import pairs as pa
from bench.analysis import stats
from bench.analysis import variance as va
from deviations import manifest as mf

ROOT = mf.repo_root(mf.default_manifest_path().parent)


def _result(wall, f1, phases=None, **extra):
    result = {
        "status": "ok",
        "exit_code": 0,
        "wall_time_s": wall,
        "wall_time_with_preprocess_s": wall + 50.0,
        "started_at": "2026-09-06T10:00:00+00:00",
        "ended_at": "2026-09-06T10:10:00+00:00",
        "work_dir": "/scratch/bench/x",
        "preprocess": {"argv": ["convert"], "exit_code": 0, "duration_s": 50.0},
        "status_evidence": {"exit_code": 0},
        "telemetry": {
            "sample_interval_ms": 100.0,
            "sample_count": 100,
            "peak_host_rss_bytes": 8 << 30,
            "peak_device_mem_bytes": 4 << 30,
            "device_mem_attribution": "process",
            "energy_j": 1000.0 + wall,
            "io_write_bytes": 1 << 28,
        },
        "phases": phases
        if phases is not None
        else [
            {"name": "run", "total_s": wall - 1, "self_s": 0.1, "count": 1,
             "unclosed": 0, "source": "method"},
            {"name": "patchmatch", "total_s": 0.6 * wall, "self_s": 0.4 * wall,
             "count": 20, "unclosed": 0, "source": "method"},
            {"name": "patchmatch.init", "total_s": 0.2 * wall, "self_s": 0.2 * wall,
             "count": 20, "unclosed": 0, "source": "method"},
            {"name": "fusion", "total_s": 0.3 * wall, "self_s": 0.3 * wall,
             "count": 1, "unclosed": 0, "source": "method"},
            {"name": "preprocess.convert", "total_s": 50.0, "self_s": 50.0,
             "count": 1, "unclosed": 0, "source": "harness"},
        ],
        "phases_by_pass": [],
        "phase_trace_errors": [],
        "neighbour_list": {"path": None, "sha256": None},
        "point_cloud": {"path": "/a.ply", "size_bytes": 10, "sha256": "a" * 64,
                        "point_count": 5},
        "quality": [
            {"tolerance": 0.01, "accuracy": 0.7, "completeness": 0.6, "f1": f1 - 0.1},
            {"tolerance": 0.02, "accuracy": 0.8, "completeness": 0.7, "f1": f1},
        ],
        "command": ["docker", "run", "img"],
        "notes": [],
        "warm_cache_bytes": 1,
        "clock_hold": {"expected_mhz": 1800, "min_mhz": 1800, "max_mhz": 1800,
                       "samples": 100, "fraction_at_expected": 1.0, "held": True,
                       "method": "idle-clock-inference"},
    }
    result.update(extra)
    return result


def _campaign(write_spec, spec_dict, manifest, name, results, **overrides):
    """Write a finished campaign of `results` (one per repeat) and return its dir."""
    spec_dict = dict(spec_dict)
    spec_dict.update(
        {"campaign": name, "methods": ["ACMM"], "scenes": ["courtyard"],
         "widths": [3200], "repeats": len(results), **overrides}
    )
    spec = sp.load(write_spec(spec_dict, f"{name}.yaml"), manifest)
    spec.campaign_dir.mkdir(parents=True, exist_ok=True)
    entry = sp.method_entry(manifest, "ACMM")
    deviations = {"methods": [{"name": "ACMM", "fork_sha": "f" * 40}]}
    fingerprint = {"gpu_model": "RTX 2080 Ti", "image_digest": "sha256:d",
                   "image_manifest_sha256": "m", "submodules": []}
    for run in sp.expand(spec):
        tmp_dir, _ = rn.open_run_dir(spec.campaign_dir, run.key)
        record = st.build_record(
            spec, manifest, entry, run, results[run.repeat - 1], fingerprint,
            deviations, ROOT,
        )
        st.write_record(tmp_dir, record)
        rn.commit_run_dir(tmp_dir)
    return spec.campaign_dir


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def test_describe_reports_dispersion_and_refuses_to_invent_one():
    summary = stats.describe([10.0, 12.0, 14.0])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(12.0)
    assert summary["sd"] == pytest.approx(2.0)
    assert summary["cv"] == pytest.approx(2.0 / 12.0)
    assert (summary["min"], summary["max"]) == (10.0, 14.0)
    # A percentile bootstrap of the mean cannot leave the sample's range.
    assert 10.0 <= summary["ci_low"] <= 12.0 <= summary["ci_high"] <= 14.0

    single = stats.describe([10.0])
    assert single["n"] == 1 and single["mean"] == 10.0
    assert single["sd"] is None and single["ci_low"] is None
    assert stats.describe([])["n"] == 0


def test_the_bootstrap_is_reproducible_and_records_its_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = stats.bootstrap_mean_ci(values, resamples=2000, seed=7)
    assert first == stats.bootstrap_mean_ci(values, resamples=2000, seed=7)
    # A different seed draws a different resample distribution; at these
    # sizes the 2.5 % percentile itself may land on the same value, so the
    # check is on the draws, not on the rounded endpoint.
    assert stats.bootstrap_mean_ci(values, resamples=13, seed=7) != (
        stats.bootstrap_mean_ci(values, resamples=13, seed=8)
    )
    assert stats.describe(values)["seed"] == stats.BOOTSTRAP_SEED


def test_t_critical_matches_the_table_and_falls_back_to_the_normal():
    assert stats.t_critical(1) == pytest.approx(12.706)
    assert stats.t_critical(4) == pytest.approx(2.776)
    assert stats.t_critical(2.7) == pytest.approx(4.303)  # a Welch df rounds down
    assert stats.t_critical(200) == pytest.approx(stats.Z975)
    assert stats.t_critical(0) is None


def test_projected_halfwidth_is_undefined_at_one_repeat():
    projection = stats.projected_halfwidth(mean=100.0, sd=5.0, n=1)
    assert projection["t_halfwidth"] is None
    assert projection["z_halfwidth"] == pytest.approx(stats.Z975 * 5.0)

    three = stats.projected_halfwidth(100.0, 5.0, 3)
    five = stats.projected_halfwidth(100.0, 5.0, 5)
    assert three["t_halfwidth"] == pytest.approx(4.303 * 5.0 / math.sqrt(3))
    assert five["t_halfwidth_pct"] < three["t_halfwidth_pct"]


def test_welch_difference_and_what_the_sample_size_can_resolve():
    a = [100.0, 102.0, 98.0]
    b = [110.0, 112.0, 108.0]
    result = stats.welch(a, b)
    assert result["difference"] == pytest.approx(10.0)
    assert result["relative"] == pytest.approx(0.1)
    assert result["ci_low"] < 10.0 < result["ci_high"]
    assert result["ci_low"] > 0  # a 10 % shift is resolvable at 3 vs 3 here
    assert stats.minimum_detectable_difference(a, b) == pytest.approx(
        (result["ci_high"] - result["ci_low"]) / 2
    )
    # Two runs one way round is the negative of the other.
    assert stats.welch(b, a)["difference"] == pytest.approx(-10.0)
    assert stats.welch(a, [])["difference"] is None


def test_a_difference_smaller_than_the_noise_has_an_interval_containing_zero():
    a = [100.0, 110.0, 90.0]
    b = [101.0, 111.0, 91.0]
    result = stats.welch(a, b)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert stats.minimum_detectable_difference(a, b) > abs(result["difference"])


# --------------------------------------------------------------------------
# frames and variance, over a real store
# --------------------------------------------------------------------------


@pytest.fixture
def variance_store(write_spec, spec_dict, manifest):
    return _campaign(
        write_spec, spec_dict, manifest, "var-dev",
        [_result(700.0 + delta, 0.83 + delta / 10000)
         for delta in (0.0, 6.0, -4.0, 2.0, -2.0)],
    )


def test_records_and_metrics_come_back_in_the_units_the_tables_print(variance_store):
    records = fr.records(variance_store)
    assert len(records) == 5
    assert fr.metric(records[0], "peak_device_mem_bytes") == pytest.approx(4096.0)
    assert fr.metric(records[0], "wall_time_s") > 0
    assert 0.02 in fr.quality_series(records[0])
    totals = fr.phase_totals(records[0])
    # Top level only: `patchmatch.init` is inside `patchmatch`'s total already.
    assert "patchmatch" in totals and "patchmatch.init" not in totals
    assert "preprocess.convert" not in totals  # harness spans are not method phases
    shares = fr.phase_shares(records[0])
    assert shares["patchmatch"] == pytest.approx(
        totals["patchmatch"] / fr.phase_totals(records[0])["run"]
    )


def test_duckdb_view_spans_every_finished_run(variance_store):
    con = fr.connect([variance_store])
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 5
    assert con.execute("SELECT count(DISTINCT repeat) FROM runs").fetchone()[0] == 5


def test_variance_table_covers_wall_quality_phases_and_memory(variance_store):
    table = va.by_method(variance_store, resamples=200)
    rows = {r["metric"]: r for r in table["ACMM"]}
    assert rows["wall_time_s"]["n"] == 5
    assert rows["wall_time_s"]["mean"] == pytest.approx(700.4)
    assert rows["wall_time_s"]["sd"] > 0
    assert rows["wall_time_s"]["cv"] == pytest.approx(
        rows["wall_time_s"]["sd"] / rows["wall_time_s"]["mean"]
    )
    for expected in ("f1@0.01", "f1@0.02", "phase:patchmatch", "phase:fusion",
                     "peak_device_mem_bytes", "peak_host_rss_bytes"):
        assert expected in rows, expected
    assert "phase:patchmatch.init" not in rows


def test_sizing_answers_the_d17_question_for_one_three_and_five(variance_store):
    table = va.sizing(variance_store, resamples=200)
    rows = {(r["metric"], r["n"]): r for r in table["ACMM"]}
    assert {n for (_, n) in rows} == {1, 3, 5}
    assert rows[("wall_time_s", 1)]["t_halfwidth_pct"] is None
    assert rows[("wall_time_s", 1)]["z_halfwidth_pct"] is not None
    assert (
        rows[("wall_time_s", 5)]["t_halfwidth_pct"]
        < rows[("wall_time_s", 3)]["t_halfwidth_pct"]
    )
    assert ("f1@0.02", 3) in rows


def test_the_status_table_names_every_arm_setting(variance_store):
    rows = va.status_rows(variance_store)
    assert len(rows) == 5
    assert {r["status"] for r in rows} == {"ok"}
    assert {r["instrumented"] for r in rows} == {True}
    assert {r["debug_output"] for r in rows} == {"off"}
    assert {r["clock_fraction"] for r in rows} == {1.0}
    assert all(r["provenance"] == "published-reference" for r in rows)


def test_variance_renders_markdown_with_no_hand_typed_number(variance_store):
    text = va.render(variance_store, resamples=200)
    assert text.count("|") > 50
    assert "wall time (s)" in text and "F1 @ 2 cm" in text
    assert "repeat count (D17)" in text
    assert "—" in text  # the undefined N=1 t half-width is shown, not filled in


# --------------------------------------------------------------------------
# pairs
# --------------------------------------------------------------------------


@pytest.fixture
def timer_pair(write_spec, spec_dict, manifest):
    on = _campaign(
        write_spec, spec_dict, manifest, "timer-on",
        [_result(w, 0.83) for w in (700.0, 706.0, 696.0, 702.0, 698.0)],
    )
    off = _campaign(
        write_spec, spec_dict, manifest, "timer-off",
        [_result(w, 0.83, phases=[]) for w in (690.0, 694.0, 686.0)],
        phase_timer=False,
    )
    return on, off


def test_an_arm_can_be_restricted_to_a_subset_of_repeats(timer_pair):
    on, _ = timer_pair
    assert len(pa.Arm("on", [on]).records) == 5
    assert len(pa.Arm("on", [on], repeats=[1, 2, 3]).records) == 3


def test_the_timer_pair_reports_a_difference_of_means_with_an_interval(timer_pair):
    on, off = timer_pair
    arm_a = pa.Arm("timer on", [on], repeats=[1, 2, 3])
    arm_b = pa.Arm("timer off", [off])
    rows = {r["metric"]: r for r in pa.compare(arm_a, arm_b, resamples=200)["ACMM"]}
    wall = rows["wall_time_s"]
    assert (wall["n_a"], wall["n_b"]) == (3, 3)
    assert wall["difference"] == pytest.approx(690.0 - 700.666666, abs=1e-3)
    assert wall["ci_low"] < wall["difference"] < wall["ci_high"]
    assert wall["boot_low"] is not None
    assert wall["mdd"] > 0


def test_the_uninstrumented_arm_has_no_phase_totals_to_compare(timer_pair):
    on, off = timer_pair
    rows = {
        r["metric"]
        for r in pa.compare(pa.Arm("on", [on]), pa.Arm("off", [off]), resamples=200)["ACMM"]
    }
    # Phase columns are not in PAIR_METRICS at all: they exist in one arm only,
    # so a difference of means over them would be a comparison with nothing.
    assert not any(m.startswith("phase:") for m in rows)
    off_records = fr.records(off)
    assert all(fr.phase_totals(r) == {} for r in off_records)
    assert all(r["instrumented"] is False for r in off_records)


def test_comparability_flags_an_arm_that_changed_more_than_the_setting(
    write_spec, spec_dict, manifest, timer_pair
):
    on, _ = timer_pair
    other_scene = _campaign(
        write_spec, spec_dict, manifest, "other-width",
        [_result(700.0, 0.83)], widths=[1600],
    )
    _, problems = pa.comparability(pa.Arm("a", [on]), pa.Arm("b", [other_scene]))
    assert any("width" in p for p in problems)
    facts, clean = pa.comparability(pa.Arm("a", [on]), pa.Arm("b", [on]))
    assert clean == []
    assert facts["image_digest"] == ["sha256:d"]


def test_phase_share_shift_is_what_the_sync_pair_measures(write_spec, spec_dict, manifest):
    def phases(init_share):
        return [
            {"name": "run", "total_s": 100.0, "self_s": 0.0, "count": 1,
             "unclosed": 0, "source": "method"},
            {"name": "patchmatch.init", "total_s": 100.0 * init_share, "self_s": 0.0,
             "count": 1, "unclosed": 0, "source": "method"},
            {"name": "patchmatch.propagate", "total_s": 100.0 * (0.8 - init_share),
             "self_s": 0.0, "count": 1, "unclosed": 0, "source": "method"},
        ]

    off = _campaign(write_spec, spec_dict, manifest, "sync-off",
                    [_result(100.0, 0.8, phases=phases(0.05)) for _ in range(3)])
    on = _campaign(write_spec, spec_dict, manifest, "sync-on",
                   [_result(100.0, 0.8, phases=phases(0.30)) for _ in range(3)],
                   method_env={"MVS_BENCH_SYNC": "1"})
    shift = {
        r["phase"]: r
        for r in pa.phase_share_shift(pa.Arm("off", [off]), pa.Arm("on", [on]), "ACMM")
    }
    assert shift["patchmatch.init"]["shift"] == pytest.approx(0.25)
    assert shift["patchmatch.propagate"]["shift"] == pytest.approx(-0.25)
    assert shift["run"]["shift"] == pytest.approx(0.0)
    assert all(r["mvs_bench_sync"] == "1" for r in fr.records(on))
    assert all(r["mvs_bench_sync"] is None for r in fr.records(off))


def test_pair_renders_markdown_and_states_what_it_holds_constant(timer_pair):
    on, off = timer_pair
    text = pa.render(
        pa.Arm("timer on", [on], repeats=[1, 2, 3]), pa.Arm("timer off", [off]),
        "R-TIM-09", resamples=200, with_shares=True,
    )
    assert "Arm A = **timer on** (3 runs)" in text
    assert "identical across the two arms" in text
    assert "wall time (s)" in text
    assert "resolvable ±" in text


def test_campaign_settings_are_read_back_from_the_records(timer_pair):
    on, off = timer_pair
    settings = fr.campaign_settings([on, off])
    assert settings["timer-on"]["phase_timer"] == [True]
    assert settings["timer-off"]["phase_timer"] == [False]
    assert settings["timer-on"]["runs"] == 5
    assert settings["timer-off"]["runs"] == 3


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def test_markdown_table_alignment_and_missing_values():
    text = md.table(["a", "b"], [["1", "2"], ["30", "—"]], align="lr")
    lines = text.splitlines()
    assert lines[0].startswith("| a")
    assert lines[1].endswith(":|")
    assert len(lines) == 4
    assert md.fmt(None) == "—"
    assert md.fmt(float("nan")) == "—"
    assert md.fmt(True) == "yes"
    assert md.pct(None) == "—"
    assert md.pct(12.3456) == "12.35 %"


# --------------------------------------------------------------------------
# what may enter an aggregate
# --------------------------------------------------------------------------


@pytest.fixture
def mixed_store(write_spec, spec_dict, manifest):
    crashed = _result(300.0, 0.0)
    crashed.update(status="nonzero_exit", quality=None,
                   point_cloud={"path": None, "size_bytes": None, "sha256": None})
    unheld = _result(705.0, 0.83)
    unheld["clock_hold"] = {**unheld["clock_hold"], "fraction_at_expected": 0.89,
                            "held": False}
    return _campaign(
        write_spec, spec_dict, manifest, "mixed",
        [_result(700.0, 0.83), crashed, unheld],
    )


def test_a_failed_run_is_kept_in_the_store_and_out_of_the_aggregates(mixed_store):
    kept, excluded = fr.select(mixed_store)
    assert len(fr.records(mixed_store)) == 3  # nothing is dropped from the store
    assert [r["repeat"] for r in kept] == [1, 3]
    assert excluded == [("ACMM__courtyard__w3200__author__r2", "status `nonzero_exit`")]

    wall = {r["metric"]: r for r in va.by_method(mixed_store, resamples=200)["ACMM"]}[
        "wall_time_s"
    ]
    assert wall["n"] == 2
    assert wall["mean"] == pytest.approx(702.5)  # 300 s of crash is not a runtime


def test_the_clock_hold_exclusion_is_opt_in_and_named(mixed_store):
    kept, excluded = fr.select(mixed_store, require_clock_hold=True)
    assert [r["repeat"] for r in kept] == [1]
    assert any("clock_hold.held=False" in why for _, why in excluded)
    assert len(fr.select(mixed_store)[0]) == 2  # off by default


def test_every_report_says_what_it_left_out(mixed_store):
    text = va.render(mixed_store, resamples=200)
    assert "1 of 3 runs are **excluded**" in text
    assert "status `nonzero_exit`" in text
    assert "kept in the store (R-FAIL-01)" in text

    arm = pa.Arm("mixed", [mixed_store])
    assert len(arm.records) == 2
    assert arm.excluded == [("ACMM__courtyard__w3200__author__r2", "status `nonzero_exit`")]
    paired = pa.render(arm, arm, "self", resamples=200)
    assert "excluded — status `nonzero_exit`" in paired


def test_the_exclusion_note_says_so_when_nothing_was_excluded(variance_store):
    assert "All 5 runs are in the aggregates" in va.render(variance_store, resamples=200)
