import pytest

from bench import runner as rn
from bench import spec as sp


def _spec(write_spec, spec_dict, manifest, **overrides):
    spec_dict.update(overrides)
    return sp.load(write_spec(spec_dict), manifest)


def test_measured_docker_command(write_spec, spec_dict, manifest):
    spec = _spec(write_spec, spec_dict, manifest, methods=["ACMM"])
    run = next(r for r in sp.expand(spec) if r.width == 3200 and r.repeat == 1)
    entry = sp.method_entry(manifest, "ACMM")
    plan = rn.plan_commands(spec, entry, run)
    command = plan["measured"]["docker"]

    assert command[:3] == ["docker", "run", "-d"]
    assert "--gpus" in command and "device=0" in command
    raw = spec.raw_scene_dir("courtyard", 3200)
    # Calibration under both names: ACM converters want `sparse`, APD/DPE/CUMVS
    # want ETH3D's directory name.
    assert f"{raw}/images:/data/images:ro" in command
    assert f"{raw}/dslr_calibration_undistorted:/data/dslr_calibration_undistorted:ro" in command
    assert f"{raw}/dslr_calibration_undistorted:/data/sparse:ro" in command
    assert not any(m.endswith(":/data:ro") for m in command)
    assert "MVS_BENCH_PHASES=1" in command
    assert "MVS_BENCH_FILE=/out/phases.txt" in command
    assert command[command.index("--entrypoint") + 1] == "/sota/ACMM/build/ACMM"
    assert command[-1] == "/work/prepared"


def test_author_configuration_uses_the_forks_own_converter(write_spec, spec_dict, manifest):
    spec = _spec(write_spec, spec_dict, manifest, methods=["APD-MVS"], configurations=["author"])
    run = sp.expand(spec)[0]
    entry = sp.method_entry(manifest, "APD-MVS")
    argv = rn.preprocess_argv(spec, entry, run)
    assert argv[0] == "/sota/.venv/bin/python"
    assert argv[1] == "/sota/APD-MVS/colmap2mvsnet.py"
    assert argv[2:] == [
        "--dense_folder",
        "/data",
        "--save_folder",
        "/work/prepared",
    ]


def test_norm10_uses_the_shared_converter_with_the_neighbour_count(
    write_spec, spec_dict, manifest
):
    spec = _spec(
        write_spec,
        spec_dict,
        manifest,
        methods=["ACMM"],
        configurations=["norm10"],
        padding="all",
    )
    run = sp.expand(spec)[0]
    argv = rn.preprocess_argv(spec, sp.method_entry(manifest, "ACMM"), run)
    assert argv[0].endswith("colmap2mvsnet_acm_perf")
    assert "--neighbours" in argv and "10" in argv
    assert "--padding" in argv


def test_cumvs_preprocessing_is_its_own_initialiser(write_spec, spec_dict, manifest):
    spec = _spec(write_spec, spec_dict, manifest, methods=["CUMVS"], configurations=["norm10"])
    run = sp.expand(spec)[0]
    entry = sp.method_entry(manifest, "CUMVS")
    argv = rn.preprocess_argv(spec, entry, run)
    assert argv[0] == "/sota/cuda-multi-view-stereo/build/samples/app_initialize_ETH3D"
    assert argv[1] == "/data"
    assert "--output-directory=/work/prepared" in argv
    assert "--max-neighbors=10" in argv
    method_argv = rn.resolve_invocation(spec, entry, run)
    assert method_argv[-1] == "--output-directory=/work/prepared/CUMVS"


def test_converter_source_shared_uses_the_shared_converter_under_author(
    write_spec, spec_dict, manifest
):
    """A method that ships no converter has nothing for `author` to run, so the
    shared one prepares its input in both configurations -- the asymmetry the
    entry has to declare as a harness deviation."""
    spec = _spec(write_spec, spec_dict, manifest, methods=["ACMM"], configurations=["author"])
    run = sp.expand(spec)[0]
    entry = dict(sp.method_entry(manifest, "ACMM"), converter=None, converter_source="shared")
    argv = rn.preprocess_argv(spec, entry, run)
    assert argv[0].endswith("colmap2mvsnet_acm_perf")
    assert "--neighbours" not in argv
    assert argv[-4:] == ["--dense_folder", "/data", "--save_folder", "/work/prepared"]


def test_converter_source_shared_still_takes_the_normalized_neighbour_count(
    write_spec, spec_dict, manifest
):
    spec = _spec(write_spec, spec_dict, manifest, methods=["ACMM"], configurations=["norm10"])
    run = sp.expand(spec)[0]
    entry = dict(sp.method_entry(manifest, "ACMM"), converter=None, converter_source="shared")
    argv = rn.preprocess_argv(spec, entry, run)
    assert argv[-2:] == ["--neighbours", "10"]


def test_mp_mvs_gets_its_config_file_argument(write_spec, spec_dict, manifest):
    spec = _spec(write_spec, spec_dict, manifest, methods=["MP-MVS"])
    run = sp.expand(spec)[0]
    entry = sp.method_entry(manifest, "MP-MVS")
    argv = rn.resolve_invocation(spec, entry, run)
    # config.yaml must precede the flag: MP-MVS reads argv[2] as the config path (M-005).
    assert argv == ["/sota/MP-MVS/build/MPMVS", "/work/prepared", entry["config_path"], "--no-debug-output"]


@pytest.mark.parametrize(
    "exit_code,oom,stderr,timed_out,expected",
    [
        (0, False, "", False, "ok"),
        (0, False, "", True, "timeout"),
        (137, True, "", False, "oom_host"),
        (139, False, "", False, "segfault"),
        (1, False, "terminate called after throwing std::bad_alloc", False, "oom_host"),
        (1, False, "CUDA error: out of memory", False, "oom_gpu"),
        (1, False, "cudaErrorMemoryAllocation", False, "oom_gpu"),
        (1, False, "some other failure", False, "nonzero_exit"),
        (1, False, "Segmentation fault (core dumped)", False, "segfault"),
    ],
)
def test_status_classification(exit_code, oom, stderr, timed_out, expected):
    status, evidence = rn.classify(exit_code, oom, stderr, timed_out, {})
    assert status == expected
    assert evidence["exit_code"] == exit_code


def test_classification_records_its_evidence():
    _, evidence = rn.classify(
        1, False, "CUDA error: out of memory", False,
        {"last_device_mem_bytes": 11 << 30, "device_mem_total_bytes": 12 << 30},
    )
    assert evidence["log_markers"] == ["cuda error: out of memory"]
    assert evidence["last_device_mem_bytes"] == 11 << 30


def test_debug_output_upstream_removes_the_flag_from_the_invocation(
    write_spec, spec_dict, manifest
):
    entry = sp.method_entry(manifest, "APD-MVS")
    assert rn.DEBUG_OUTPUT_FLAG in entry["invocation"]

    off = _spec(write_spec, spec_dict, manifest, methods=["APD-MVS"])
    run = sp.expand(off)[0]
    assert rn.DEBUG_OUTPUT_FLAG in rn.resolve_invocation(off, entry, run)

    upstream = _spec(
        write_spec, dict(spec_dict), manifest, debug_output="upstream", methods=["APD-MVS"]
    )
    argv = rn.resolve_invocation(upstream, entry, sp.expand(upstream)[0])
    assert rn.DEBUG_OUTPUT_FLAG not in argv
    # Only the flag goes; the method still gets its dataset.
    assert argv == ["/sota/APD-MVS/build/APD", "/work/prepared"]


def test_debug_output_upstream_removes_dpe_mvs_guard_flag(write_spec, spec_dict, manifest):
    # DPE-MVS's DEBUG_COMPLEX block is compiled in and guarded by the flag at
    # run time, so removing the token restores that guard's released behaviour.
    spec = _spec(
        write_spec, spec_dict, manifest, debug_output="upstream", methods=["DPE-MVS"]
    )
    entry = sp.method_entry(manifest, "DPE-MVS")
    argv = rn.resolve_invocation(spec, entry, sp.expand(spec)[0])
    assert rn.DEBUG_OUTPUT_FLAG not in argv
    assert argv == ["/sota/DPE-MVS/DPE-MVS/build/DPE", "/work/prepared"]


def test_debug_output_state_distinguishes_reversed_from_inapplicable(
    write_spec, spec_dict, manifest
):
    off = _spec(write_spec, spec_dict, manifest)
    upstream = _spec(write_spec, dict(spec_dict), manifest, debug_output="upstream")

    with_flag = sp.method_entry(manifest, "APD-MVS")
    without = sp.method_entry(manifest, "ACMM")

    state = rn.debug_output_state(off, with_flag)
    assert state == {
        "setting": "off",
        "method_has_flag": True,
        "flag_passed": True,
        "normalization_in_force": True,
        "reason": state["reason"],
    }
    assert "skipped" in state["reason"]

    state = rn.debug_output_state(upstream, with_flag)
    assert state["method_has_flag"] and not state["flag_passed"]
    assert not state["normalization_in_force"]
    assert "NOT in force" in state["reason"]

    for spec in (off, upstream):
        state = rn.debug_output_state(spec, without)
        assert not state["method_has_flag"]
        assert not state["normalization_in_force"]
        assert "carries no --no-debug-output" in state["reason"]


def test_a_method_without_the_flag_is_invoked_identically_in_both_arms(
    write_spec, spec_dict, manifest
):
    off = _spec(write_spec, spec_dict, manifest, methods=["ACMM"])
    upstream = _spec(
        write_spec, dict(spec_dict), manifest, debug_output="upstream", methods=["ACMM"]
    )
    entry = sp.method_entry(manifest, "ACMM")
    run = sp.expand(off)[0]
    assert rn.resolve_invocation(off, entry, run) == rn.resolve_invocation(upstream, entry, run)


def test_phase_timer_false_gives_the_measured_container_no_trace_env(
    write_spec, spec_dict, manifest
):
    on = _spec(write_spec, spec_dict, manifest, methods=["ACMM"])
    assert rn.method_env(on) == {
        "MVS_BENCH_PHASES": "1",
        "MVS_BENCH_FILE": "/out/phases.txt",
    }

    off = _spec(write_spec, dict(spec_dict), manifest, methods=["ACMM"], phase_timer=False)
    assert rn.method_env(off) == {}
    command = rn.plan_commands(off, sp.method_entry(manifest, "ACMM"), sp.expand(off)[0])[
        "measured"
    ]["docker"]
    assert not any("MVS_BENCH" in token for token in command)


def test_method_env_reaches_the_measured_container(write_spec, spec_dict, manifest):
    # R-TIM-08: CUMVS's two unsynchronised launches are attributed to the
    # following span unless MVS_BENCH_SYNC is set.
    spec = _spec(
        write_spec,
        spec_dict,
        manifest,
        methods=["CUMVS"],
        method_env={"MVS_BENCH_SYNC": "1"},
    )
    entry = sp.method_entry(manifest, "CUMVS")
    plan = rn.plan_commands(spec, entry, sp.expand(spec)[0])
    assert plan["measured"]["env"]["MVS_BENCH_SYNC"] == "1"
    command = plan["measured"]["docker"]
    assert "MVS_BENCH_SYNC=1" in command
    # Passed with -e, before the image name, or the runtime would not see it.
    assert command.index("MVS_BENCH_SYNC=1") < command.index(spec.image)
    assert command[command.index("MVS_BENCH_SYNC=1") - 1] == "-e"


def test_missing_phase_trace_is_not_a_failure(tmp_path):
    from bench import phases as ph

    trace = ph.parse_file(tmp_path / "phases.txt")
    assert trace.spans == []
    assert ph.totals(trace) == []
    # The absence is recorded as a parser note, and classification never reads it.
    assert trace.errors and "no phase trace was written" in trace.errors[0]
    status, _ = rn.classify(0, False, "", False, {})
    assert status == "ok"
