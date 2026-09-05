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
    assert f"{spec.raw_scene_dir('courtyard', 3200)}:/data:ro" in command
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
