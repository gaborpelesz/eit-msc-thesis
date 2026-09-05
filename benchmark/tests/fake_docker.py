#!/usr/bin/env python3
"""A stand-in for the docker CLI, so the runner's container path is testable.

It understands only what `bench.runner` and `bench.evaluate` invoke: `run`
(detached and not), `logs`, `wait`, `inspect`, `kill` and `rm`. What the
"container" does is driven by environment variables set by the test:

    FAKE_DOCKER_STATE       directory for the fake container state
    FAKE_DOCKER_OUTPUT_PLY  point cloud to create under the prepared dataset
    FAKE_DOCKER_EXIT        exit code of the measured process (default 0)
    FAKE_DOCKER_STDERR      text the measured process writes to stderr
    FAKE_DOCKER_NO_PHASES   set to skip writing a phase trace
    FAKE_DOCKER_CONVERT_EXIT exit code of the converter (default 0)
    FAKE_DOCKER_EMPTY_PLY   write a header-only point cloud (0 vertices)
"""

import json
import os
import sys
from pathlib import Path

STATE = Path(os.environ.get("FAKE_DOCKER_STATE", "/tmp/fake-docker"))
CID = "0123456789abcdef"

PHASE_TRACE = """PHASE problem_list BEGIN 0
PHASE problem_list END 500000000
PHASE image_pass BEGIN 1000000000 pass=photometric
PHASE patchmatch BEGIN 1100000000
PHASE patchmatch END 1900000000
PHASE image_pass END 2000000000 pass=photometric
PHASE fusion BEGIN 2100000000
PHASE fusion END 2400000000
"""


def mounts(argv):
    found = {}
    for index, token in enumerate(argv):
        if token == "-v":
            parts = argv[index + 1].split(":")
            found[parts[1]] = Path(parts[0])
    return found


def do_run(argv):
    entrypoint = argv[argv.index("--entrypoint") + 1]
    mounted = mounts(argv)

    if entrypoint.endswith("ETH3DMultiViewEvaluation"):
        print("Tolerances: 0.01 0.02 0.05 0.1 0.2")
        print("Accuracies: 0.50 0.60 0.70 0.80 0.90")
        print("Completenesses: 0.40 0.50 0.60 0.70 0.80")
        print("F1-scores: 0.44 0.55 0.65 0.74 0.85")
        return 0

    prepared = mounted.get("/work/prepared")
    if "-d" not in argv:  # the converter runs in the foreground
        convert_exit = int(os.environ.get("FAKE_DOCKER_CONVERT_EXIT", "0"))
        if convert_exit:
            # A half-written prepared directory, as HPM-MVS's converter leaves
            # when it dies partway through.
            prepared.mkdir(parents=True, exist_ok=True)
            print("Traceback: fake converter died", file=sys.stderr)
            return convert_exit
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "pair.txt").write_text("2\n0\n1 1 100.0\n1\n1 0 100.0\n")
        print("fake converter wrote pair.txt")
        return 0

    out = mounted.get("/out")
    if not os.environ.get("FAKE_DOCKER_NO_PHASES"):
        (out / "phases.txt").write_text(PHASE_TRACE)
    output_ply = os.environ.get("FAKE_DOCKER_OUTPUT_PLY")
    if output_ply:
        target = prepared / output_ply
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.environ.get("FAKE_DOCKER_EMPTY_PLY"):
            target.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        else:
            target.write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\nend_header\n1 2 3\n"
            )
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "stdout").write_text("fake method stdout\n")
    (STATE / "stderr").write_text(os.environ.get("FAKE_DOCKER_STDERR", ""))
    (STATE / "state.json").write_text(
        json.dumps(
            {
                "exit_code": int(os.environ.get("FAKE_DOCKER_EXIT", "0")),
                "oom_killed": os.environ.get("FAKE_DOCKER_OOM", "false"),
                "started_at": "2026-09-05T18:00:00.123456789Z",
                "finished_at": "2026-09-05T18:00:12.223456789Z",
                "pid": 4000000,
            }
        )
    )
    print(CID)
    return 0


def state():
    return json.loads((STATE / "state.json").read_text())


def main(argv):
    command = argv[0]
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "calls.log", "a") as log:
        log.write(" ".join(argv) + "\n")
    if command == "run":
        return do_run(argv)
    if command == "logs":
        sys.stdout.write((STATE / "stdout").read_text())
        sys.stderr.write((STATE / "stderr").read_text())
        return 0
    if command == "wait":
        print(state()["exit_code"])
        return 0
    if command == "inspect":
        template = argv[argv.index("-f") + 1]
        values = {
            "{{.State.Pid}}": state()["pid"],
            "{{.State.OOMKilled}}": state()["oom_killed"],
            "{{.State.StartedAt}}": state()["started_at"],
            "{{.State.FinishedAt}}": state()["finished_at"],
            "{{.State.ExitCode}}": state()["exit_code"],
        }
        print(values.get(template, ""))
        return 0
    if command in ("rm", "kill"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
