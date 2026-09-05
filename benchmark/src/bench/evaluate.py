"""ETH3D multi-view evaluation of a run's point cloud.

R-QUA-01, R-QUA-03, R-QUA-04, R-ART-03, R-STO-03.

The evaluator is the one built into the batch image from a pinned commit
(R-QUA-02); it is invoked in a container from that same image so that the
scores of a campaign cannot depend on whatever binary happens to be on the
host. The invocation and the stdout parsing follow the v0 harness
(`src/eval/eval.py`), which is where the argument names were established.
"""

import hashlib
import re
import subprocess
from pathlib import Path


class EvaluationFailed(Exception):
    def __init__(self, message, stdout="", stderr="", argv=None):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.argv = argv or []


def _floats(line, prefix):
    return [float(x) for x in line.split(prefix, 1)[1].strip().split()]


def parse_output(stdout, requested_tolerances):
    """Tolerances, accuracies, completenesses and F1 scores from the evaluator.

    The evaluator prints one value per tolerance on each of three lines, in the
    order it was given the tolerances; it prints the tolerance line itself only
    in some versions, so the requested list is the fallback.
    """
    tolerances = None
    accuracies = completenesses = f1_scores = None
    for line in stdout.splitlines():
        if "Tolerances:" in line:
            tolerances = _floats(line, "Tolerances:")
        elif "Accuracies:" in line:
            accuracies = _floats(line, "Accuracies:")
        elif "Completenesses:" in line:
            completenesses = _floats(line, "Completenesses:")
        elif "F1-scores:" in line:
            f1_scores = _floats(line, "F1-scores:")
    if accuracies is None or completenesses is None or f1_scores is None:
        raise EvaluationFailed("the evaluator printed no scores", stdout=stdout)
    if tolerances is None:
        tolerances = list(requested_tolerances)
    if not (len(tolerances) == len(accuracies) == len(completenesses) == len(f1_scores)):
        raise EvaluationFailed(
            "the evaluator returned a different number of scores than tolerances",
            stdout=stdout,
        )
    return [
        {
            "tolerance": tolerances[i],
            "accuracy": accuracies[i],
            "completeness": completenesses[i],
            "f1": f1_scores[i],
        }
        for i in range(len(tolerances))
    ]


def ply_point_count(path):
    """Vertex count from the PLY header; None when the header is unreadable."""
    try:
        with open(path, "rb") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                text = line.decode("ascii", errors="replace").strip()
                match = re.match(r"^element vertex (\d+)$", text)
                if match:
                    return int(match.group(1))
                if text == "end_header":
                    break
    except OSError:
        return None
    return None


def sha256_file(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def evaluation_argv(spec, ply_path, scene, width, docker="docker"):
    """`docker run` argv for the evaluator, mounting the cloud and the scan."""
    # Paths are used as the operator wrote them: resolving symlinks here and
    # not in the run mounts would put two different spellings of one directory
    # into the same record.
    ply_path = Path(ply_path).absolute()
    scene_dir = spec.scene_dir(scene, width)
    relative_mlp = spec.layout["ground_truth_mlp"].format(scene=scene, width=width)
    return [
        docker,
        "run",
        "--rm",
        "-v",
        f"{ply_path.parent}:/ply:ro",
        "-v",
        f"{scene_dir}:/gt:ro",
        "--entrypoint",
        spec.container["evaluator"],
        spec.image,
        "--reconstruction_ply_path",
        f"/ply/{ply_path.name}",
        "--ground_truth_mlp_path",
        f"/gt/{relative_mlp}",
        "--tolerances",
        ",".join(str(t) for t in spec.tolerances),
    ]


def evaluate(spec, ply_path, scene, width, log_dir=None, docker="docker", timeout=None):
    """Run the evaluator; return the score rows. Raises EvaluationFailed."""
    argv = evaluation_argv(spec, ply_path, scene, width, docker=docker)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.SubprocessError as exc:
        raise EvaluationFailed(str(exc), argv=argv) from exc
    if log_dir is not None:
        Path(log_dir, "eval.stdout.log").write_text(proc.stdout)
        Path(log_dir, "eval.stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise EvaluationFailed(
            f"the evaluator exited {proc.returncode}",
            stdout=proc.stdout,
            stderr=proc.stderr,
            argv=argv,
        )
    return parse_output(proc.stdout, spec.tolerances)
