import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--datasets",
    nargs="+",
    default=[],
    type=str,
    help="List of datasets to evaluate e.g. `--datasets courtyard`",
)
parser.add_argument("--ground_truth_mlp_path", type=str, required=True)
parser.add_argument("--reconstruction_ply_path", type=str, required=True)
parser.add_argument("--ground_truth_mlp_path", type=str, required=True)

args = parser.parse_args()

# 1. download datasets
# 2. run matrix
# 3. run evaluation



subprocess.call(
    [
        "eth3d/multi-view-evaluation/build/ETH3DMultiViewEvaluation",
        "--reconstruction_ply_path",
        args.reconstruction_ply_path,
        "--ground_truth_mlp_path",
        args.ground_truth_mlp_path,
        "--tolerances",
        "0.01,0.02,0.05,0.1,0.2,0.5",
    ]
)
