import subprocess
import argparse
import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "eth3d"))
    from eth3d import prepare_datasets, ETH3D_DATASETS_TRAINING, ETH3D_DATASETS_TEST
except ImportError:
    print("Error: eth3d module not found. Please ensure it is installed correctly.")
    sys.exit(1)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--datasets",
    nargs="+",
    default=[],
    choices=ETH3D_DATASETS_TRAINING + ETH3D_DATASETS_TEST,
    help="List of datasets to evaluate e.g. `--datasets courtyard`",
)
parser.add_argument("--width", type=int, default=None)

args = parser.parse_args()

# 1. download datasets
prepare_datasets(args.datasets, "evaluation", args.width)

for dataset in args.datasets:
    # 1.1 convert colmap to mvsnet format
    subprocess.check_call(
        [
            sys.executable,
            "colmap2mvsnet_acm_perf.py",
            "--dense_folder",
            f"evaluation/{dataset}/{dataset}_dslr_undistorted/{dataset}",
            "--save_folder",
            f"evaluation_mvsnet_{dataset}",
        ]
    )
    # 2. will run mvs matrix, currently only ACMH
    subprocess.check_call(
        [
            "ACMH/build/ACMH",
            f"evaluation_mvsnet_{dataset}",
        ]
    )
    # 3. run evaluation
    subprocess.check_call(
        [
            "eth3d/multi-view-evaluation/build/ETH3DMultiViewEvaluation",
            "--reconstruction_ply_path",
            f"evaluation_mvsnet_{dataset}/ACMH/ACMH_model.ply",
            "--ground_truth_mlp_path",
            f"evaluation/{dataset}/{dataset}_dslr_scan_eval/{dataset}/dslr_scan_eval/scan_alignment.mlp",
            "--tolerances",
            "0.01,0.02,0.05,0.1,0.2,0.5",
        ]
    )
