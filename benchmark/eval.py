import subprocess
import argparse
import sys
import os
import time

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "eth3d"))
    from eth3d import prepare_datasets, ETH3D_DATASETS_TRAINING, ETH3D_DATASETS_TEST
except ImportError:
    print("Error: eth3d module not found. Please ensure it is installed correctly.")
    sys.exit(1)


class EvaluationResult:
    dataset: str
    method: str
    time: float
    tolerances: list[float]
    accuracies: list[float]
    completenesses: list[float]
    f1_scores: list[float]

    def __init__(self, dataset, method, time, tolerances, accuracies, completenesses, f1_scores):
        self.dataset = dataset
        self.method = method
        self.time = time
        self.tolerances = tolerances
        self.accuracies = accuracies
        self.completenesses = completenesses
        self.f1_scores = f1_scores

    def __str__(self):
        output = [
            f"\n{'='*50}",
            f"Dataset: {self.dataset}",
            f"Method: {self.method}",
            f"Runtime: {self.time:.2f} seconds",
            f"{'='*50}",
            "\nMetrics:",
                f"{'Tolerance':>10} | {'Accuracy':>10} | {'Completeness':>12} | {'F1 Score':>10}",
                f"{'-'*10} | {'-'*10} | {'-'*12} | {'-'*10}",
        ]
        for i in range(len(self.accuracies)):
            output.append(f"{self.tolerances[i]:>10.3f} | {self.accuracies[i]:>10.3f} | {self.completenesses[i]:>12.3f} | {self.f1_scores[i]:>10.3f}")
        output.append(f"{'='*50}\n")
        return "\n".join(output)


def parse_stdout_into_eval_result(dataset: str, method: str, time: float, tolerances: list[float], stdout: str) -> EvaluationResult:
    find_and_parse_to_vals = lambda line,s: [float(x) for x in line.lstrip(s).strip().split(" ")]
    accuracies = None
    completenesses = None
    f1_scores = None
    for line in stdout.split("\n"):
        if "Accuracies:" in line:
            accuracies = find_and_parse_to_vals(line, "Accuracies:")
        elif "Completenesses:" in line:
            completenesses = find_and_parse_to_vals(line, "Completenesses:")
        elif "F1-scores:" in line:
            f1_scores = find_and_parse_to_vals(line, "F1-scores:")
    assert accuracies and len(accuracies) == len(tolerances)
    assert completenesses and len(completenesses) == len(tolerances)
    assert f1_scores and len(f1_scores) == len(tolerances)
    return EvaluationResult(dataset, method, time, tolerances, accuracies, completenesses, f1_scores)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        choices=ETH3D_DATASETS_TRAINING + ETH3D_DATASETS_TEST,
        help="List of datasets to evaluate e.g. `--datasets courtyard`",
    )
    parser.add_argument("--output", type=str, default="evaluation")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--tolerances", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5")
    args = parser.parse_args()

    tolerances = [float(x) for x in args.tolerances.split(",")]

    # 1. download datasets
    prepare_datasets(args.datasets, os.path.join(args.output, "datasets"), args.width)

    results: list[EvaluationResult] = []

    for dataset in args.datasets:
        # 1.1 convert colmap to mvsnet format
        subprocess.check_call(
            [
                sys.executable,
                "colmap2mvsnet_acm_perf.py",
                "--dense_folder",
                f"{args.output}/datasets/{dataset}/{dataset}_dslr_undistorted/{dataset}",
                "--save_folder",
                f"{args.output}/datasets/{dataset}_mvsnet",
            ]
        )
        # 2. will run mvs matrix, currently only ACMH
        method = "ACMH"
        t0 = time.time()
        subprocess.check_call(
            [
                "ACMH/build/ACMH",
                f"{args.output}/datasets/{dataset}_mvsnet",
            ]
        )
        # 3. run evaluation
        method_time = time.time() - t0
        print(f"Method {method} took {method_time} seconds")
        print("running evaluation")
        result = subprocess.run(
            [
                "eth3d/multi-view-evaluation/build/ETH3DMultiViewEvaluation",
                "--reconstruction_ply_path",
                f"{args.output}/datasets/{dataset}_mvsnet/ACMH/ACMH_model.ply",
                "--ground_truth_mlp_path",
                f"{args.output}/datasets/{dataset}/{dataset}_dslr_scan_eval/{dataset}/dslr_scan_eval/scan_alignment.mlp",
                "--tolerances",
                args.tolerances,
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)

        results.append(parse_stdout_into_eval_result(dataset, method, method_time, tolerances, result.stdout))

    log_file = f"{args.output}/evaluation_results.log"
    with open(log_file, 'w') as f:
        f.writelines(str(result) for result in results)

if __name__ == "__main__":
    main()
