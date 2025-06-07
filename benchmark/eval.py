import subprocess
import argparse
import sys
import os
import time
import shutil

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "eth3d"))
    from eth3d import prepare_datasets, ETH3D_DATASETS_TRAINING, ETH3D_DATASETS_TEST  # type: ignore
except ImportError:
    print("Error: eth3d module not found. Please ensure it is installed correctly.")
    sys.exit(1)


class Method:
    name: str
    exe: str

    def __init__(self, name: str, exe: str, outply_path: str, padding: bool = False):
        self.name = name
        self.exe = exe
        self.prepared_dataset_dir: str | None = None
        self.outply_path = outply_path
        self.padding = padding

        assert os.path.exists(self.exe), f"Executable {self.exe} does not exist"

    def prepare(self, dataset_dir: str, dataset_name):
        self.prepared_dataset_dir = dataset_dir.rstrip("/") + "_mvsnet"
        cmd = [
            sys.executable,
            "colmap2mvsnet_acm_perf.py",
            "--dense_folder",
            os.path.join(
                dataset_dir, f"{dataset_name}_dslr_undistorted/{dataset_name}"
            ),
            "--save_folder",
            self.prepared_dataset_dir,
        ]
        if self.padding:
            cmd.append("--padding")
        subprocess.check_call(cmd)


    def run(self) -> bool:
        if not self.prepared_dataset_dir:
            raise ValueError("Dataset not prepared")
        cmd = [
            self.exe,
            self.prepared_dataset_dir,
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Running cmd: {' '.join(cmd)}", file=sys.stderr)
            return False
        return True

    def get_reconstructed_ply_path(self):
        if not self.prepared_dataset_dir:
            raise ValueError("Dataset not prepared")
        return os.path.join(self.prepared_dataset_dir, self.outply_path)


methods = [
    Method("ACMH", "ACMH/build/ACMH", "ACMH/ACMH_model.ply"),
    Method("ACMM", "ACMM/build/ACMM", "ACMM/ACMM_model.ply"),
    Method("ACMP", "ACMP/build/ACMP", "ACMP/ACMP_model.ply"),
    Method("ACMMP", "ACMMP/build/ACMMP", "ACMMP/ACMMP_model.ply"),
    Method("HPM", "HPM-MVS/HPM-MVS/build/HPM", "HPM/HPM_model.ply"),
    Method("APD", "APD-MVS/build/APD", "APD/APD_model.ply", padding=True),
    # runs into segfault in CUDA
    # Method("HPM++", "HPM-MVS_plusplus/build/HPM-MVS_plusplus", "doesn't exists"),
]


class EvaluationResult:
    dataset: str
    method: str
    time: float
    tolerances: list[float]
    accuracies: list[float]
    completenesses: list[float]
    f1_scores: list[float]

    def __init__(
        self, dataset, method, time, tolerances, accuracies, completenesses, f1_scores
    ):
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
            output.append(
                f"{self.tolerances[i]:>10.3f} | {self.accuracies[i]:>10.3f} | {self.completenesses[i]:>12.3f} | {self.f1_scores[i]:>10.3f}"
            )
        output.append(f"{'='*50}\n")
        return "\n".join(output)

    def write_to_sqlite(self, db_path: str):
        """Write the evaluation results to an SQLite database.

        Args:
            db_path: Path to the SQLite database file
        """
        import sqlite3

        # Create table if it doesn't exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset TEXT,
                    method TEXT,
                    time REAL,
                    tolerance REAL,
                    accuracy REAL,
                    completeness REAL,
                    f1_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Insert the results
            for i in range(len(self.tolerances)):
                cursor.execute(
                    """
                    INSERT INTO evaluation_results (
                        dataset, method, time,
                        tolerance,
                        accuracy,
                        completeness,
                        f1_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.dataset,
                        self.method,
                        self.time,
                        self.tolerances[i],
                        self.accuracies[i],
                        self.completenesses[i],
                        self.f1_scores[i],
                    ),
                )
            conn.commit()


def parse_stdout_into_eval_result(
    stdout: str,
) -> tuple[list[float], list[float], list[float]]:
    find_and_parse_to_vals = lambda line, s: [
        float(x) for x in line.lstrip(s).strip().split(" ")
    ]
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
    assert accuracies and completenesses and f1_scores
    return accuracies, completenesses, f1_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        default=[],
        choices=ETH3D_DATASETS_TRAINING + ETH3D_DATASETS_TEST,
        help="List of datasets to evaluate e.g. `--datasets courtyard`",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[m.name for m in methods],
        choices=[m.name for m in methods],
        help="List of methods to evaluate e.g. `--methods ACMH ACMM`. By default, all methods are evaluated.",
    )
    parser.add_argument("--output", type=str, default="evaluation")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--tolerances", type=str, default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--clean-db", action="store_true")
    args = parser.parse_args()

    if args.clean_db:
        if os.path.exists(os.path.join(args.output, "evaluation.db")):
            os.remove(os.path.join(args.output, "evaluation.db"))
            print("Database cleaned")

    tolerances = [float(x) for x in args.tolerances.split(",")]

    # 1. download datasets
    prepare_datasets(args.datasets, os.path.join(args.output, "datasets"), args.width)

    log_file = f"{args.output}/evaluation_results.log"
    if os.path.exists(log_file):
        shutil.move(log_file, f"{log_file}_{time.strftime('%Y%m%d_%H%M%S')}")
    with open(log_file, "w") as f:
        f.write(
            "Evaluation Results\n"
            f"{'-' * 50}\n"
            f"Datasets: {', '.join(args.datasets)}\n"
            f"Methods: {', '.join(args.methods)}\n"
            f"Tolerances: {args.tolerances}\n"
            f"Width: {args.width}\n"
            f"{'-' * 50}\n"
        )

    selected_methods = [m for m in methods if m.name in args.methods]

    failed_methods = []
    for dataset in args.datasets:
        dataset_dir = os.path.join(args.output, "datasets", dataset)
        if args.width:
            dataset_dir += f"_{args.width}"
        for method in selected_methods:
            method.prepare(dataset_dir, dataset)

            t0 = time.time()
            if not method.run():
                failed_methods.append((dataset, method.name))
                continue
            method_time = time.time() - t0
            print(f"Method {method.name} took {method_time} seconds")

            print("Running evaluation")
            res = subprocess.run(
                [
                    "eth3d/multi-view-evaluation/build/ETH3DMultiViewEvaluation",
                    "--reconstruction_ply_path",
                    method.get_reconstructed_ply_path(),
                    "--ground_truth_mlp_path",
                    os.path.join(
                        dataset_dir,
                        f"{dataset}_dslr_scan_eval/{dataset}/dslr_scan_eval/scan_alignment.mlp",
                    ),
                    "--tolerances",
                    args.tolerances,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            print(res.stdout)

            accuracies, completenesses, f1_scores = parse_stdout_into_eval_result(
                res.stdout
            )
            result = EvaluationResult(
                dataset,
                method.name,
                method_time,
                tolerances,
                accuracies,
                completenesses,
                f1_scores,
            )
            result.write_to_sqlite(os.path.join(args.output, "evaluation.db"))

            with open(log_file, "a") as f:
                f.write(str(result))

    if failed_methods:
        print("=" * 50)
        print("Failed methods:")
        for m in failed_methods:
            print(f" - {m[0]} | {m[1]}")
        print("=" * 50)


if __name__ == "__main__":
    main()
