import subprocess
import argparse
import sys
import os
import time
import shutil
from pathlib import Path

try:
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
        self.prepared_dataset_dir = dataset_dir.rstrip("/") + f"_{self.name}"
        cmd = [
            "colmap2mvsnet_acm_perf",
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
            print(f"[ERROR] Running cmd: {' '.join([str(p) for p in cmd])}", file=sys.stderr)
            return False
        return True

    def get_reconstructed_ply_path(self):
        if not self.prepared_dataset_dir:
            raise ValueError("Dataset not prepared")
        return os.path.join(self.prepared_dataset_dir, self.outply_path)


class CumvsMethod(Method):
    def __init__(self, name: str, exe: str, outply_path: str, padding: bool = False):
        super().__init__(name, exe, outply_path, padding)

    def prepare(self, dataset_dir: str, dataset_name):
        self.prepared_dataset_dir = dataset_dir.rstrip("/") + f"_{self.name}"
        app_initialize_ETH3D = os.path.join(
            os.path.dirname(os.path.abspath(self.exe)), "app_initialize_ETH3D"
        )
        subprocess.check_call(
            [
                app_initialize_ETH3D,
                os.path.join(
                    dataset_dir, f"{dataset_name}_dslr_undistorted/{dataset_name}"
                ),
                f"--output-directory={self.prepared_dataset_dir}",
            ]
        )

    def run(self) -> bool:
        if not self.prepared_dataset_dir:
            raise ValueError("Dataset not prepared")
        cmd = [
            self.exe,
            self.prepared_dataset_dir,
            f"--output-directory={self.prepared_dataset_dir}/CUMVS",
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Running cmd: {' '.join(cmd)}", file=sys.stderr)
            return False
        return True


mvs_method_root = Path(
    os.getenv("SOTA_MVS_METHOD_ROOT", Path(__file__).parent.parent.parent)
)

methods = [
    Method("ACMH", mvs_method_root / "ACMH/build/ACMH", "ACMH/ACMH_model.ply"),
    Method("ACMM", mvs_method_root / "ACMM/build/ACMM", "ACMM/ACMM_model.ply"),
    Method("ACMP", mvs_method_root / "ACMP/build/ACMP", "ACMP/ACMP_model.ply"),
    Method("ACMMP", mvs_method_root / "ACMMP/build/ACMMP", "ACMMP/ACMMP_model.ply"),
    Method("HPM", mvs_method_root / "HPM-MVS/HPM-MVS/build/HPM", "HPM/HPM_model.ply"),
    Method("APD", mvs_method_root / "APD-MVS/build/APD", "APD/APD.ply", padding=True),
    # can run into segfault in CUDA because of memory leak, but possible to run with lower scale
    Method(
        "HPM++",
        mvs_method_root / "HPM-MVS_plusplus/build/HPM-MVS_plusplus",
        "HPM_MVS_plusplus/HPM_MVS_plusplus.ply",
    ),
    Method("MP", mvs_method_root / "MP-MVS/build/MPMVS", "MP_MVS/MPMVS_model.ply"),
    CumvsMethod(
        "CUMVS",
        mvs_method_root / "cuda-multi-view-stereo/build/samples/app_patch_match_mvs",
        "CUMVS/point_cloud_dense.ply",
    ),
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
            f"\n{'=' * 50}",
            f"Dataset: {self.dataset}",
            f"Method: {self.method}",
            f"Runtime: {self.time:.2f} seconds",
            f"{'=' * 50}",
            "\nMetrics:",
            f"{'Tolerance':>10} | {'Accuracy':>10} | {'Completeness':>12} | {'F1 Score':>10}",
            f"{'-' * 10} | {'-' * 10} | {'-' * 12} | {'-' * 10}",
        ]
        for i in range(len(self.accuracies)):
            output.append(
                f"{self.tolerances[i]:>10.3f} | {self.accuracies[i]:>10.3f} | {self.completenesses[i]:>12.3f} | {self.f1_scores[i]:>10.3f}"
            )
        output.append(f"{'=' * 50}\n")
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
    def find_and_parse_to_vals(line, s):
        return [float(x) for x in line.lstrip(s).strip().split(" ")]

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

    # 1. download and prepare all requested datasets
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
            cmd = [
                "ETH3DMultiViewEvaluation",
                "--reconstruction_ply_path",
                method.get_reconstructed_ply_path(),
                "--ground_truth_mlp_path",
                os.path.join(
                    dataset_dir,
                    f"{dataset}_dslr_scan_eval/{dataset}/dslr_scan_eval/scan_alignment.mlp",
                ),
                "--tolerances",
                args.tolerances,
            ]
            try:
                res = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Running cmd failed: {' '.join(cmd)}", file=sys.stderr)
                print(e.stderr)
                raise

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
