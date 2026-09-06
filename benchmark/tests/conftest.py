import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deviations import manifest as mf  # noqa: E402

MANIFEST_PATH = mf.default_manifest_path()


@pytest.fixture(scope="session")
def manifest():
    return mf.load(MANIFEST_PATH)


@pytest.fixture
def spec_dict(tmp_path):
    return {
        "version": 1,
        "campaign": "unit-test",
        "image": "sota-deps:latest",
        "methods": ["ACMH", "ACMM"],
        "scenes": ["courtyard"],
        "widths": [1600, 3200],
        "configurations": ["author"],
        "repeats": 2,
        "timeout_seconds": 600,
        "gpu_index": 0,
        "order_seed": 7,
        "paths": {
            "dataset_root": str(tmp_path / "datasets"),
            "results_root": str(tmp_path / "results"),
            "work_root": str(tmp_path / "work"),
            "artifacts_root": str(tmp_path / "clouds"),
        },
    }


@pytest.fixture
def write_spec(tmp_path):
    def _write(data, name="spec.yaml"):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write


def braces_balance(text):
    """(open, close) counts of unescaped braces, comments removed.

    No TeX distribution is installed on the harness host, so compiling a
    generated table cannot be part of the test suite; a brace count catches the
    failure mode such a table actually has -- an unescaped cell running into
    the surrounding markup.
    """
    stripped = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    stripped = stripped.replace("\\\\", "").replace("\\{", "").replace("\\}", "")
    return stripped.count("{"), stripped.count("}")
