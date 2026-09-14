"""The image build manifest: what the batch image was built from (D24.1).

The image carries a copy of every fork and of `src/eval`, and it has no `.git`.
A run record's `fork_sha`, `deviations.json` and converter hashes are read from
the *working tree*, so an image built before the working tree moved makes every
record name code that did not run -- which happened once already (M-006, "Open,
after this session" 1).

`bench image-manifest` writes this file from the working tree; the Dockerfile
COPYs it into `/sota/image-manifest.json`; `bench run` reads it back out of the
image and refuses to start when the two differ. The manifest therefore has to
be regenerated before every `docker build`.

Paths are recorded relative to the superproject root: the manifest is compared
between the machine that built the image and the machine that runs it, and an
absolute path would differ between two clones of the same tree while naming the
same file.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from deviations import manifest as mf
from deviations.manifest import SHARED_CONVERTER

SCHEMA_VERSION = 1

DOCKERFILE = "benchmark/Dockerfile"
PYPROJECT = "benchmark/pyproject.toml"
UV_LOCK = "benchmark/uv.lock"
EVALUATOR_SUBMODULE = "benchmark/eth3d/multi-view-evaluation"

DEFAULT_MANIFEST_NAME = "image-manifest.json"
CONTAINER_PATH = "/sota/image-manifest.json"

# CUMVS preprocesses with a compiled initializer rather than a converter
# script, and that binary exists only inside the image. Its source is what
# identifies it in the working tree; methods.yaml names the executable, not the
# translation unit, so the mapping lives here.
INITIALIZER_SOURCES = {"CUMVS": "samples/app_initialize_ETH3D.cpp"}


def sha256_file(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_record(repo_root, relative):
    path = Path(repo_root) / relative
    try:
        digest = sha256_file(path)
    except OSError:
        digest = None
    return {"path": str(relative), "sha256": digest}


def _git(args, cwd):
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _fork_state(fork_dir):
    """(HEAD sha, dirty) for a fork, or (None, None) when git cannot answer.

    `dirty` covers tracked modifications and untracked files alike: both change
    what `docker build` copies into the image, and neither is visible in the
    SHA the record quotes.
    """
    head = _git(["rev-parse", "HEAD"], fork_dir)
    if head is None:
        return None, None
    status = _git(["status", "--porcelain"], fork_dir)
    return head.strip(), None if status is None else bool(status.strip())


def initializer_source(entry):
    """Repo-relative source of a method's initializer binary, or None."""
    source = INITIALIZER_SOURCES.get(entry["name"])
    return f"{entry['path']}/{source}" if source else None


def converter_identity(repo_root, entry):
    """What identifies this method's preprocessing step in the working tree.

    `kind: initializer_source` means the image builds a binary from this source
    and the source is the only thing the host can hash (M-006 open 3).
    """
    if entry.get("converter"):
        record = _file_record(repo_root, f"{entry['path']}/{entry['converter']}")
        return {"kind": "converter", **record}
    if mf.converter_source(entry) == "shared":
        # Recorded per method as well as once at the top level: a method that
        # consumes the shared converter in every configuration has no other
        # preprocessing identity, and `kind: none` would claim it has none.
        return {"kind": "shared", **_file_record(repo_root, SHARED_CONVERTER)}
    source = initializer_source(entry)
    if source:
        return {"kind": "initializer_source", **_file_record(repo_root, source)}
    return {"kind": "none", "path": None, "sha256": None}


def compute(repo_root, manifest, now=None):
    """The build manifest of the image that *would* be built from this tree."""
    repo_root = Path(repo_root)
    methods = []
    for entry in sorted(manifest.get("methods") or [], key=lambda e: e["name"]):
        head, dirty = _fork_state(repo_root / entry["path"])
        methods.append(
            {
                "name": entry["name"],
                "path": entry["path"],
                "status": entry.get("status") or "active",
                "provenance": entry.get("provenance"),
                "fork_sha": head,
                "dirty": dirty,
                "converter": converter_identity(repo_root, entry),
            }
        )
    evaluator, _ = _fork_state(repo_root / EVALUATOR_SUBMODULE)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "dockerfile": _file_record(repo_root, DOCKERFILE),
        "pyproject": _file_record(repo_root, PYPROJECT),
        "uv_lock": _file_record(repo_root, UV_LOCK),
        "shared_converter": _file_record(repo_root, SHARED_CONVERTER),
        "evaluator": {"path": EVALUATOR_SUBMODULE, "sha": evaluator},
        "methods": methods,
    }


def canonical(document):
    """The comparable part of a manifest: everything but the timestamp."""
    return {k: v for k, v in (document or {}).items() if k != "generated_at"}


def digest(document):
    """A stable sha256 over the comparable part, for the run record."""
    if document is None:
        return None
    return hashlib.sha256(
        json.dumps(canonical(document), sort_keys=True, default=str).encode()
    ).hexdigest()


def _flatten(value, prefix=""):
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            # A list of named records is addressed by name, so a reordered or
            # shortened list does not report every entry as different.
            label = item["name"] if isinstance(item, dict) and "name" in item else str(index)
            flat.update(_flatten(item, f"{prefix}[{label}]"))
    else:
        flat[prefix] = value
    return flat


_MISSING = object()


def compare(image_document, working_document):
    """Differences between the image's manifest and the working tree's.

    Returns a list of `{key, image, working}`, one per differing leaf, sorted
    by key. `generated_at` is excluded: rebuilding the same tree at a different
    time is not a difference in what was built.
    """
    image_flat = _flatten(canonical(image_document))
    working_flat = _flatten(canonical(working_document))
    differences = []
    for key in sorted(set(image_flat) | set(working_flat)):
        was = image_flat.get(key, _MISSING)
        now = working_flat.get(key, _MISSING)
        if was != now:
            differences.append(
                {
                    "key": key,
                    "image": None if was is _MISSING else was,
                    "working": None if now is _MISSING else now,
                }
            )
    return differences


def write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text)
    return text
