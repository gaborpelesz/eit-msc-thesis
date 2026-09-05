"""Reading and shape-checking `benchmark/methods/methods.yaml`."""

import os
import re
import subprocess
from pathlib import Path

import yaml

# CLAUDE.md, "Deviation policy" -> Classes.
DEVIATION_CLASSES = (
    "build",
    "compat",
    "interface",
    "instrumentation",
    "bugfix",
    "normalization",
    "optimization",
)

# `optimization` is reserved for the author's own reimplementation; a fork of
# someone else's published code may never carry it.
CLASSES_ALLOWED_IN_FOREIGN_FORK = tuple(c for c in DEVIATION_CLASSES if c != "optimization")

PROVENANCES = ("published-reference", "third-party-optimization", "own")

AFFECTS_VALUES = ("none", "timing", "memory", "quality", "io")

REQUIRED_METHOD_FIELDS = (
    "name",
    "path",
    "upstream",
    "upstream_default_branch",
    "upstream_base",
    "provenance",
    "paper",
    "source_subdir",
    "executable",
    "invocation",
    "converter",
    "output_ply",
    "requires_equal_image_sizes",
    "passes",
    "known_limitations",
    "harness_deviations",
    "fork_sha",
)

REQUIRED_TOP_LEVEL_FIELDS = (
    "canonical_passes",
    "canonical_phases",
    "configurations",
    "global_deviations",
    "methods",
)

REQUIRED_HARNESS_DEVIATION_FIELDS = ("class", "description", "reversible", "affects")

REQUIRED_GLOBAL_DEVIATION_FIELDS = ("class", "description", "affects", "reversible", "status")

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def default_manifest_path():
    """The manifest sits next to the submodule directories, in the superproject."""
    return Path(__file__).resolve().parents[2] / "methods" / "methods.yaml"


def repo_root(start=None):
    """Superproject working tree containing `start` (default: this file)."""
    start = Path(start) if start else Path(__file__).resolve().parent
    return Path(
        run_git(["rev-parse", "--show-toplevel"], cwd=start, check=True).strip()
    )


def load(path):
    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)


def run_git(args, cwd, check=False):
    """Run git in `cwd`; return stdout, or "" when git fails and check is False."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if check:
            raise RuntimeError(
                f"git {' '.join(args)} in {cwd} failed: {proc.stderr.strip()}"
            )
        return ""
    return proc.stdout


def is_initialised_submodule(fork_dir):
    """A submodule is initialised once its .git file/dir exists."""
    return os.path.exists(os.path.join(fork_dir, ".git"))


def head_sha(fork_dir):
    return run_git(["rev-parse", "HEAD"], cwd=fork_dir).strip()


def tag_sha(fork_dir, tag):
    """Peel `tag` to a commit; "" when the tag does not exist."""
    return run_git(["rev-parse", f"{tag}^{{commit}}"], cwd=fork_dir).strip()


def gitlink_sha(root, path):
    """The SHA the superproject records for the submodule at `path`."""
    out = run_git(["ls-tree", "HEAD", path], cwd=root).strip()
    parts = out.split()
    if len(parts) >= 3 and parts[1] == "commit":
        return parts[2]
    return ""


def commits_since(fork_dir, base):
    """Commits in `base..HEAD`, oldest first, as dicts.

    Returns None when the range cannot be resolved (missing tag or base).
    """
    sep = "\x1e"
    out = run_git(
        ["log", "--reverse", f"--format=%H%x1f%s%x1f%B{sep}", f"{base}..HEAD"],
        cwd=fork_dir,
    )
    if out == "":
        # Empty output is ambiguous: no commits, or a failed range. Resolve the
        # base explicitly to tell the two apart.
        if not run_git(["rev-parse", "--verify", f"{base}^{{commit}}"], cwd=fork_dir).strip():
            return None
        return []
    commits = []
    for record in out.split(sep):
        record = record.strip("\n")
        if not record:
            continue
        sha, subject, body = record.split("\x1f", 2)
        commits.append({"sha": sha, "subject": subject, "message": body})
    return commits


def parse_trailers(message):
    """Extract the deviation trailers from a commit message.

    Trailers are `Key: value` lines with optional indented continuation lines.
    They are not required to be the last block of the message -- the forks put
    `Co-Authored-By:` after them -- so the whole message is scanned and only the
    keys the policy defines are collected.
    """
    keys = ("Deviation", "Affects", "Reversible", "Rationale", "Upstream-ref")
    trailers = {}
    current = None
    for line in message.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z-]*):\s?(.*)$", line)
        if match and match.group(1) in keys:
            current = match.group(1)
            trailers[current] = match.group(2).strip()
        elif current and line.startswith((" ", "\t")) and line.strip():
            trailers[current] = (trailers[current] + " " + line.strip()).strip()
        elif not line.strip():
            current = None
        else:
            current = None
    return trailers
