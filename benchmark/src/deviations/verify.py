"""`deviations verify` -- the gate in front of every benchmark run.

Every check prints exactly one line. The exit status is 0 only when every line
says OK, because CLAUDE.md forbids quoting a measurement taken while `verify`
was failing.
"""

import re
import subprocess
from pathlib import Path

from . import manifest as mf


class Report:
    """Collects pass/fail lines and prints them as they are produced."""

    def __init__(self):
        self.failures = 0
        self.checks = 0

    def check(self, scope, name, ok, reason="", note=""):
        """`reason` explains a failure; `note` annotates a pass."""
        self.checks += 1
        if ok:
            print(f"OK    {scope:<20} {name}" + (f"  ({note})" if note else ""))
        else:
            self.failures += 1
            print(f"FAIL  {scope:<20} {name}: {reason}")
        return ok


def verify_manifest_shape(data, report):
    """Top-level structure of methods.yaml, independent of any fork."""
    for field in mf.REQUIRED_TOP_LEVEL_FIELDS:
        report.check(
            "<manifest>",
            f"top-level field `{field}`",
            field in data,
            "" if field in data else "missing",
        )

    for entry in data.get("global_deviations", []) or []:
        label = str(entry.get("description", ""))[:40].replace("\n", " ")
        missing = [f for f in mf.REQUIRED_GLOBAL_DEVIATION_FIELDS if f not in entry]
        report.check(
            "<manifest>",
            f"global deviation `{label}...`",
            not missing,
            f"missing {', '.join(missing)}",
        )
        cls = entry.get("class")
        report.check(
            "<manifest>",
            f"global deviation class `{cls}`",
            cls in mf.DEVIATION_CLASSES,
            f"`{cls}` is not one of {', '.join(mf.DEVIATION_CLASSES)}",
        )

    canonical = set(data.get("canonical_passes", {}) or {})
    return canonical


def verify_fields(method, report):
    name = method.get("name", "<unnamed>")
    missing = [f for f in mf.REQUIRED_METHOD_FIELDS if f not in method]
    report.check(name, "required fields", not missing, f"missing {', '.join(missing)}")

    provenance = method.get("provenance")
    report.check(
        name,
        "provenance",
        provenance in mf.PROVENANCES,
        f"`{provenance}` is not one of {', '.join(mf.PROVENANCES)}",
    )

    base = str(method.get("upstream_base", ""))
    report.check(
        name,
        "upstream_base is a full SHA",
        bool(mf.SHA_RE.match(base)),
        f"`{base}` is not a 40-character SHA",
    )

    for entry in method.get("harness_deviations", []) or []:
        label = str(entry.get("description", ""))[:32].replace("\n", " ")
        missing = [f for f in mf.REQUIRED_HARNESS_DEVIATION_FIELDS if f not in entry]
        report.check(
            name,
            f"harness deviation `{label}...`",
            not missing,
            f"missing {', '.join(missing)}",
        )
        cls = entry.get("class")
        report.check(
            name,
            f"harness deviation class `{cls}`",
            cls in mf.DEVIATION_CLASSES,
            f"`{cls}` is not a deviation class",
        )


def verify_passes(method, canonical, report):
    name = method.get("name", "<unnamed>")
    passes = method.get("passes") or {}
    unknown = sorted({v for v in passes.values() if v not in canonical})
    report.check(
        name,
        "pass mapping",
        not unknown,
        f"maps to non-canonical pass name(s) {', '.join(unknown)}",
    )


def verify_files(method, root, report):
    """The paths methods.yaml promises must exist in the checked-out fork."""
    name = method.get("name", "<unnamed>")
    fork = root / method["path"]

    subdir = method.get("source_subdir") or "."
    cmakelists = fork / subdir / "CMakeLists.txt"
    report.check(
        name,
        "source_subdir/CMakeLists.txt",
        cmakelists.is_file(),
        f"{cmakelists} does not exist",
    )

    converter = method.get("converter")
    if converter is None:
        initializer = method.get("initializer")
        report.check(
            name,
            "converter",
            initializer is not None,
            "converter is null but no `initializer` is declared",
        )
    else:
        path = fork / converter
        report.check(name, "shipped converter", path.is_file(), f"{path} does not exist")


def verify_git(method, root, report):
    """Fork point, HEAD, and the superproject's record of it."""
    name = method.get("name", "<unnamed>")
    fork = root / method["path"]

    initialised = mf.is_initialised_submodule(fork)
    report.check(
        name, "submodule initialised", initialised, f"{fork} has no .git entry"
    )
    if not initialised:
        return None

    base = method["upstream_base"]
    tag = mf.tag_sha(fork, "upstream-base")
    report.check(
        name,
        "upstream-base tag",
        tag == base,
        f"tag is `{tag or 'absent'}`, manifest says `{base}`",
    )

    head = mf.head_sha(fork)
    fork_sha = method.get("fork_sha")
    if fork_sha is None:
        report.check(name, "fork_sha", True, note=f"null, pinned to HEAD {head[:12]}")
    else:
        report.check(
            name,
            "fork_sha",
            fork_sha == head,
            f"manifest says `{fork_sha}`, HEAD is `{head}`",
        )

    recorded = mf.gitlink_sha(root, method["path"])
    report.check(
        name,
        "superproject gitlink",
        recorded == head,
        f"superproject records `{recorded or 'nothing'}`, fork HEAD is `{head}`",
    )

    return fork


def verify_commits(method, fork, report):
    """Every commit in upstream-base..HEAD carries well-formed trailers."""
    name = method.get("name", "<unnamed>")
    excluded = method.get("status") == "excluded"

    commits = mf.commits_since(fork, "upstream-base")
    if commits is None:
        report.check(
            name,
            "deviation range upstream-base..HEAD",
            False,
            "cannot resolve `upstream-base`; tag the fork point first",
        )
        return

    if excluded:
        report.check(
            name,
            "excluded fork is pristine",
            not commits,
            f"{len(commits)} commit(s) after upstream-base; "
            f"`status: excluded` requires an untouched fork",
        )
        return

    report.check(
        name,
        "deviation range upstream-base..HEAD",
        True,
        note=f"{len(commits)} commit(s)",
    )

    provenance = method.get("provenance")
    foreign = provenance in ("published-reference", "third-party-optimization")

    for commit in commits:
        short = commit["sha"][:12]
        scope = f"{name}@{short}"
        trailers = mf.parse_trailers(commit["message"])

        missing = [
            key
            for key in ("Deviation", "Affects", "Reversible", "Rationale", "Upstream-ref")
            if not trailers.get(key)
        ]
        if not report.check(
            scope,
            f"trailers ({commit['subject'][:44]})",
            not missing,
            f"missing or empty {', '.join(missing)}",
        ):
            continue

        cls = trailers["Deviation"]
        report.check(
            scope,
            "Deviation class",
            cls in mf.DEVIATION_CLASSES,
            f"`{cls}` is not one of {', '.join(mf.DEVIATION_CLASSES)}",
        )

        if foreign:
            report.check(
                scope,
                "class allowed for this provenance",
                cls in mf.CLASSES_ALLOWED_IN_FOREIGN_FORK,
                f"`{cls}` may not appear in a `{provenance}` fork",
            )

        affects = [a.strip() for a in trailers["Affects"].split(",") if a.strip()]
        bad = [a for a in affects if a not in mf.AFFECTS_VALUES]
        report.check(
            scope,
            "Affects values",
            not bad and not ("none" in affects and len(affects) > 1),
            f"`{trailers['Affects']}` -- allowed: {', '.join(mf.AFFECTS_VALUES)}; "
            f"`none` must stand alone",
        )

        if cls in ("normalization", "optimization"):
            reversible = trailers["Reversible"]
            report.check(
                scope,
                "Reversible declares a flag",
                reversible.startswith("flag"),
                f"`Deviation: {cls}` requires `Reversible: flag ...`, got `{reversible}`",
            )


def elf_architectures(binary):
    """sm_XX values embedded in a CUDA binary, via cuobjdump --list-elf."""
    proc = subprocess.run(
        ["cuobjdump", "--list-elf", str(binary)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "cuobjdump failed")
    return sorted(set(re.findall(r"sm_(\d+)", proc.stdout)))


def resolve_binary(method, binaries_dir):
    """The built executable, under either of the two layouts the image uses."""
    fork_name = Path(method["path"]).name
    for candidate in (
        binaries_dir / fork_name / method["executable"],
        binaries_dir / method["executable"],
    ):
        if candidate.is_file():
            return candidate
    return None


def verify_arch(method, binaries_dir, arch, report):
    name = method.get("name", "<unnamed>")
    binary = resolve_binary(method, binaries_dir)
    if binary is None:
        report.check(
            name,
            "built binary",
            False,
            f"`{method['executable']}` not found under {binaries_dir}",
        )
        return
    try:
        architectures = elf_architectures(binary)
    except (RuntimeError, FileNotFoundError) as exc:
        report.check(name, "cuobjdump --list-elf", False, str(exc))
        return
    report.check(
        name,
        f"CUDA architecture sm_{arch}",
        architectures == [str(arch)],
        f"{binary} carries {', '.join('sm_' + a for a in architectures) or 'no SASS'}",
    )


def verify(manifest_path, arch=None, binaries=None):
    """Run every check. Returns the process exit code."""
    path = Path(manifest_path)
    report = Report()

    try:
        data = mf.load(path)
    except Exception as exc:
        report.check("<manifest>", "parses", False, f"{path}: {exc}")
        print(f"\n1 check, 1 failure -- verify FAILED")
        return 1
    report.check("<manifest>", "parses", True, note=str(path))

    canonical = verify_manifest_shape(data, report)
    root = mf.repo_root(path.parent)

    for method in data.get("methods", []) or []:
        verify_fields(method, report)
        verify_passes(method, canonical, report)
        verify_files(method, root, report)
        fork = verify_git(method, root, report)
        if fork is not None:
            verify_commits(method, fork, report)
        # An excluded method is not built, so there is no binary to check.
        if arch is not None and binaries is not None and method.get("status") != "excluded":
            verify_arch(method, Path(binaries), arch, report)

    print()
    if report.failures:
        print(f"{report.checks} checks, {report.failures} failures -- verify FAILED")
        return 1
    print(f"{report.checks} checks, 0 failures -- verify PASSED")
    return 0


def list_deviations(manifest_path):
    """Print the deviation table read from the forks' git logs."""
    path = Path(manifest_path)
    data = mf.load(path)
    root = mf.repo_root(path.parent)

    header = f"{'METHOD':<20} {'SHA':<12} {'CLASS':<16} {'AFFECTS':<22} SUBJECT"
    print(header)
    print("-" * len(header))

    for method in data.get("methods", []) or []:
        name = method["name"]
        fork = root / method["path"]
        if not mf.is_initialised_submodule(fork):
            print(f"{name:<20} {'-':<12} {'-':<16} {'-':<22} (submodule not initialised)")
            continue
        commits = mf.commits_since(fork, "upstream-base")
        if commits is None:
            print(f"{name:<20} {'-':<12} {'-':<16} {'-':<22} (no upstream-base tag)")
            continue
        if not commits:
            print(f"{name:<20} {'-':<12} {'pristine':<16} {'none':<22} (no commits after upstream-base)")
            continue
        for commit in commits:
            trailers = mf.parse_trailers(commit["message"])
            print(
                f"{name:<20} {commit['sha'][:12]:<12} "
                f"{trailers.get('Deviation', '(none)'):<16} "
                f"{trailers.get('Affects', '(none)'):<22} {commit['subject']}"
            )
    return 0
