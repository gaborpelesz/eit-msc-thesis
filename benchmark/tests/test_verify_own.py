"""`provenance: own` in `verify_git` -- which SHA the method is pinned to.

An own method has no upstream and no `upstream-base` tag, so the fork-point
machinery does not apply to it. What does apply is the pin: a run record quotes
one SHA for the code that ran, and the verifier has to be the thing that says
that SHA is real. Which SHA it is depends on where the method lives.
"""

import subprocess

from deviations import verify as vf


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(cwd),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


def _head(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True
    ).stdout.strip()


def _repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    (path / "f").write_text("1\n")
    _git(path, "add", "f")
    _git(path, "commit", "-qm", "one")
    return path


def _superproject_with_own_submodule(tmp_path):
    """A gitlink written by hand: `submodule add` needs a transport, and the
    only thing under test is the 160000 entry it would have produced."""
    root = _repo(tmp_path / "root")
    sub = _repo(root / "methods" / "m")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{_head(sub)},methods/m")
    _git(root, "commit", "-qm", "pin")
    return root, sub


ENTRY = {
    "name": "OWN",
    "path": "methods/m",
    "provenance": "own",
    "upstream": None,
    "upstream_base": None,
    "fork_sha": None,
}


def test_an_own_submodule_is_pinned_by_the_gitlink(tmp_path):
    root, sub = _superproject_with_own_submodule(tmp_path)
    report = vf.Report()
    assert vf.verify_git(ENTRY, root, report) is None
    assert report.failures == 0
    assert _head(sub) != _head(root)


def test_a_moved_own_submodule_fails_the_gitlink_check(tmp_path):
    root, sub = _superproject_with_own_submodule(tmp_path)
    (sub / "f").write_text("2\n")
    _git(sub, "commit", "-qam", "two")
    report = vf.Report()
    vf.verify_git(ENTRY, root, report)
    assert report.failures == 1


def test_an_own_method_in_the_repository_is_pinned_by_the_superproject(tmp_path):
    root = _repo(tmp_path / "root")
    (root / "methods" / "m").mkdir(parents=True)
    report = vf.Report()
    assert vf.verify_git(ENTRY, root, report) is None
    assert report.failures == 0
