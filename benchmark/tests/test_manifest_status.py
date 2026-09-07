"""`status:` in methods.yaml -- what may and may not be measured.

`active` is the default and the only measurable status. The other two say why a
fork is in the manifest without being measurable, and the verifier demands
different things of them: an `excluded` fork must stay pristine, a `pending`
one is expected to grow its audit commits.
"""

import pytest
from deviations import manifest as mf


def test_absent_status_means_active():
    assert mf.status({}) == "active"
    assert mf.is_measurable({})


@pytest.mark.parametrize("status", ["pending", "excluded"])
def test_only_active_is_measurable(status):
    assert not mf.is_measurable({"status": status})


def test_every_manifest_status_is_in_the_vocabulary(manifest):
    for entry in manifest["methods"]:
        assert mf.status(entry) in mf.STATUSES, entry["name"]


def test_a_method_that_is_not_measurable_says_why(manifest):
    unmeasurable = [e for e in manifest["methods"] if not mf.is_measurable(e)]
    assert unmeasurable, "DVP-MVS is excluded from the campaign (D19-b, F-018)"
    for entry in unmeasurable:
        assert str(entry.get("reason", "")).strip(), entry["name"]


def test_the_excluded_method_is_the_one_the_decision_log_names(manifest):
    excluded = sorted(e["name"] for e in manifest["methods"] if mf.status(e) == "excluded")
    # DVP-MVS: D19-b (2026-09-05). TSAR-MVS: D30 (2026-09-07).
    assert excluded == ["DVP-MVS", "TSAR-MVS"], "a new exclusion needs a decision-log entry"
