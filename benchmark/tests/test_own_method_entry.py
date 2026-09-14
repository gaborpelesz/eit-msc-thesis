"""The `own` method's manifest entry -- the facts a reader must not have to
infer.

Provenance is load-bearing here in both directions: this method is the author's
and CUMVS is not, and neither may be described as the other. The rest of this
file pins the two ways its entry differs in kind from every other one.
"""

from bench import runner as rn
from bench import spec as sp
from deviations import manifest as mf

NAME = "quorum-mvs"


def _entry(manifest):
    entry = sp.method_entry(manifest, NAME)
    assert entry is not None, f"{NAME} is not in methods.yaml"
    return entry


def test_it_is_the_only_own_method_and_declares_no_upstream(manifest):
    own = [e for e in manifest["methods"] if mf.is_own(e)]
    assert [e["name"] for e in own] == [NAME]
    entry = _entry(manifest)
    assert entry["upstream"] is None and entry["upstream_base"] is None


def test_cumvs_is_not_attributed_to_the_author(manifest):
    assert sp.method_entry(manifest, "CUMVS")["provenance"] == "third-party-optimization"


def test_it_ships_no_converter_and_says_so(manifest):
    entry = _entry(manifest)
    assert entry["converter"] is None
    assert mf.converter_source(entry) == "shared"
    # The asymmetry is what has to be disclosed, not the null field.
    assert any(
        "converter" in str(d["description"]).lower() for d in entry["harness_deviations"]
    )


def test_author_preprocessing_is_the_shared_converter(write_spec, spec_dict, manifest):
    spec_dict.update(methods=[NAME], configurations=["author"])
    spec = sp.load(write_spec(spec_dict), manifest)
    run = sp.expand(spec)[0]
    argv = rn.preprocess_argv(spec, _entry(manifest), run)
    assert argv[0].endswith("colmap2mvsnet_acm_perf")


def test_the_invocation_carries_a_per_repeat_seed_and_no_debug_flag(
    write_spec, spec_dict, manifest
):
    spec_dict.update(methods=[NAME], repeats=3)
    spec = sp.load(write_spec(spec_dict), manifest)
    entry = _entry(manifest)
    assert rn.DEBUG_OUTPUT_FLAG not in entry["invocation"], "its parser rejects unknown flags"
    seeds = set()
    for run in sp.expand(spec):
        argv = rn.resolve_invocation(spec, entry, run)
        seeds.add(argv[argv.index("--seed") + 1])
        assert argv[-1].endswith(f"/{entry['output_ply']}")
    assert seeds == {"1", "2", "3"}


def test_the_view_budget_is_the_twenty_every_number_was_measured_at(manifest):
    invocation = _entry(manifest)["invocation"]
    assert invocation[invocation.index("--num-source-views") + 1] == "20"
