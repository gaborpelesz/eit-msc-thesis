from bench import phases as ph

TRACE = """
PHASE problem_list BEGIN 1000000000
PHASE problem_list END 1500000000
PHASE image_pass BEGIN 2000000000 pass=photometric image=0
PHASE patchmatch BEGIN 2100000000
PHASE patchmatch.propagate BEGIN 2200000000 iter=0
PHASE patchmatch.propagate END 2400000000 iter=0
PHASE patchmatch END 2500000000
PHASE image_pass END 2600000000 pass=photometric image=0
PHASE image_pass BEGIN 3000000000 pass=geom image=1
PHASE image_pass END 3400000000 pass=geom image=1
garbage line that is not a phase
PHASE broken BEGIN not-a-number
PHASE fusion END 4000000000
"""


def test_totals_sum_repeated_spans():
    trace = ph.parse(TRACE)
    totals = {row["name"]: row for row in ph.totals(trace)}
    assert totals["image_pass"]["count"] == 2
    assert totals["image_pass"]["total_s"] == 1.0
    # self time excludes the nested patchmatch span
    assert round(totals["image_pass"]["self_s"], 3) == 0.6
    assert totals["patchmatch"]["total_s"] == 0.4
    assert round(totals["patchmatch"]["self_s"], 3) == 0.2
    assert totals["problem_list"]["total_s"] == 0.5


def test_attributes_and_pass_split():
    rows = {(r["name"], r["pass"]): r for r in ph.totals_by_pass(ph.parse(TRACE))}
    assert rows[("image_pass", "photometric")]["total_s"] == 0.6
    assert rows[("image_pass", "geom")]["total_s"] == 0.4


def test_malformed_lines_are_recorded_not_raised():
    trace = ph.parse(TRACE)
    assert any("timestamp is not an integer" in e for e in trace.errors)
    assert any("END without BEGIN" in e for e in trace.errors)


def test_truncated_trace_reports_unclosed_spans():
    trace = ph.parse(
        "PHASE fusion BEGIN 1000\nPHASE fusion.load BEGIN 1100\nPHASE fusion.load END 1200\n"
    )
    names = {span.name for span in trace.unclosed}
    assert names == {"fusion"}
    totals = {row["name"]: row for row in ph.totals(trace)}
    assert totals["fusion"]["unclosed"] == 1
    assert totals["fusion"]["total_s"] == 0.0
    assert totals["fusion.load"]["count"] == 1


def test_missing_file_is_an_error_not_an_exception(tmp_path):
    trace = ph.parse_file(tmp_path / "absent.txt")
    assert trace.spans == []
    assert trace.errors


def test_format_tree_indents_nested_spans():
    text = ph.format_tree(ph.parse(TRACE))
    lines = [line for line in text.splitlines() if "patchmatch.propagate" in line]
    assert lines and lines[0].startswith("    ")
