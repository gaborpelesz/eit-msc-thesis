"""Result store -> manuscript tables and figures.

CLAUDE.md, "thesis/ -- the manuscript": every table and figure the manuscript
carries is produced by a checked-in, re-runnable script from the result store;
no number is ever typed by hand. This package is that script.

Input is one or more campaign directories as `bench run` writes them
(`spec.yaml`, `fingerprint.json`, `deviations.json`, one directory per run with
`run.json`, `telemetry.parquet`, `phases.txt`). Output is
`thesis/generated/results-<campaign>-*.tex` plus pgfplots figure sources.

Two rules the generator enforces rather than assumes:

* a method whose provenance the manifest does not state is refused, because a
  result table that cannot say whether a method is the authors' release or a
  third party's optimisation is not publishable (CLAUDE.md, "Method
  provenance");
* failed runs are counted in a status column, never dropped, because a method
  that fell over is a result (CLAUDE.md, "Measurement integrity").
"""

__all__ = ["load", "tables", "figures", "cli"]
