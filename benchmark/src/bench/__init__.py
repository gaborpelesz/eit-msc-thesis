"""`bench` -- the benchmarking harness specified by benchmarking/SPEC.md.

Experiment YAML -> run list -> one container per run -> host-side sampler ->
a directory-per-run result store queried with DuckDB. No database.

Nothing under a campaign directory is ever deleted or overwritten by this
package: an aborted or superseded run directory is renamed aside, never
removed.
"""

SCHEMA_VERSION = 1
