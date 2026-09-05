"""Deviation manifest and verifier for the PatchMatch-MVS benchmark.

`benchmark/methods/methods.yaml` is the source of truth; this package reads it
and the forks' git logs. `deviations verify` gates every benchmark run: a
measurement may not be produced from code whose deviations are not recorded.
"""

__all__ = ["manifest", "verify", "cli"]
