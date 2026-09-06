"""Analysis of the result store: variance, repeat-count sizing, paired arms.

R-STA-02 (every aggregate carries its repeat count and a dispersion), R-STA-03
(the repeat count comes from the variance pilot), R-TIM-08, R-TIM-09,
R-EXP-11, R-REP-01.

Nothing here writes to the store. Every table is computed from `run.json`
records read through DuckDB, so no number in a note or in the manuscript is
transcribed by hand.
"""

from . import markdown, pairs, stats, variance  # noqa: F401
