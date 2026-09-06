"""Paired arms: the same cell measured under two harness settings.

R-TIM-09 (phase timer on/off), R-EXP-11 (debug artefacts off/upstream) and
R-TIM-08 (CUMVS's MVS_BENCH_SYNC off/on) are all the same shape: two campaigns
that differ in one setting, compared per method.

A run key does not name the arm, so the two arms are two campaigns and the
comparison is between campaigns rather than within one. Everything a
comparison assumes -- that the arms ran on one machine, from one image, with
one fork -- is checked here rather than trusted, and reported next to the
numbers.
"""

from . import frames as fr
from . import markdown as md
from . import stats

PAIR_METRICS = [
    ("wall_time_s", "wall time (s)"),
    ("wall_time_with_preprocess_s", "wall + preprocess (s)"),
    ("peak_device_mem_bytes", "peak GPU memory (MiB)"),
    ("peak_host_rss_bytes", "peak host RSS (MiB)"),
    ("io_write_bytes", "I/O written (MiB)"),
    ("intermediates_bytes", "scratch footprint (MiB)"),
    ("energy_j", "GPU energy (J)"),
]


class Arm:
    """One side of a pair: some campaigns, optionally a subset of repeats."""

    def __init__(self, name, campaign_dirs, repeats=None,
                 statuses=fr.DEFAULT_STATUSES, require_clock_hold=False):
        self.name = name
        self.campaign_dirs = campaign_dirs
        self.repeats = None if repeats is None else set(repeats)
        self.statuses = statuses
        self.require_clock_hold = require_clock_hold

    def _in_arm(self, record):
        return self.repeats is None or record.get("repeat") in self.repeats

    @property
    def records(self):
        kept, _ = fr.select(self.campaign_dirs, self.statuses, self.require_clock_hold)
        return [r for r in kept if self._in_arm(r)]

    @property
    def excluded(self):
        _, dropped = fr.select(self.campaign_dirs, self.statuses, self.require_clock_hold)
        keys = {r.get("run_key") for r in fr.records(self.campaign_dirs) if self._in_arm(r)}
        return [(key, why) for key, why in dropped if key in keys]

    def for_method(self, method):
        return [r for r in self.records if r.get("method") == method]


def comparability(arm_a, arm_b):
    """What the pair holds constant, and where it does not.

    Returns (facts, problems). A problem is not fatal -- the numbers are still
    printed -- but it has to appear next to them, because a pair that also
    changed the image or the fork is not measuring the setting it names.
    """
    facts, problems = {}, []
    for field in ("image_digest", "image_manifest_sha256", "fork_sha", "hostname",
                  "evaluator_sha", "scene", "width", "configuration"):
        values_a = {r.get(field) for r in arm_a.records}
        values_b = {r.get(field) for r in arm_b.records}
        facts[field] = sorted(str(v) for v in values_a | values_b)
        if values_a != values_b:
            problems.append(
                f"`{field}` differs between the arms: {arm_a.name}={sorted(map(str, values_a))} "
                f"vs {arm_b.name}={sorted(map(str, values_b))}; this pair does not isolate "
                "the setting it names"
            )
        elif len(values_a | values_b) > 1 and field != "fork_sha":
            problems.append(
                f"`{field}` is not constant within the arms ({facts[field]}), so the "
                "comparison averages over it"
            )
    return facts, problems


def compare(arm_a, arm_b, metrics=PAIR_METRICS, resamples=stats.BOOTSTRAP_RESAMPLES):
    """{method: [row]} with the difference of means (b - a) and its intervals."""
    methods = sorted(
        {r.get("method") for r in arm_a.records} | {r.get("method") for r in arm_b.records}
    )
    out = {}
    for method in methods:
        records_a = arm_a.for_method(method)
        records_b = arm_b.for_method(method)
        rows = []
        columns = list(metrics) + [
            (f"f1@{t}", f"F1 @ {t * 100:g} cm")
            for t in sorted(
                {t for r in records_a + records_b for t in fr.quality_series(r)}
            )
        ]
        for column, label in columns:
            values_a = _values(records_a, column)
            values_b = _values(records_b, column)
            if not any(v is not None for v in values_a + values_b):
                continue
            result = stats.welch(values_a, values_b)
            boot_low, boot_high = stats.bootstrap_difference_ci(
                values_a, values_b, resamples=resamples
            )
            rows.append(
                {
                    "metric": column,
                    "label": label,
                    **result,
                    "boot_low": boot_low,
                    "boot_high": boot_high,
                    "mdd": stats.minimum_detectable_difference(values_a, values_b),
                }
            )
        out[method] = rows
    return out


def _values(records, column):
    if column.startswith("f1@"):
        tolerance = float(column[3:])
        return [fr.quality_series(r).get(tolerance) for r in records]
    return [fr.metric(r, column) for r in records]


def phase_share_shift(arm_a, arm_b, method, phases=None):
    """How each phase's share of the `run` span moves between the arms.

    The measurement R-TIM-08 actually asks for: MVS_BENCH_SYNC does not add
    work, it moves the cost of two unsynchronised launches out of
    `patchmatch.propagate` and into `patchmatch.init`, so the shares move even
    where the total does not.
    """
    shares_a = [fr.phase_shares(r) for r in arm_a.for_method(method)]
    shares_b = [fr.phase_shares(r) for r in arm_b.for_method(method)]
    names = phases or sorted({k for s in shares_a + shares_b for k in s})
    rows = []
    for name in names:
        values_a = [s.get(name) for s in shares_a]
        values_b = [s.get(name) for s in shares_b]
        if not any(v is not None for v in values_a + values_b):
            continue
        summary_a = stats.describe(values_a)
        summary_b = stats.describe(values_b)
        rows.append(
            {
                "phase": name,
                "n_a": summary_a["n"], "n_b": summary_b["n"],
                "share_a": summary_a["mean"], "share_b": summary_b["mean"],
                "sd_a": summary_a["sd"], "sd_b": summary_b["sd"],
                "shift": (
                    None
                    if summary_a["mean"] is None or summary_b["mean"] is None
                    else summary_b["mean"] - summary_a["mean"]
                ),
            }
        )
    return sorted(rows, key=lambda r: -(abs(r["shift"]) if r["shift"] is not None else 0))


def render(arm_a, arm_b, title, note=None, resamples=stats.BOOTSTRAP_RESAMPLES,
           with_shares=False):
    facts, problems = comparability(arm_a, arm_b)
    parts = []
    header = [
        f"Arm A = **{arm_a.name}** ({len(arm_a.records)} runs), "
        f"arm B = **{arm_b.name}** ({len(arm_b.records)} runs). "
        "Differences are B − A; intervals are 95 %."
    ]
    if note:
        header.append(note)
    for arm in (arm_a, arm_b):
        if arm.excluded:
            header.append(
                f"- {arm.name}: "
                + "; ".join(f"`{key}` excluded — {why}" for key, why in arm.excluded)
            )
    for problem in problems:
        header.append(f"- ⚠ {problem}")
    if not problems:
        header.append(
            "- image digest, fork SHA, host, evaluator, scene, width and configuration "
            "are identical across the two arms."
        )
    parts.append("\n\n".join(header))

    for method, rows in compare(arm_a, arm_b, resamples=resamples).items():
        if not rows:
            continue
        body = md.table(
            ["metric", "n A", "n B", "mean A", "mean B", "B − A", "rel.",
             "Welch 95 % CI", "bootstrap 95 % CI", "resolvable ±"],
            [
                [
                    r["label"], str(r["n_a"]), str(r["n_b"]),
                    md.fmt(r["mean_a"]), md.fmt(r["mean_b"]), md.fmt(r["difference"]),
                    md.pct(None if r["relative"] is None else 100 * r["relative"]),
                    f"{md.fmt(r['ci_low'])} – {md.fmt(r['ci_high'])}"
                    if r["ci_low"] is not None else "—",
                    f"{md.fmt(r['boot_low'])} – {md.fmt(r['boot_high'])}"
                    if r["boot_low"] is not None else "—",
                    md.fmt(r["mdd"]),
                ]
                for r in rows
            ],
            align="lrrrrrrllr",
        )
        parts.append(md.section(f"{title} — {method}", body))
        if with_shares:
            shares = phase_share_shift(arm_a, arm_b, method)
            if shares:
                share_body = md.table(
                    ["phase", "share A", "share B", "shift"],
                    [
                        [
                            f"`{r['phase']}`",
                            md.pct(None if r["share_a"] is None else 100 * r["share_a"]),
                            md.pct(None if r["share_b"] is None else 100 * r["share_b"]),
                            md.pct(None if r["shift"] is None else 100 * r["shift"]),
                        ]
                        for r in shares[:20]
                    ],
                    align="lrrr",
                )
                parts.append(
                    md.section(
                        f"{title} — {method}: phase shares of the `run` span",
                        "A span's total includes its nested spans, so these shares "
                        "overlap and do not sum to 100 %. Only the shift is read.\n\n"
                        + share_body,
                        level=4,
                    )
                )
    return "\n".join(parts)
