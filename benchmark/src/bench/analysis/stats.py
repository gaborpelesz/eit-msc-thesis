"""Dispersion, bootstrap intervals and the repeat-count sizing arithmetic.

Deliberately dependency-free: `statistics` and `math` only. numpy is present
in this environment as a transitive dependency of opencv-python and scipy is
not present at all, so neither may be relied on by code that has to keep
producing the thesis's numbers.
"""

import math
import random
import statistics

# The number of resamples and the seed are recorded next to every interval:
# a percentile bootstrap is a random procedure and an unreproducible interval
# is not evidence.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260906

# Two-sided 95 % Student-t critical values, t(0.975, df). A pilot has three to
# five repeats, where the normal quantile understates the interval by 30-300 %,
# so the table is what the sizing arithmetic uses. Above df=30 the normal
# quantile is within 5 % and is used instead.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
Z975 = 1.959964


def t_critical(df):
    """t(0.975, df); None when df < 1, because there is no interval then."""
    if df is None or df < 1:
        return None
    df = int(math.floor(df))  # a fractional Welch df rounds down: wider, never narrower
    return _T975.get(df, Z975)


def _clean(values):
    return [float(v) for v in values if v is not None and not _isnan(v)]


def _isnan(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def describe(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """n, mean, SD, CV, min-max and a bootstrap 95 % CI of the mean.

    `sd`, `cv` and the interval are None at n < 2: one run of a randomized
    algorithm has no dispersion, and inventing one would be the exact error
    R-STA-01 exists to prevent.
    """
    values = _clean(values)
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": None, "sd": None, "cv": None, "min": None, "max": None,
            "ci_low": None, "ci_high": None, "ci_halfwidth": None,
            "ci_halfwidth_pct": None, "resamples": 0, "seed": seed,
        }
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    low, high = bootstrap_mean_ci(values, resamples=resamples, seed=seed) if n > 1 else (
        None,
        None,
    )
    halfwidth = None if low is None else (high - low) / 2
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cv": None if sd is None or mean == 0 else sd / mean,
        "min": min(values),
        "max": max(values),
        "ci_low": low,
        "ci_high": high,
        "ci_halfwidth": halfwidth,
        "ci_halfwidth_pct": (
            None if halfwidth is None or mean == 0 else 100 * halfwidth / abs(mean)
        ),
        "resamples": resamples if n > 1 else 0,
        "seed": seed,
    }


def bootstrap_mean_ci(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, level=0.95):
    """Percentile bootstrap interval for the mean.

    With n <= 5 the resampling distribution has at most n**n distinct draws and
    the interval cannot reach beyond min..max of the sample; it is reported
    because it makes no normality assumption, next to the t interval, which
    does and can.
    """
    values = _clean(values)
    n = len(values)
    if n < 2:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    tail = (1 - level) / 2
    return (
        means[max(0, int(math.floor(tail * resamples)) - 1)],
        means[min(resamples - 1, int(math.ceil((1 - tail) * resamples)) - 1)],
    )


def t_mean_ci(values, level=0.95):
    """The Student-t interval for the mean; the one the sizing table extends."""
    values = _clean(values)
    n = len(values)
    if n < 2:
        return None, None
    mean = statistics.fmean(values)
    half = t_critical(n - 1) * statistics.stdev(values) / math.sqrt(n)
    return mean - half, mean + half


def projected_halfwidth(mean, sd, n):
    """What a campaign of `n` repeats would get, from a pilot's mean and SD.

    This is the D17 question. Two answers are given because they differ by a
    factor of six at n=1:

    - `t`: the SD is re-estimated from the n runs themselves, so the interval
      uses t(0.975, n-1). Undefined at n=1 -- which is the argument against
      n=1, not a gap in the table.
    - `z`: the pilot's SD is taken as known, so the interval uses 1.96. It is
      the optimistic bound, and it is only as good as the pilot's five runs.
    """
    if mean is None or sd is None or n < 1:
        return {"n": n, "t_halfwidth": None, "t_halfwidth_pct": None,
                "z_halfwidth": None, "z_halfwidth_pct": None}
    z_half = Z975 * sd / math.sqrt(n)
    critical = t_critical(n - 1)
    t_half = None if critical is None else critical * sd / math.sqrt(n)
    scale = abs(mean) or None
    return {
        "n": n,
        "t_halfwidth": t_half,
        "t_halfwidth_pct": None if t_half is None or not scale else 100 * t_half / scale,
        "z_halfwidth": z_half,
        "z_halfwidth_pct": None if not scale else 100 * z_half / scale,
    }


def welch(a, b, level=0.95):
    """Welch's difference of means (b - a) with its interval and t statistic.

    Welch rather than Student: the two arms of a pair are not assumed to have
    equal variance, and with three runs an equal-variance assumption is not
    checkable.
    """
    a, b = _clean(a), _clean(b)
    na, nb = len(a), len(b)
    out = {
        "n_a": na, "n_b": nb, "mean_a": None, "mean_b": None, "difference": None,
        "ci_low": None, "ci_high": None, "t": None, "df": None, "relative": None,
    }
    if na == 0 or nb == 0:
        return out
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    out.update(mean_a=ma, mean_b=mb, difference=mb - ma)
    out["relative"] = None if ma == 0 else (mb - ma) / abs(ma)
    if na < 2 or nb < 2:
        return out
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        out.update(ci_low=mb - ma, ci_high=mb - ma, t=None, df=None)
        return out
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    half = t_critical(df) * se
    out.update(
        ci_low=(mb - ma) - half, ci_high=(mb - ma) + half, t=(mb - ma) / se, df=df
    )
    return out


def bootstrap_difference_ci(a, b, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
                            level=0.95):
    """Percentile bootstrap interval for mean(b) - mean(a), resampled within arm."""
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        sa = sum(a[rng.randrange(len(a))] for _ in a) / len(a)
        sb = sum(b[rng.randrange(len(b))] for _ in b) / len(b)
        diffs.append(sb - sa)
    diffs.sort()
    tail = (1 - level) / 2
    return (
        diffs[max(0, int(math.floor(tail * resamples)) - 1)],
        diffs[min(resamples - 1, int(math.ceil((1 - tail) * resamples)) - 1)],
    )


def minimum_detectable_difference(a, b, level=0.95):
    """The smallest true difference this pair's sample size could resolve.

    Reported next to every pair because "no significant difference" at three
    runs against three is mostly a statement about the sample size. It is the
    half-width of the Welch interval: a true difference smaller than this
    would have produced an interval containing zero whatever the data.
    """
    result = welch(a, b, level=level)
    if result["ci_low"] is None:
        return None
    return (result["ci_high"] - result["ci_low"]) / 2
