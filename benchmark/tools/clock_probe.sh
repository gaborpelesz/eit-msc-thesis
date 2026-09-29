#!/usr/bin/env bash
# D25 -- what graphics clock can this card hold under sustained MVS load?
#
# A consumer card locked near its boost ceiling will not hold that clock once a
# real method saturates it: it hits the power cap and drops, `clock_hold` falls
# below the D24.2 threshold of 0.95, and every run of the campaign is flagged.
# The lock therefore has to be a measured number, not a guess.
#
# Run a real method container by hand (the commands `bench run --dry-run` prints
# for one cheap run key), and run this alongside it. It samples the benchmark
# GPU at 1 Hz, drops the first SKIP seconds so the ramp does not enter the
# statistics, and prints the distribution of the rest. Lock at or a little below
# p05:
#
#     sudo nvidia-smi -i 0 -lgc <MHz>,<MHz>
#
# Usage:  tools/clock_probe.sh [seconds]        (GPU=0 SKIP=60 OUT=<csv>)
set -euo pipefail

gpu=${GPU:-0}
seconds=${1:-600}
skip=${SKIP:-60}
out=${OUT:-/data/bench/diag/clock-probe-$(date +%Y%m%dT%H%M%S).csv}
mkdir -p "$(dirname "$out")"

echo "sampling GPU $gpu for ${seconds}s at 1 Hz -> $out (first ${skip}s dropped)" >&2
echo "timestamp,clocks_sm_mhz,power_w,temperature_c,utilization_pct" > "$out"
timeout "$seconds" nvidia-smi -i "$gpu" \
  --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 1000 >> "$out" || true

python3 - "$out" "$skip" <<'PY'
import statistics, sys

path, skip = sys.argv[1], int(sys.argv[2])
rows = []
with open(path) as handle:
    next(handle, None)
    for line in handle:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append((int(parts[1]), float(parts[2]), int(parts[3]), int(parts[4])))
        except ValueError:
            continue

window = rows[skip:]
if len(window) < 30:
    sys.exit(f"only {len(window)} samples after dropping the first {skip}; "
             "run the probe for longer, or lower SKIP")

clocks = sorted(r[0] for r in window)
busy = [r for r in window if r[3] >= 50]


def pct(values, q):
    return values[min(len(values) - 1, int(q * len(values)))]


print(f"samples            {len(window)} ({len(busy)} at >=50% utilization)")
print(f"clock  min/p05/med {clocks[0]} / {pct(clocks, 0.05)} / {statistics.median(clocks)} MHz")
print(f"clock  p95/max     {pct(clocks, 0.95)} / {clocks[-1]} MHz")
print(f"power  med/max     {statistics.median(r[1] for r in window):.0f} / "
      f"{max(r[1] for r in window):.0f} W")
print(f"temp   med/max     {statistics.median(r[2] for r in window)} / "
      f"{max(r[2] for r in window)} C")
print()
print(f"suggested lock:    {int(pct(clocks, 0.05) // 15 * 15)} MHz "
      "(p05 rounded down to a supported step)")
if len(busy) < 0.8 * len(window):
    print("WARNING: the GPU was idle for much of the window; the load was not "
          "running throughout and these numbers are not a sustained-load "
          "measurement.")
PY
