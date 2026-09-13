#!/usr/bin/env bash
# Sampling profile of one variant. Usage: profile.sh <worktree-name> [dataset] [threads]
# Uses macOS `sample`, which attaches to the running process and returns a call
# tree with per-symbol sample counts. Output: results/profile_<name>.txt
set -uo pipefail
WT=/Users/gaborpelesz/Desktop/elte-thesis-2026/mve-opt-worktrees
H="$WT/_harness"
NAME="${1:?usage: profile.sh <worktree-name> [dataset] [threads]}"
DS="${2:-bench}"
THREADS="${3:-1}"
BIN="$WT/$NAME/build/ETH3DMultiViewEvaluation"
OUT="$H/results/profile_${NAME}_${DS}_t${THREADS}.txt"
[ -x "$BIN" ] || { echo "no binary: $BIN"; exit 2; }

env -u OMP_NUM_THREADS OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=FALSE \
    OMP_WAIT_POLICY=PASSIVE OMP_MAX_ACTIVE_LEVELS=1 \
    "$BIN" --tolerances 0.01,0.02,0.05,0.1,0.2,0.5 \
      --reconstruction_ply_path "$H/data/$DS/reconstruction.ply" \
      --ground_truth_mlp_path   "$H/data/$DS/scan_alignment.mlp" >/dev/null 2>&1 &
PID=$!
# 1 ms interval for the life of the process; `sample` stops early when it exits.
sample "$PID" 600 1 -file "$OUT" >/dev/null 2>&1
wait "$PID" 2>/dev/null
echo "=== heaviest leaf symbols: $NAME ($DS, ${THREADS}t) ==="
python3 - "$OUT" <<'PYEOF'
import re, sys
txt = open(sys.argv[1], errors="replace").read()
sec = txt.split("Sort by top of stack, same collapsed", 1)
if len(sec) < 2:
    print("no leaf histogram in", sys.argv[1]); raise SystemExit(1)
rows = []
for line in sec[1].splitlines()[1:]:
    m = re.match(r"^\s+(.*?)\s+\(in ([^)]+)\)(?:.*?)\s+(\d+)\s*$", line)
    if not m:
        if line.strip() == "" and rows:
            break
        continue
    sym, img, n = m.group(1).strip(), m.group(2), int(m.group(3))
    rows.append((sym, img, n))
# Idle OpenMP workers parked in the kernel are not work; they would otherwise
# dominate every multi-thread profile and hide the actual hot code.
IDLE = ("__workq_kernreturn", "__psynch_cvwait", "_pthread_cond_wait",
        "start_wqthread", "__semwait_signal", "mach_msg2_trap", "swtch_pri")
work = [(s_, i, n) for s_, i, n in rows if not any(k in s_ for k in IDLE)]
total = sum(n for _, _, n in work) or 1
def short(s_):
    s_ = re.sub(r"\(.*", "", s_)                      # drop argument lists
    s_ = re.sub(r"<[^<>]*(?:<[^<>]*>)?[^<>]*>", "<>", s_)
    return s_[:76]
print(f"  total non-idle samples: {total}")
print(f"{'self%':>7} {'n':>7}  symbol")
acc = 0
for sym, img, n in work[:24]:
    pct = 100.0 * n / total
    acc += pct
    print(f"{pct:6.1f}% {n:7d}  {short(sym):<78} [{img.split('/')[-1]}]")
    if pct < 0.5:
        break
PYEOF
echo
echo "full call tree: $OUT"
