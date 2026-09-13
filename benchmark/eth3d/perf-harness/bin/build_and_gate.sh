#!/usr/bin/env bash
# Identical build + correctness gate for every variant. Usage:
#   build_and_gate.sh <worktree-dir-name>        e.g. build_and_gate.sh h1-nnsearch
# Builds with the canonical flags, then runs the byte-exactness gate on both the
# tiny and bench datasets. Exits nonzero if the build fails or either gate fails.
set -uo pipefail
WT=/Users/gaborpelesz/Desktop/elte-thesis-2026/mve-opt-worktrees
H="$WT/_harness"
NAME="${1:?usage: build_and_gate.sh <worktree-dir-name>}"
SRC="$WT/$NAME"
GATE_SCRATCH="/private/tmp/claude-501/-Users-gaborpelesz-Desktop-elte-thesis-2026-eit-msc-thesis-benchmark-eth3d-multi-view-evaluation/d1f4aefe-f043-44d6-b718-a071cefe5b5f/scratchpad/gate/$NAME"

[ -d "$SRC" ] || { echo "no such worktree: $SRC"; exit 2; }

echo "=== configure+build $NAME ==="
# cmake creates $SRC/build itself, but the redirect below opens the log BEFORE
# cmake runs, so a fresh worktree would fail with "No such file or directory".
mkdir -p "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" -G Ninja -Wno-dev \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DOpenMP_ROOT=/opt/homebrew/opt/libomp \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/pcl;/opt/homebrew/opt/eigen;/opt/homebrew/opt/boost;/opt/homebrew/opt/flann" \
  > "$SRC/build/configure.log" 2>&1 || { echo "CONFIGURE FAILED"; tail -30 "$SRC/build/configure.log"; exit 3; }
cmake --build "$SRC/build" 2>&1 | tail -40
BIN="$SRC/build/ETH3DMultiViewEvaluation"
[ -x "$BIN" ] || { echo "BUILD FAILED: no binary"; exit 3; }

RC=0
for DS in tiny bench; do
  echo "=== gate $NAME on $DS ==="
  python3 "$H/bin/mvebench.py" gate --binary "$BIN" --dataset "$H/data/$DS" \
    --workdir "$GATE_SCRATCH/$DS" --golden "$H/golden/$DS.json" --threads 1 \
    --json "$H/results/gate_${NAME}_${DS}.json" || RC=1
done
# The variant must also be deterministic under threads, or its timing at
# threads>1 would not be comparable to its own single-threaded output.
echo "=== gate $NAME on bench @ 12 threads ==="
python3 "$H/bin/mvebench.py" gate --binary "$BIN" --dataset "$H/data/bench" \
  --workdir "$GATE_SCRATCH/bench_t12" --golden "$H/golden/bench.json" --threads 12 \
  --json "$H/results/gate_${NAME}_bench_t12.json" || RC=1
rm -rf "$GATE_SCRATCH"
[ $RC -eq 0 ] && echo "ALL GATES PASS: $NAME" || echo "GATE FAILURE: $NAME"
exit $RC
