# `mvebench` — self-verifying performance harness for `ETH3DMultiViewEvaluation`

Measures the ETH3D multi-view evaluation binary in isolation, proves that a
variant's output is byte-identical to the reference, and refuses to report a
timing it does not trust.

```
bin/mvebench.py           the harness (golden / gate / bench)
bin/gen_dataset.py        synthetic ETH3D-shaped dataset generator
bin/build_and_gate.sh     identical build + correctness gate for every variant
bin/edge_gate.py          differential gate over the adversarial datasets
bin/gen_edge_datasets.py  10 small adversarial datasets (degenerate / boundary)
bin/gen_edge2_datasets.py 5 large adversarial geometries (70k-120k points)
data/<preset>/            generated datasets (+ manifest.json with SHA-256 of every file)
golden/<preset>.json      reference fingerprints produced from bench/baseline
results/                  JSON reports
```

## Why a bespoke harness

The program is *itself* parallel (OpenMP in the completeness phase) and is
normally invoked from a pipeline that is *also* parallel. Timing it naively
gives numbers that measure the pipeline, not the code. Three failure modes had
to be designed out, not documented away:

| failure | guard |
|---|---|
| an outer pipeline's `OMP_NUM_THREADS=1` (or a pipeline-level lock) silently serializes the run | every run's child `cpu/wall` is measured and asserted; `threads=N` with `cpu/wall ≈ 1` is a **hard error**, not a datum |
| an outer pipeline leaks its own parallelism in and oversubscribes the machine | `cpu/wall > threads + 0.5` is a hard error |
| the thread pool is fully busy but every thread is spinning on the same lock, so `cpu/wall` looks healthy while the phase gets *slower* | per-phase **measured scaling** across thread counts; anything `< 1.0` is reported as `SELF-SERIALIZING` |

Plus: exactly one measured process at a time, an exclusive lock file so two
harness invocations cannot overlap, a refusal to start on battery or on a busy
machine, and variants interleaved round-robin in seeded-random order so thermal
drift hits every variant equally.

Two deliberate choices worth knowing:

* **`OMP_WAIT_POLICY=PASSIVE`.** With libomp's default spin-waiting, threads
  blocked on the `omp critical` in `ComputeCompleteness` burn CPU doing nothing,
  `cpu/wall` reads ~6 even when the region is fully serialized, and idle workers
  keep spinning *into the next, serial phase* — measured at +35% on the accuracy
  phase. `PASSIVE` removes that confound. Same setting for every variant.
* **`posix_spawn` + `wait4`, not `subprocess` + an RSS poller.** A polling
  sampler would fork hundreds of `ps` children per run whose CPU lands in
  `RUSAGE_CHILDREN` and inflates the very `cpu/wall` the guards depend on.
  `wait4` returns the measured process's own rusage and perturbs nothing.
  (Verified: harness reports 425.3 MiB where `/usr/bin/time -l` reports
  445,087,744 B = 424.5 MiB.)

Per-phase timings come from timestamping the program's **own** progress lines on
stdout (`Loading reconstruction:` / `Computing completeness` / `Computing
accuracy` / `Tolerances:`). That works on an unmodified binary, so no variant
carries an instrumentation deviation and every variant is measured identically.

## The byte-exactness gate

`golden` records, and `gate` re-checks:

1. the four stdout result lines (`Tolerances` / `Completenesses` / `Accuracies`
   / `F1-scores`), compared byte for byte;
2. SHA-256 of **12 per-point classification clouds** — 6 tolerances ×
   {accuracy, completeness} — written via `--accuracy_cloud_output_path` /
   `--completeness_cloud_output_path`. These encode the classification of *every
   one of the 2.79M reconstruction points and 3.17M scan points*, so the gate is
   a full semantic comparison, not a spot check on four scalars.

`build_and_gate.sh` runs the gate on `tiny` and `bench` at 1 thread and on
`bench` at 12 threads; a variant that is exact at 1 thread but not at 12 fails.

**The gate is demonstrated to fail, on branch `qa/negative-control`:**

* widening the laser beam model by 2% and nothing else → rejected on all six
  accuracy clouds *and* the `Accuracies`/`F1-scores` lines;
* accumulating the completeness cell average in `float` instead of `double` →
  rejected on the result lines alone (`0.597488` → `0.597487`) while every
  per-point cloud stayed identical.

Those are the two arms of the gate, each observed firing independently.

*Known limit, stated rather than hidden:* the scalar arm compares the program's
printed 6-significant-digit output, because that is what the benchmark pipeline
consumes. A change that alters the cell-average summation below ~1e-6 relative
would pass the scalar arm. The per-point arm has no such tolerance — it is exact.

## The adversarial gates

The byte-exactness gate above measures well-formed data. It says nothing about
what a variant does with an empty cloud, a NaN, or a point sitting exactly on a
tolerance boundary -- and a round-2 variant that replaces the nearest-neighbour
index can diverge on exactly that geometry while staying exact on `bench`.
`edge_gate.py` closes that hole. For each case directory it runs the binary at
1 and 12 threads with that case's own tolerance list and records the exit code,
the four result lines, the first stderr line, and SHA-256 of every cloud the
run emits; the golden is captured from `bench/baseline` and compared field by
field. Only the four result lines are kept from stdout, because the progress
lines carry per-variant absolute paths.

The child environment is scrubbed of every `OMP_*` / `KMP_*` variable the
calling shell might carry and rebuilt explicitly, so the thread count under test
is the thread count that runs.

`gen_edge_datasets.py` -- 10 small cases:

| case | what it probes |
|---|---|
| `empty-recon`, `empty-scan` | zero-point clouds on either side |
| `single-recon-point` | a cloud too small for any partitioning |
| `single-tolerance` | the `tolerances.size() - 1` loop bounds |
| `non-finite` | NaN and +/-Inf coordinates |
| `far-from-origin` | coordinates ~1e5 m out, where voxel cell ids overflow naively |
| `tolerance-boundary` | points placed exactly on a tolerance |
| `coincident` | many points at one location |
| `tolerance-dupes` | unsorted and duplicated tolerance lists |
| `beam-boundary` | points straddling the laser-beam radius at 5 m |

`beam-boundary` was added after the first nine failed to discriminate the
negative control: none of them was sensitive to the beam model, so all nine
passed on a branch known to be wrong. A gate that a known-bad build passes is
not a gate.

`gen_edge2_datasets.py` -- 5 large cases (`shell-80k`, `split-clusters-100k`,
`flat-plane-90k`, `collinear-70k`, `density-extreme-120k`). These exist because
every case above is <= 800 points, which is below the 32768-point threshold at
which the partitioned index path partitions at all: the small gate runs the
unchanged fallback in all ten cases and proves nothing about that code. The
large cases are the degenerate *distributions* -- hollow, clustered, planar,
collinear, 1000:1 density ratio -- that break spatial indices while `bench`
stays green.

Both gates are demonstrated to fail on `qa/negative-control`: 12 mismatches on
the small set, 44 on the large set.

```sh
python3 bin/gen_edge_datasets.py  data/edge
python3 bin/gen_edge2_datasets.py data/edge2
python3 bin/edge_gate.py --binary <baseline-bin> --edge-root data/edge \
        --workdir /tmp/e --write-golden golden/edge.json
python3 bin/edge_gate.py --binary <variant-bin> --edge-root data/edge \
        --workdir /tmp/e --golden golden/edge.json
```

Both generators are seeded (`20260914`, `20260915`), so `data/edge` and
`data/edge2` reproduce bit-for-bit and are not tracked.

## The dataset

`gen_dataset.py` ray-casts a room scene (walls, floor, ceiling, boxes, spheres)
from 3 levelled tripod positions on a regular azimuth × inclination lattice, so
the ground-truth scans are genuine terrestrial-laser-scan geometry: points lying
on rays from the scanner origin, in scanner-local coordinates, with the pose in
the `.mlp`. This matters — `ComputeAccuracy` bins scan points into a 2048×1024
spherical grid and walks an angular neighbourhood of it per reconstruction
point, so only ray-structured data reproduces the real occupancy of that grid
and the real cost of the hot loop. Uniform noise in a box would not.

The reconstruction is then the ground truth mapped to global coordinates with
12% of points dropped, σ = 6 mm Gaussian noise, and 2.5% of points displaced by
σ = 35 cm as gross outliers. The outliers are a deliberate addition to the
brief: without them accuracy is ≈1.0 everywhere, the `inaccurate` branch of
`ClassifyPoint` never executes, and the hot path is unrepresentative.

| preset | scans | GT points | recon points | baseline @ 1 thread |
|---|---|---|---|---|
| `tiny` | 2 | 461k | 405k | 0.99 s |
| `bench` | 3 | 3.17M | 2.79M | 8.79 s |
| `large` | 4 | ~9M | ~8M | (not yet run) |

Everything is seeded; `manifest.json` carries the seed, every parameter and a
SHA-256 per file.

## Measured noise floor (Apple M5 Max, 6P+12E, AC power, idle machine)

Two independent harness invocations, back to back, 11 repeats each after 2
warmups:

| regime | A median | B median | drift | relMAD | p95/median |
|---|---|---|---|---|---|
| 1 thread | 8.786 s | 8.815 s | **+0.34%** | 0.33% / 0.75% | 1.018 / 1.038 |
| 12 threads | 11.448 s | 11.908 s | **+4.02%** | 0.23% / 1.97% | 1.015 / 1.236 |

Per phase at 1 thread, drift between invocations: load +1.80%, completeness
**+0.05%**, accuracy **+0.34%**.

**Consequence: the 1-thread regime is the measurement regime.** Its median is
reproducible to well under 1%, so a >1.05× median speedup with a bootstrap CI
excluding 1.0 is real. The 12-thread regime drifts 4% between invocations and
has a fat tail (one run at 14.7 s against an 11.6 s median), so nothing below
~1.5× should be believed there. `bench` reports a bootstrap 95% CI on the median
speedup ratio and a Mann-Whitney U p-value, and marks `sig` only when the CI
excludes 1.0 *and* p < 0.05.

## Usage

```sh
python3 bin/gen_dataset.py --out data/bench --preset bench
python3 bin/mvebench.py golden --binary <baseline-bin> --dataset data/bench \
        --workdir /tmp/g --out golden/bench.json
bin/build_and_gate.sh h1-nnsearch
python3 bin/mvebench.py bench --dataset data/bench --threads 1,12 \
        --repeats 11 --warmup 2 --baseline baseline \
        --variant baseline=../baseline/build/ETH3DMultiViewEvaluation \
        --variant h1=../h1-nnsearch/build/ETH3DMultiViewEvaluation \
        --json results/campaign.json
```
