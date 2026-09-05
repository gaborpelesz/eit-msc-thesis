# Assessing the State of the Art

## Docker

Build docker image
```bash
docker buildx build -t sota-mvs .
```

Run docker container
```bash
docker run --rm --gpus all -it sota-mvs:latest
```

## Evaluation

```bash
eval-cli --datasets courtyard --methods ACMH --width 1300 2>&1 | tee evaluation.log
```

## Deviations

`methods/methods.yaml` is the source of truth for the harness: per method its
provenance, upstream URL and fork point, the executable and invocation, the
output PLY, the mapping from the method's own pass names to the canonical
vocabulary, its known limitations, and the deviations that live in the harness
rather than in a fork. `global_deviations:` records what applies to every
method (Docker toolchain, CUDA architecture target, shared converter, planned
debug-output flag and phase timer). `DEVIATIONS.md`, `deviations.json` and
`deviations.tex` are generated from it and must never be hand-edited.

`deviations verify` is the gate in front of every benchmark run. **A
measurement may not be produced from code whose deviations are not recorded,
and no number taken while `verify` was failing may be quoted.**

```bash
deviations verify                                  # the whole manifest
deviations verify --manifest path/to/methods.yaml
deviations verify --arch 120 --binaries /sota      # also check the built binaries
deviations list                                    # the deviation table, from the forks' git logs
deviations render                                  # not implemented yet
```

It prints one `OK`/`FAIL` line per check per method and exits non-zero unless
every check passes. What it checks:

- the manifest parses and every required field is present, for the file as a
  whole, for each method, and for each recorded deviation;
- `provenance` is one of `published-reference`, `third-party-optimization`,
  `own`, and every method-native pass name maps to a `canonical_passes` entry;
- `path` is an initialised submodule, `source_subdir/CMakeLists.txt` exists,
  and the shipped converter exists (or, for CUMVS, an `initializer` is
  declared in its place);
- the fork carries a `upstream-base` tag equal to the manifest's
  `upstream_base`; a non-null `fork_sha` equals the fork's HEAD; and the
  superproject's recorded gitlink equals the fork's HEAD;
- every commit in `upstream-base..HEAD` carries `Deviation:`, `Affects:`,
  `Reversible:`, `Rationale:` and `Upstream-ref:` trailers, with `Deviation:`
  one of the seven classes and `Affects:` drawn from
  `none|timing|memory|quality|io`;
- `Deviation: normalization|optimization` declares `Reversible: flag …`;
- a `published-reference` or `third-party-optimization` fork carries no
  `optimization` commit;
- a method with `status: excluded` (DVP-MVS) has zero commits after
  `upstream-base`, i.e. its fork is pristine;
- with `--arch` and `--binaries`, `cuobjdump --list-elf` on each built binary
  reports exactly the configured architecture. Five forks build through
  FindCUDA's `cuda_add_executable`, which ignores `CMAKE_CUDA_ARCHITECTURES`,
  so this check is what keeps the "all methods target the same architecture"
  claim honest.

`--arch` and `--binaries` are given together; `--binaries` is the directory the
image builds into (`/sota`).

## Benchmark harness (`bench`)

`bench` executes a campaign to the SPEC: an experiment YAML expands into a run
list, each run gets one container, a host-side sampler watches it at 100 ms,
and the result is a directory per run under `results/<campaign>/`:

```
results/<campaign>/
  spec.yaml  fingerprint.json  deviations.json
  <run key>/       run.json telemetry.parquet phases.txt phases.harness.txt
                   stdout.log stderr.log convert.*.log eval.*.log
  <run key>.tmp/   in flight
```

A run key is `<method>__<scene>__w<width>__<config>__r<repeat>`. A run is
finished when `run.json` exists; the directory is renamed into place only then.
**Nothing in the store is ever deleted or overwritten**: an interrupted attempt
becomes `<key>.tmp.<timestamp>.aborted` and a re-run moves the old record to
`<key>.<timestamp>.superseded`. Only bare run-key directories are queried.

```bash
bench plan   experiments/pilot-a.yaml [--pilot results/pilot-a]
bench run    experiments/pilot-a.yaml [--only KEY ...] [--dry-run] [--rerun]
bench fingerprint --image sota-deps:latest
bench status results/pilot-a --spec experiments/pilot-a.yaml
bench query  results/pilot-a [--report wall|memory|quality|status] [--sql SQL]
bench phases results/pilot-a/<run key>
```

`bench run` refuses to start when `deviations verify` fails (R-ENV-01), when the
benchmark GPU has unlocked clocks, persistence mode off or a display attached
(R-ENV-02), or when the campaign's `fingerprint.json` was written on a machine
with a different GPU model, driver, CUDA version or image digest (R-ENV-06).
The first SIGINT pauses after the run in flight has been recorded; the second
aborts it.

`--dry-run` prints the exact docker commands and executes nothing.

## Layout

- [ETH3D dataset downloader](./eth3d)
- benchmark harness: `src/bench/`, experiment specs in `experiments/`
- evaluation (v0, superseded by `bench`): `src/eval/eval.py`
- deviation manifest and verifier: `methods/methods.yaml`, `src/deviations/`
- tests: `tests/` (`uv run pytest`; no GPU or docker needed)
