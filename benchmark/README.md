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

## Layout

- [ETH3D dataset downloader](./eth3d)
- evaluation: `src/eval/eval.py`
- deviation manifest and verifier: `methods/methods.yaml`, `src/deviations/`
