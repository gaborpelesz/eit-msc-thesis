# MSc thesis monorepo (ELTE FI)

*"High-Performance Patch-Match Multi-View Stereo: Separating Algorithmic Cost
from Implementation Overhead"* — due 15 November 2026.

## Layout

- **`benchmark/`** — the benchmarking harness and eleven forked MVS
  implementations under `benchmark/methods/`. The deviation policy below
  governs every change to a method fork or to the measurement pipeline.
- **`thesis/`** — LaTeX manuscript (ELTE FI template).
- **`resources/`** — thesis declaration form, `papers.yaml`, and the private
  papers and obsidian submodules.

Two rules that outrank convenience everywhere in this repo:

1. Method provenance is load-bearing and easy to get wrong. See the provenance
   table below.
2. Numbers in the manuscript must be generated from the results store, never
   transcribed by hand.

## Working with the author

The author decides; Claude drafts, surveys, transcribes and builds. A design
question is resolved only by an explicit answer from the author — "no
objection" is not an answer. Prefer a brief with two to four costed options and
a recommendation over an open question.

Plan, work items and findings live in the private vault (`resources/obsidian/`):
`plan.md`, `TODO.md`, `findings/` (one note per finding, each with a
verification procedure), `benchmarking/` (the harness SPEC and decision log).
Read `plan.md` and `TODO.md` when resuming. Anything observed about a method's
code or a paper that the thesis could cite goes into `findings/` the same day,
with `file:line` and the fork SHA it was read at.

## Submodules

All submodules are declared in the repo-root `.gitmodules`; git does not read a
`.gitmodules` in a subdirectory.

- `benchmark/methods/*` — eleven public method forks under `gaborpelesz/`.
- `resources/papers` — submodule name `papers`, pointing at the **private**
  `eit-msc-thesis-papers`. It holds publisher versions of record (IEEE TPAMI
  camera-ready, IEEE PDFeXpress-certified PDFs) that may not be redistributed.
  **This repo is public; that one must never be made public, and no workflow may
  mirror, attach, or publish its contents.** `git clone --recursive` therefore
  fails on `papers` for anyone without access; the failure is benign.
- `resources/obsidian` — submodule name `obsidian`, pointing at the **private**
  `eit-msc-thesis-obsidian`. It holds unfiltered working notes: reading
  commentary, critiques of published work, open questions. Written for the
  author, not for publication. **Never move note content into this public
  repository**; `--recursive` fails on it too, benignly.

`resources/papers.yaml` maps each PDF filename to its arXiv ID or DOI, so the
bibliography stays reproducible for readers who cannot clone the submodule.

---

# `benchmark/` — benchmarking harness

The thesis claims that much of the runtime/memory growth across the PatchMatch
lineage is inherited implementation debt rather than algorithmic necessity.
Every number this repo produces is evidence for that claim, so **measurement
integrity outranks convenience in every design decision here.**

## State of the harness (2026-09-05)

- `benchmark/src/eval/` is the **v0** harness (SQLite, wall time + F1). It is
  being replaced by `benchmark/src/bench/`, built to the SPEC in the vault
  (`benchmarking/SPEC.md`): experiment YAML → run list → one container per run →
  host-side sampler → a **directory-per-run result store** (`run.json`,
  `telemetry.parquet`, `phases.txt`, logs) queried with DuckDB. No database.
- `benchmark/methods/methods.yaml`, the `upstream-base` tags and the
  `deviations verify` CLI exist (audit of 2026-09-05; `uv run deviations
  verify` must pass before any measurement). `deviations render` is a stub.
- DVP-MVS is excluded from the campaign (its release does not implement the
  paper's prior and the DL preprocessing was never published — D19-b, F-018);
  the fork stays pristine and the verifier enforces zero own commits.
- Phase timer (`bench_timer.h`) and the `--no-debug-output` flag are being
  added to the ten campaign forks; the Dockerfile still builds CUMVS twice and
  targets `sm_75` until the fleet image (`sm_120`) lands.
- What we built and how is journalled in the vault, `methodology/` (one note
  per artefact); the manuscript's methodology chapter is written from it.

## Method provenance

`benchmark/methods/` holds forks (under `gaborpelesz/`) of eleven
implementations. They are not equivalent in status, and the distinction must
never be blurred:

- **`published-reference`** — the authors' own release accompanying a paper
  (ACMH, ACMM, ACMP, ACMMP, APD-MVS, HPM-MVS, HPM-MVS++, MP-MVS, DPE-MVS, DVP-MVS).
- **`third-party-optimization`** — an external, non-authorial attempt at
  optimizing the ACM family. **CUMVS (`cuda-multi-view-stereo`) is this.** It is
  *not* the thesis author's work; it is prior art cited as evidence that
  optimization headroom exists.
- **`own`** — the thesis author's own reimplementation (planned: a fork of
  APD-MVS, optimised in measured steps), once it exists.

Whenever a method is described in prose, in a table, or in the thesis, its
provenance must be stated correctly. Do not attribute CUMVS to the author.

## Deviation policy

Every fork differs from its upstream. Undisclosed differences are a threat to
validity and the first thing a reviewer attacks. The governing rule:

> **A measurement may not be produced from code whose deviations are not
> recorded.** `deviations verify` gates every benchmark run.

### Classes

| Class | Meaning | Perf-neutral? |
|---|---|---|
| `build` | compiler, flags, CUDA arch, dependency versions | **No — assume it affects results** |
| `compat` | make it build/run on the current toolchain | Assess |
| `interface` | CLI/IO plumbing to fit the harness | Should be; verify |
| `instrumentation` | the phase timer (`bench_timer.h`), env-gated, byte-identical when off | Must be measured, not assumed |
| `bugfix` | repairs a real defect (crash, wrong output) | Assess |
| `normalization` | deliberately changes behaviour to make methods comparable | **Yes, by construction** |
| `optimization` | performance changes (only for the `own` method) | **Yes, by construction** |

What a fork may contain after the audit: `build`, `compat`, `bugfix`,
`interface`, the `instrumentation` phase timer, and one `normalization` flag
to disable debug artefact output. Nothing else. Anything that changes what a
published method computes stays out of the forks; parameter normalization is
done at the harness level (shared converter, neighbour count) and recorded in
`methods.yaml`.

### Rules

1. **Normalization and optimization are additive, never destructive.** Add an
   alternative path behind a runtime flag; leave the upstream behaviour reachable
   and default. Then measure both — a normalization delta is a result, not just a
   disclosure.
2. Every commit in a fork carries deviation trailers (below). One deviation per
   commit; never batch unrelated changes.
3. `Deviation: normalization|optimization` **must** declare `Reversible: flag …`.
   The verifier enforces this.
4. Deviations living in the harness rather than in a fork (converter choice,
   padding, neighbour count, parameter overrides) are recorded in
   `benchmark/methods/methods.yaml`, not only in code.
5. Deviations applying to all methods (the Docker toolchain and CUDA target, the
   shared converter used by normalized configurations) are recorded under
   `global_deviations:` in the same manifest.

### Commit trailer format

```
<short summary>

<body>

Deviation: normalization
Affects: quality,timing          # none | timing,memory,quality,io
Reversible: flag --no-debug-output    # or: no
Rationale: <why this was necessary, in full sentences>
Upstream-ref: <file or symbol in the original implementation>
```

### Source of truth

- `benchmark/methods/methods.yaml` — per method: repo, upstream URL,
  `upstream_base` SHA, `provenance`, executable, arg template, converter,
  output PLY path, pass-name mapping to the canonical vocabulary, known
  limitations, harness deviations. Plus `global_deviations:`.
- Each fork tags its fork point: `git tag upstream-base <sha>`.
- `git log upstream-base..HEAD` in a fork is the exhaustive deviation list.

Generated (never hand-edited), at writing time: `DEVIATIONS.md`,
`deviations.json` (embedded in every run record), `deviations.tex` (thesis
appendix table). Until then `verify` is the only tool: tags present, trailers
well-formed, manifest SHAs equal to the submodule SHAs.

## Process — follow this every time

Before editing anything under `benchmark/methods/`:
1. Read the method's `methods.yaml` entry; confirm its provenance and existing
   deviations.
2. Decide the deviation class *before* writing code. If it is `normalization` or
   `optimization`, design it as a flag-selectable alternative from the start.

After editing:
3. Commit in the fork with complete trailers. Update `upstream_base` only if the
   fork was rebased onto new upstream work.
4. Bump the submodule SHA in `methods.yaml` and in the superproject.
5. Run `deviations verify`.
6. Never quote a measurement taken while `verify` was failing.

When reporting results: state provenance, the configuration (`author` or
`norm10`) and any `normalization` in force. If a normalization has a measured
counterpart, report the pair, not just the normalized number.

## Measurement integrity

The normative requirements are in the vault SPEC. The non-negotiables:

- Locked GPU clocks, persistence mode, no display on the benchmark GPU;
  randomized run order across repeats; one campaign is bound to one hardware
  fingerprint (GPU model, driver, CUDA, image digest) and refuses to continue
  on a different one.
- Every run record carries full provenance: hardware, driver, CUDA, image
  digest, all submodule SHAs, `deviations.json`, parameters, width,
  configuration, repeat index.
- Failures (OOM, segfault, timeout, no output) are recorded as results, never
  discarded or retried silently. A re-run moves the old record aside; nothing
  in the result store is ever deleted.
- PatchMatch is randomized: no single-run claim about quality or runtime.
  Repeat count comes from the variance pilot.
- Preprocessing outside the method binary (converter, deep-learning inference)
  is timed as its own phase; runtime is reported both with and without it.

---

# `resources/` — paperwork, papers and notes

- The topic is described in `resources/Thesis declaration form.md`.
- Accompanying papers are in `resources/papers/` (private submodule, see above).
- Working notes are in `resources/obsidian/` (private submodule, see above); the
  Obsidian vault root is that directory, so wiki-links resolve within it. Dated
  session logs live in `resources/obsidian/sessions/`; the plan, TODO list,
  findings collection and benchmarking design sit alongside it at that level.
