# MSc thesis monorepo (ELTE FI)

*"High-Performance Patch-Match Multi-View Stereo: Separating Algorithmic Cost
from Implementation Overhead"*

## Layout

- **`benchmark/`** — the benchmarking harness and eleven forked MVS
  implementations under `benchmark/methods/`. The deviation-documentation policy
  below governs every change to a method fork or to the measurement pipeline.
- **`thesis/`** — LaTeX manuscript (ELTE FI template).
- **`resources/`** — thesis paperwork, plus the papers and obsidian submodules.

Two rules that outrank convenience everywhere in this repo:

1. Method provenance is load-bearing and easy to get wrong: CUMVS
   (`cuda-multi-view-stereo`) is **third-party prior art**, not the author's own
   work. See the provenance table below.
2. Numbers in the manuscript must be generated from the results store, never
   transcribed by hand.

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

Benchmarking harness for the PatchMatch-MVS family.

The thesis claims that much of the runtime/memory growth across the PatchMatch
lineage is inherited implementation debt rather than algorithmic necessity.
Every number this repo produces is evidence for that claim, so **measurement
integrity outranks convenience in every design decision here.**

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
- **`own`** — the thesis author's own reimplementation, once it exists.

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
| `instrumentation` | NVTX ranges, timers, counters | Must be measured, not assumed |
| `bugfix` | repairs a real defect (crash, wrong output) | Assess |
| `normalization` | deliberately changes behaviour to make methods comparable | **Yes, by construction** |
| `optimization` | performance changes (only for the reimplemented method) | **Yes, by construction** |

### Rules

1. **Normalization and optimization are additive, never destructive.** Add an
   alternative path behind a runtime flag; leave the upstream behaviour reachable
   and default. Then measure both — a normalization delta is a result, not just a
   disclosure.
2. Every commit in a fork carries deviation trailers (below). One deviation per
   commit; never batch unrelated changes.
3. `Deviation: normalization|optimization` **must** declare `Reversible: flag …`.
   The verifier enforces this.
4. Deviations living in the harness rather than in a fork (preprocessing choice,
   padding, parameter overrides) are recorded in `benchmark/methods/methods.yaml`,
   not only in code.
5. Deviations applying to all methods (the shared `colmap2mvsnet_acm_perf`
   converter, the Docker toolchain and `sm_75` target) are recorded under
   `global_deviations:` in the same manifest.

### Commit trailer format

```
<short summary>

<body>

Deviation: normalization
Affects: quality,timing          # none | timing,memory,quality,io
Reversible: flag --preprocessing=acm-shared    # or: no
Rationale: <why this was necessary, in full sentences>
Upstream-ref: <file or symbol in the original implementation>
```

### Source of truth

- `benchmark/methods/methods.yaml` — per method: repo, upstream URL,
  `upstream_base` SHA, `provenance`, executable, arg template, preprocessing
  mode, output PLY path, known limitations, harness deviations. Plus
  `global_deviations:`.
- Each fork tags its fork point: `git tag upstream-base <sha>`.
- `git log upstream-base..HEAD` in a fork is the exhaustive deviation list.

Generated (never hand-edited): `DEVIATIONS.md`, `deviations.json`,
`deviations.tex` (thesis appendix table). `deviations.json` is embedded in every
run record so each figure traces to an exact patch set.

## Process — follow this every time

Before editing anything under `benchmark/methods/`:
1. Read the method's `methods.yaml` entry; confirm its provenance and existing deviations.
2. Decide the deviation class *before* writing code. If it is `normalization` or
   `optimization`, design it as a flag-selectable alternative from the start.

After editing:
3. Commit in the fork with complete trailers. Update `upstream_base` only if the
   fork was rebased onto new upstream work.
4. Bump the submodule SHA in `methods.yaml` and in the superproject.
5. Run `deviations verify`, then `deviations render`.
6. Never quote a measurement taken while `verify` was failing.

When reporting results: state provenance and any `normalization` in force. If a
normalization has a measured counterpart, report the pair, not just the
normalized number.

## Measurement integrity

- Benchmarks require locked GPU clocks, persistence mode, no display on the
  benchmark GPU, and randomized run order across repeats.
- Every run record carries full provenance: hardware, driver, CUDA, image digest,
  all submodule SHAs, `deviations.json`, parameters, resolution, repeat index.
- Failures (OOM, segfault, timeout) are recorded as results, not discarded.
- PatchMatch is randomized: no single-run claim about quality or runtime.

---

# `resources/` — paperwork, papers and notes

- The topic is described in `resources/Thesis declaration form.md`.
- Accompanying papers are in `resources/papers/` (private submodule, see above).
- Working notes are in `resources/obsidian/` (private submodule, see above); the
  Obsidian vault root is that directory, so wiki-links resolve within it. Dated
  session logs live in `resources/obsidian/sessions/`, and further note
  categories belong alongside it at that level.
