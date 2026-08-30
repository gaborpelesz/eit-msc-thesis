# High-Performance Patch-Match Multi-View Stereo

*Separating Algorithmic Cost from Implementation Overhead*

MSc thesis, ELTE Faculty of Informatics.

## Layout

| Path | Contents |
|---|---|
| `benchmark/` | Benchmarking harness for the PatchMatch-MVS family, plus eleven forked method implementations as submodules under `benchmark/methods/`. |
| `thesis/` | LaTeX manuscript (ELTE FI template) and its build. |
| `resources/` | Thesis paperwork, the reference paper collection, and the working-notes vault. |

## Cloning

```sh
git clone --recursive https://github.com/gaborpelesz/eit-msc-thesis
```

**`--recursive` will fail on `resources/papers` and `resources/obsidian` unless
you have access to them.** Both point at private repositories: the paper
collection contains publisher versions of record (IEEE camera-ready PDFs) that
may not be redistributed, and the vault holds personal working notes. The
failure is benign — every other submodule is public and will have been checked
out. To skip them deliberately:

```sh
git clone https://github.com/gaborpelesz/eit-msc-thesis
cd eit-msc-thesis
git submodule update --init --recursive benchmark/methods
```

`resources/papers.yaml` maps every PDF in that collection to its arXiv ID or
DOI, so the sources can be retrieved independently.
