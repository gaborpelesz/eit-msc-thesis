#!/usr/bin/env python3
"""Second edge set: LARGE adversarial geometries.

The first edge set is all <= 800 points, which is below the thresholds at which
several round-2 variants switch on their new code path at all (the sub-tree
index only partitions above 32768 points, the voxel table only estimates above
65536). Those cases therefore proved nothing about the paths that actually run
in a campaign. These are sized above every such threshold and shaped to be the
worst case for spatial partitioning and for uniform-grid bucketing.
"""
import numpy as np, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_edge_datasets import write_ply_xyz, write_mlp, case, I4, T1  # noqa: E402

def main():
    root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260915)

    # 1. Hollow spherical shell, 80k points. Every axis-aligned partition of a
    #    shell has a large empty interior, so a box-min-distance bound prunes
    #    badly and the sub-tree index degrades toward searching every partition.
    n = 80000
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    shell = (v * 4.0).astype(np.float32)
    case(root, "shell-80k", [(shell, I4)],
         (shell + rng.normal(0, 0.004, shell.shape)).astype(np.float32))

    # 2. Two dense clusters 500 m apart, 100k points. The bounding box is huge
    #    while the occupied volume is tiny: a dense uniform grid sized from the
    #    bounding box would allocate enormously, and a partition scheme sees a
    #    pathological split.
    a = rng.normal(0, 0.3, size=(50000, 3))
    b = rng.normal(0, 0.3, size=(50000, 3)) + np.array([500.0, 0.0, 0.0])
    two = np.vstack([a, b]).astype(np.float32)
    case(root, "split-clusters-100k", [(two, I4)],
         (two + rng.normal(0, 0.004, two.shape)).astype(np.float32))

    # 3. A perfectly flat plane, 90k points, degenerate in z. Any median split on
    #    z is meaningless and a grid gets one cell deep.
    xy = rng.uniform(-5, 5, size=(90000, 2))
    plane = np.column_stack([xy, np.zeros(len(xy))]).astype(np.float32)
    case(root, "flat-plane-90k", [(plane, T1)],
         (plane + rng.normal(0, 0.004, plane.shape)).astype(np.float32))

    # 4. A single dense line, 70k collinear points: degenerate in two axes.
    t = rng.uniform(-10, 10, size=70000)
    line = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)]).astype(np.float32)
    case(root, "collinear-70k", [(line, I4)],
         (line + rng.normal(0, 0.004, line.shape)).astype(np.float32))

    # 5. 120k points where 99.9% sit in a 1 cm box and 0.1% are 200 m away: the
    #    extreme density ratio that breaks a single fixed cell size.
    core = rng.normal(0, 0.005, size=(119880, 3))
    tail = rng.uniform(-200, 200, size=(120, 3))
    mix = np.vstack([core, tail]).astype(np.float32)
    case(root, "density-extreme-120k", [(mix, I4)],
         (mix + rng.normal(0, 0.002, mix.shape)).astype(np.float32))

if __name__ == "__main__":
    main()
