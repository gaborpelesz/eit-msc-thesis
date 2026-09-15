"""Split an ETH3D accuracy gap into weighting, footprint and point quality.

`ETH3DMultiViewEvaluation --accuracy_cloud_output_path` writes one point per
reconstruction point coloured by its verdict. This re-aggregates those verdicts
without changing a single one of them, so every difference it reports is
attributable to the aggregation alone.

Accuracy as ETH3D defines it (accuracy.cc:984-1000) is the unweighted mean over
occupied 1 cm cells of each cell's accurate fraction, over two half-shifted
grids. The cells come from the reconstruction, so the denominator moves with
the cloud being scored: a method emitting more points resolves its own error
distribution into more cells, and every cell votes once whether it holds one
point or a thousand. Completeness has no such asymmetry -- its cells come from
the ground truth, one fixed set for every method -- which is why only accuracy
is decomposed here.

The three rows beyond the first isolate the two causes:

  point-weighted      every point counts once, so the cell weighting is gone
  shared footprint    only 5 cm voxels both clouds occupy, so reaching further
                      into harder surface cannot contribute

What survives both is the like-for-like deficit. See F-075, F-076, F-077.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_cloud import ply_layout  # noqa: E402

VOX = 0.01
SHIFTS = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
# One lattice for both clouds, so a shared cell is the same cell in each. Large
# enough to keep every ETH3D scene's coordinates positive.
ORIGIN = np.array([-1000.0, -1000.0, -1000.0])


def load(path):
    """Positions and verdicts. Returns (all points, accurate, inaccurate)."""
    off, n, dt, _ = ply_layout(path)
    a = np.memmap(path, dtype=np.dtype(dt), mode="r", offset=off, shape=(n,))
    P = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    g, r = np.asarray(a["green"]), np.asarray(a["red"])
    return P, (g > 128) & (r < 128), (r > 128) & (g < 128)


def cell_mean(P, acc):
    """ETH3D's aggregation, reproduced. Matches the printed `Accuracies:`."""
    total, cells = 0.0, 0
    for shift in SHIFTS:
        q = np.floor((P - ORIGIN) / VOX + np.array(shift)).astype(np.int64)
        key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        order = np.argsort(key, kind="stable")
        key, a = key[order], acc[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
        per_cell = np.diff(np.concatenate((starts, [key.size])))
        total += float((np.add.reduceat(a.astype(np.int64), starts) / per_cell).sum())
        cells += per_cell.size
    return total / cells, cells // len(SHIFTS)


def coarse_key(P, coarse):
    q = np.floor((P - ORIGIN) / coarse).astype(np.int64)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud_a", type=Path, help="accuracy visualization PLY")
    ap.add_argument("cloud_b", type=Path)
    ap.add_argument("--names", nargs=2, default=("A", "B"))
    ap.add_argument("--coarse", type=float, default=0.05,
                    help="voxel edge at which the shared footprint is defined")
    args = ap.parse_args()

    na, nb = args.names
    loaded = []
    print(f"{'':<8} {'points':>12} {'accurate':>12} {'inaccurate':>11} {'unobserved':>12}")
    for name, path in ((na, args.cloud_a), (nb, args.cloud_b)):
        P, acc, inacc = load(path)
        n = P.shape[0]
        unobs = n - int(acc.sum()) - int(inacc.sum())
        print(f"{name:<8} {n:>12,} {int(acc.sum()):>12,} {int(inacc.sum()):>11,} "
              f"{unobs:>12,} ({unobs * 100.0 / n:4.1f}%)")
        valid = acc | inacc
        loaded.append((P[valid], acc[valid]))
    (A, aA), (B, aB) = loaded

    shared = np.intersect1d(np.unique(coarse_key(A, args.coarse)),
                            np.unique(coarse_key(B, args.coarse)),
                            assume_unique=True)
    mA = np.isin(coarse_key(A, args.coarse), shared)
    mB = np.isin(coarse_key(B, args.coarse), shared)
    fa, ca = cell_mean(A, aA)
    fb, cb = cell_mean(B, aB)
    sa, csa = cell_mean(A[mA], aA[mA])
    sb, csb = cell_mean(B[mB], aB[mB])

    print(f"\nshared {args.coarse:g} m voxels: {shared.size:,}")
    print(f"1 cm cells on shared surface: {na} {csa:,}  {nb} {csb:,}  "
          f"ratio {csb / max(1, csa):.2f}x\n")

    print(f"{'weighting / footprint':<34} {na:>9} {nb:>9} {'gap pp':>8}")
    rows = [("ETH3D: cell-mean, own footprint", fa, fb),
            ("cell-mean, shared footprint", sa, sb),
            ("point-weighted, own footprint", aA.mean(), aB.mean()),
            ("point-weighted, shared footprint", aA[mA].mean(), aB[mB].mean())]
    for label, x, y in rows:
        print(f"{label:<34} {x:>9.4f} {y:>9.4f} {(x - y) * 100:>8.2f}")

    print(f"\nreaching further        {(rows[0][1] - rows[0][2] - (rows[1][1] - rows[1][2])) * 100:>6.2f} pp")
    print(f"cell weighting          {((rows[1][1] - rows[1][2]) - (rows[3][1] - rows[3][2])) * 100:>6.2f} pp")
    print(f"worse points, same surface {(rows[3][1] - rows[3][2]) * 100:>6.2f} pp")


if __name__ == "__main__":
    main()
