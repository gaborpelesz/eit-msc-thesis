"""Do two reconstructions fail to reach the SAME ground truth, or merely as much?

A completeness number matching to a fraction of a point is weak evidence that
two methods share a mechanism: equal-sized holes in different places produce the
same number. This scores every arm against ONE ground-truth sample in one order
and reports the overlap of their miss masks, so "the same misses" is a
measurement rather than an inference.

It deliberately uses none of ETH3D's machinery, for the same reason [[F-076]]
does not: an argument about what the evaluator rewards cannot be settled with
the evaluator. The completeness printed here is therefore a raw point fraction,
not the evaluator's cell-mean, and its LEVEL will differ from a scored run. Only
the ordering and the mask geometry are meant to carry across.

Two nulls ship with it, both of which have to fail before an overlap means
anything:

--difficulty  Ranks ground truth by how far the BASELINE arm's nearest point
              lies, and takes as many of the hardest as the reference arm
              misses. If that alone reproduces the overlap, the agreement is
              difficulty ordering and the arm under test adds nothing.
--thin        Keeps one ground-truth point per 1 cm voxel, which is the
              evaluator's own unit. Raw scan density over-weights whatever the
              scanner stood closest to -- usually the ground -- and an overlap
              that survives thinning is not an artefact of that weighting.

The independence null is always printed: the expected IoU if each arm missed
points at its own observed per-band rate but independently of the other.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage3d import gt_points, cloud_points  # noqa: E402


def masks(mlp, arms, stride, tol):
    G = gt_points(mlp, stride)
    out = {}
    for name, ply in arms.items():
        C = cloud_points(ply)
        d, _ = cKDTree(C).query(G, k=1, workers=-1, distance_upper_bound=1.0)
        out[name] = (np.where(np.isfinite(d), d, np.inf), C.shape[0])
    return G, out


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def report(ref, other, sel=None):
    a, b = (ref, other) if sel is None else (ref[sel], other[sel])
    inter = (a & b).sum()
    return iou(a, b), inter / max(1, a.sum()), inter / max(1, b.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mlp", help="the scene's scan_alignment.mlp")
    ap.add_argument("arms", nargs="+", metavar="NAME=PLY",
                    help="first is the reference every other is scored against")
    ap.add_argument("--stride", type=int, default=4,
                    help="take every Nth scan point; 4 is ~4 M on an ETH3D scene")
    ap.add_argument("--tolerance", type=float, default=0.20)
    ap.add_argument("--bands", type=int, default=12,
                    help="equal-count height bands, 0 to skip")
    ap.add_argument("--baseline", metavar="NAME",
                    help="arm to rank difficulty by for the --difficulty null")
    ap.add_argument("--difficulty", action="store_true")
    ap.add_argument("--thin", action="store_true")
    args = ap.parse_args()

    arms = dict(a.split("=", 1) for a in args.arms)
    ref_name = next(iter(arms))
    G, dist = masks(args.mlp, arms, args.stride, args.tolerance)
    miss = {k: (d > args.tolerance) for k, (d, _) in dist.items()}
    print(f"GT sample {G.shape[0]:,} (stride {args.stride}), tolerance "
          f"{args.tolerance} m, reference {ref_name}\n")
    for k, (_, n) in dist.items():
        print(f"  {k:<16} cloud {n:>12,}  reached {1 - miss[k].mean():.4f}  "
              f"missed {miss[k].sum():,}")

    ref = miss[ref_name]
    print(f"\nmiss-mask overlap against {ref_name}")
    for k, m in miss.items():
        if k == ref_name:
            continue
        i, r, p = report(ref, m)
        print(f"  {k:<16} IoU {i:.3f}  recall {r:.3f}  precision {p:.3f}")

    if args.bands:
        z = G[:, 2]
        edges = np.quantile(z, np.linspace(0, 1, args.bands + 1))
        print(f"\nreached by height band (z, m), {args.bands} equal-count bands")
        print("  band                 n" + "".join(f"{k:>16s}" for k in miss))
        num = den = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (z >= a) & (z < b)
            if not sel.sum():
                continue
            print(f"  {a:6.2f}..{b:6.2f} {sel.sum():>8,}"
                  + "".join(f"{1 - miss[k][sel].mean():>16.4f}" for k in miss))
            for k in miss:
                if k == ref_name:
                    continue
                p, q, n = ref[sel].mean(), miss[k][sel].mean(), sel.sum()
                num += n * p * q
                den += n * (p + q - p * q)
        others = len(miss) - 1
        if others:
            print(f"\n  E[IoU] if each arm missed at its own per-band rate but "
                  f"independently: {num / max(den, 1e-9):.3f}")

    if args.difficulty:
        base = args.baseline or [k for k in miss if k != ref_name][-1]
        order = np.argsort(-dist[base][0], kind="stable")
        hard = np.zeros(G.shape[0], bool)
        hard[order[:int(ref.sum())]] = True
        i, r, _ = report(ref, hard)
        print(f"\ndifficulty null: the {ref.sum():,} hardest points for {base}"
              f"\n  IoU {i:.3f}  recall {r:.3f}")

    if args.thin:
        cell = np.floor(G.astype(np.float64) / 0.01).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        keep = np.zeros(G.shape[0], bool)
        keep[first] = True
        print(f"\none point per 1 cm voxel: {keep.sum():,} of {G.shape[0]:,}")
        # Coverage under thinning is the quantity that can disagree in SIGN with
        # the raw fraction above: raw counts weight a surface by how densely the
        # scanner happened to hit it, and one cell per centimetre is what the
        # ETH3D evaluator's completeness actually averages over.
        for k, m in miss.items():
            print(f"  {k:<16} reached {1 - m[keep].mean():.4f}")
        for k, m in miss.items():
            if k == ref_name:
                continue
            i, r, p = report(ref, m, keep)
            print(f"  {k:<16} IoU {i:.3f}  recall {r:.3f}  precision {p:.3f}")


if __name__ == "__main__":
    main()
