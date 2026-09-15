"""Surface thickness per method, measured against laser ground-truth planes.

A critique of a metric cannot also be the evidence that the metric is wrong, so
this uses none of ETH3D's machinery: it asks only how thick each method makes a
surface the scanner says is flat.

The reference plane is fitted to the ground truth, not to the method being
measured, so the mean residual is a bias and not an artefact of fitting a cloud
to its own errors. Patches are selected on the ground truth alone for the same
reason -- selecting them on a reconstruction would pick that method's flat
regions and flatter it.

Thickness (MAD, RMS, p95) is sign-invariant, but the SIGNED mean is only a
"bias" once every patch normal points the same way relative to the sensor: SVD
returns a normal whose sign is arbitrary, so pooling residuals across patches
without orienting them first averages a quantity whose sign flips per patch.
Pass --orient-to with the scene's COLMAP calibration and each normal is turned
to face the nearest camera centre, which makes a positive bias mean "in front of
the true surface, toward the camera". Without it the tool still fixes the sign
deterministically (largest component positive) so runs reproduce, and prints the
bias column parenthesised to mark it as unidentified.

See F-076.
"""
import argparse
import glob
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_cloud import ply_layout, read_ascii  # noqa: E402


def load_xyz(path):
    off, n, dt, fmt = ply_layout(path)
    names = [nm for nm, _ in dt]
    ci = {nm: i for i, nm in enumerate(names)}
    if fmt == "ascii":
        out = np.empty((n, 3), np.float32)
        got = 0
        for blk in read_ascii(path, off, n, len(names)):
            k = blk.shape[0]
            out[got:got + k] = blk[:, [ci["x"], ci["y"], ci["z"]]]
            got += k
        return out[:got]
    a = np.memmap(path, dtype=np.dtype(dt), mode="r", offset=off, shape=(n,))
    return np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float32)


def fit_plane(P):
    c = P.mean(axis=0)
    _, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    return c, Vt[2], s


def orient(normal, centre, cameras):
    """Turn a plane normal to face the sensor, or to a fixed sign without one."""
    if cameras is None:
        k = int(np.argmax(np.abs(normal)))
        return normal if normal[k] >= 0 else -normal
    cam = cameras[int(np.argmin(((cameras - centre) ** 2).sum(axis=1)))]
    return normal if normal @ (cam - centre) >= 0 else -normal


def camera_centres(cal):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from select_views import read_colmap
    return read_colmap(cal)[1]


def select_patches(gt, box, min_points, max_thickness, min_extent, limit, cameras):
    q = np.floor(gt.astype(np.float64) / box).astype(np.int64)
    q -= q.min(axis=0)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    order = np.argsort(key, kind="stable")
    key, gt = key[order], gt[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    ends = np.concatenate((starts[1:], [key.size]))

    patches = []
    for a, b in zip(starts, ends):
        if b - a < min_points:
            continue
        Q = gt[a:b]
        c, normal, s = fit_plane(Q)
        rms = s[2] / np.sqrt(Q.shape[0])
        # The second singular value rejects cubes whose points are collinear or
        # clustered: a plane through those is unconstrained in one direction.
        if rms < max_thickness and s[1] / np.sqrt(Q.shape[0]) > min_extent:
            patches.append((rms, c, orient(normal, c, cameras)))
    patches.sort(key=lambda t: t[0])
    return patches[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ground_truth", help="scan PLY, or a completeness visualization "
                                         "cloud (one point per scan point)")
    ap.add_argument("clouds", help="glob over the reconstructions to compare")
    ap.add_argument("--box", type=float, default=1.5, help="patch edge, m")
    ap.add_argument("--band", type=float, default=0.15,
                    help="half-thickness of the slab kept around the plane, m")
    ap.add_argument("--min-gt-points", type=int, default=4000)
    ap.add_argument("--max-thickness", type=float, default=0.004)
    ap.add_argument("--min-extent", type=float, default=0.15)
    ap.add_argument("--patches", type=int, default=25)
    ap.add_argument("--orient-to", metavar="COLMAP_DIR",
                    help="scene calibration directory; orients each plane normal "
                         "toward the nearest camera centre so the bias column is "
                         "signed relative to the sensor")
    args = ap.parse_args()

    cameras = camera_centres(args.orient_to) if args.orient_to else None
    gt = load_xyz(args.ground_truth)
    patches = select_patches(gt, args.box, args.min_gt_points,
                             args.max_thickness, args.min_extent, args.patches,
                             cameras)
    if not patches:
        raise SystemExit("no planar patches met the criteria; loosen --max-thickness")
    print(f"GT points {gt.shape[0]:,}; {len(patches)} planar patches "
          f"(GT RMS thickness {np.mean([p[0] for p in patches]) * 1000:.2f} mm)\n")
    del gt

    bias_col = "bias mm" if cameras is not None else "(bias)"
    print(f"{'method':<11} {'patches':>7} {'points':>10} {bias_col:>8} "
          f"{'MAD mm':>7} {'RMS mm':>7} {'p95 mm':>7}")
    for f in sorted(glob.glob(args.clouds)):
        method = os.path.basename(f).split("__")[0]
        P = load_xyz(f)
        residuals, used = [], 0
        for _, c, normal in patches:
            d = P - c
            in_box = (np.abs(d) < args.box / 2).all(axis=1)
            if in_box.sum() < 200:
                continue
            r = d[in_box] @ normal
            r = r[np.abs(r) < args.band]
            if r.size < 200:
                continue
            used += 1
            residuals.append(r)
        if not residuals:
            print(f"{method:<11} {'--':>7}")
            continue
        r = np.concatenate(residuals)
        print(f"{method:<11} {used:>7} {r.size:>10,} {r.mean() * 1000:>8.2f} "
              f"{np.median(np.abs(r - np.median(r))) * 1000:>7.2f} "
              f"{np.sqrt((r ** 2).mean()) * 1000:>7.2f} "
              f"{np.percentile(np.abs(r), 95) * 1000:>7.2f}", flush=True)
        del P


if __name__ == "__main__":
    main()
