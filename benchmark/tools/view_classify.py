"""Classify one view's pixels by what a method reconstructed there, against the
laser scan rendered into the same view by gt_depth.py.

The campaign's accuracy and completeness are scalars over a whole scene, and
F-075 showed they are not even weighted the way intuition assumes. This answers
a different question -- *where* on the image a method wins or loses -- which is
what separates a depth-map deficit from a fusion deficit, and what a scalar
cannot express.

Five outcomes per pixel, from the cross of "did the scan see it" and "did the
method put something there":

    hit     scan saw it, method within tolerance          -- the only good cell
    wrong   scan saw it, method disagrees beyond tolerance
    missed  scan saw it, method has nothing               -- coverage failure
    extra   scan saw nothing, method has something        -- FREE under ETH3D
    void    scan saw nothing, method has nothing

`extra` is called out separately because ETH3D charges nothing for it (F-077:
45% of what the eye sees in CUMVS's playground cloud lands here), so a method
can look denser to a viewer without that density reaching either metric.

Input is a PLY by default, which measures the cloud AFTER fusion. Pass a depth
map instead (--depth-npz) to measure the matcher alone; the classification is
identical either way, which is the point -- the same instrument reads both
sides of the fusion boundary.
"""
import argparse, sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_cloud import ply_layout, read_ascii, read_cam

COLOURS = {                      # BGR-free, plain RGB
    'hit':    (0.16, 0.68, 0.38),
    'wrong':  (0.85, 0.24, 0.55),
    'missed': (0.90, 0.22, 0.20),
    'extra':  (0.95, 0.78, 0.25),
    'void':   (0.10, 0.11, 0.13),
}


def rasterize_ply(ply, cam, W, H, scale, chunk=4_000_000):
    """Nearest-surface depth per pixel, at the same geometry as gt_depth.py."""
    R, t, K = read_cam(cam)
    s = 1.0 / scale
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
    off, n, dt, fmt = ply_layout(ply)
    names = [nm for nm, _ in dt]
    ci = {nm: i for i, nm in enumerate(names)}

    if fmt == 'ascii':
        def blocks():
            for blk in read_ascii(ply, off, n, len(names)):
                yield blk[:, [ci['x'], ci['y'], ci['z']]].copy()
    else:
        arr = np.memmap(ply, dtype=np.dtype(dt), mode='r', offset=off, shape=(n,))

        def blocks():
            for a in range(0, n, chunk):
                b = arr[a:min(a + chunk, n)]
                yield np.stack([b['x'], b['y'], b['z']], axis=1).astype(np.float32)

    zbuf = np.full(W * H, np.inf, dtype=np.float32)
    Rf, tf = R.astype(np.float32), t.astype(np.float32)
    for P in blocks():
        P = P @ Rf.T + tf
        z = P[:, 2]
        ok = z > 1e-3
        P, z = P[ok], z[ok]
        if not z.size:
            continue
        u = (fx * P[:, 0] / z + cx).astype(np.int32)
        v = (fy * P[:, 1] / z + cy).astype(np.int32)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        np.minimum.at(zbuf, v[ok] * W + u[ok], z[ok])
    return zbuf.reshape(H, W)


def classify(gt_depth, gt_count, m_depth, tol, min_count):
    known = np.isfinite(gt_depth) & (gt_count >= min_count)
    present = np.isfinite(m_depth)
    # inf-inf where neither has a value; compare only where both do, so the
    # invalid-subtract warning does not hide a real nan in the depth data.
    both = np.isfinite(m_depth) & np.isfinite(gt_depth)
    within = np.zeros_like(both)
    within[both] = np.abs(m_depth[both] - gt_depth[both]) <= tol
    return {
        'hit': known & present & within,
        'wrong': known & present & ~within,
        'missed': known & ~present,
        'extra': ~known & present,
        'void': ~known & ~present,
    }, known, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_npz")
    ap.add_argument("cam")
    ap.add_argument("--ply")
    ap.add_argument("--depth-npz", help="measure the depth map instead of the cloud")
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--min-count", type=int, default=1,
                    help="scan points a pixel needs before its GT depth is trusted")
    ap.add_argument("--label", default="")
    ap.add_argument("-o", "--out", help="write a classified PNG here")
    ap.add_argument("--npz-out", help="write the masks for downstream analysis")
    a = ap.parse_args()

    g = np.load(a.gt_npz)
    gt_depth, gt_count = g['depth'], g['count']
    H, W = gt_depth.shape

    if a.depth_npz:
        d = np.load(a.depth_npz)['depth'].astype(np.float32)
        m_depth = np.where(d > 0, d, np.inf)
    else:
        m_depth = rasterize_ply(a.ply, a.cam, W, H, a.scale)

    masks, known, present = classify(gt_depth, gt_count, m_depth, a.tol, a.min_count)
    total = gt_depth.size
    nk = int(known.sum())
    stats = {k: int(v.sum()) for k, v in masks.items()}
    cov = stats['hit'] / nk if nk else 0.0
    acc = stats['hit'] / max(stats['hit'] + stats['wrong'], 1)

    print(f"{a.label or Path(a.ply or a.depth_npz).name}")
    print(f"  scan-known pixels {nk} ({nk*100.0/total:.1f}% of frame)")
    for k in ('hit', 'wrong', 'missed', 'extra', 'void'):
        print(f"  {k:7s} {stats[k]:8d}  {stats[k]*100.0/total:5.1f}% of frame")
    print(f"  view completeness (hit / scan-known)     {cov*100:.2f}%")
    print(f"  view accuracy     (hit / scored present) {acc*100:.2f}%")
    print(f"  extra / present                          "
          f"{stats['extra']*100.0/max(int(present.sum()),1):.1f}%")

    if a.npz_out:
        np.savez_compressed(a.npz_out, **{k: v for k, v in masks.items()},
                            known=known, present=present, m_depth=m_depth)
    if a.out:
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for k, c in COLOURS.items():
            rgb[masks[k]] = c
        Image.fromarray((rgb * 255).astype(np.uint8)).save(a.out, quality=92)
        print(f"  -> {a.out}")


if __name__ == '__main__':
    main()
