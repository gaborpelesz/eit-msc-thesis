"""What fraction of the laser scan lies outside each method's depth range?

Round 4 dismissed the depth range by comparing range WIDTHS. That is the wrong
quantity: a range matters only through the ground truth it excludes, and two
ranges of equal width can exclude very different amounts of it. It also compared
the converter's file values, but Quorum widens them at load
(`src/io/cam_txt.h:42-43`, the lineage's 0.6 / 1.2), so the numbers compared
were never the ones either matcher used.

Both methods compute the range identically -- 0.75 x the 1st percentile and
1.25 x the 99th of the sparse points' camera-frame Z (CUMVS
`cuda_multi_view_stereo.cpp:47-50, 552-553`; converter
`colmap2mvsnet_acm_perf.py:509`). They differ in two places that this measures:
CUMVS percentiles over every sparse point with Z > 0 and applies no further
widening; the converter percentiles over only the tracks that image observes,
and Quorum then widens by 0.6 / 1.2.

Depth range is a hard constraint in CUMVS -- propagation rejects out-of-range
hypotheses (`.cu:804`) and init and refinement sample only inside it -- so
ground truth outside the range is ground truth the matcher cannot reach at any
cost. This reports that fraction per reference view, for both methods, over the
pixels where the scan actually gives a depth.

Views are joined by image NAME. The two pipelines number images in opposite
directions (see `select_views.py`), and an index means nothing across them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_depth import load_mlp, ply_vertex_layout  # noqa: E402
from coverage3d import cloud_points  # noqa: E402

LAMBDA_MIN, LAMBDA_MAX = 0.75, 1.25
C_MIN, C_MAX = 0.01, 0.99
QUORUM_WIDEN_MIN, QUORUM_WIDEN_MAX = 0.6, 1.2


def sparse_points(ply):
    # CUMVS writes this one through VTK, in ASCII (F-074 is the same trap on its
    # dense cloud), so it needs the reader that handles both encodings rather
    # than gt_depth's binary-only layout.
    return cloud_points(str(ply)).astype(np.float64)


def to_camera(P, R, t):
    """world -> camera, matching CUMVS's projectToCamera.

    `global_poses.json` stores the camera's pose IN the world: the projection at
    `cuda_multi_view_stereo.cpp:192-193` transposes R and negates t before use,
    so applying the stored pair forwards puts every point in the wrong frame and
    collapses the depth spread to nothing.
    """
    return (P - t) @ R


def cumvs_range(P, R, t):
    """CUMVS: every sparse point in front of the camera, no frustum test."""
    z = to_camera(P, R, t)[:, 2]
    z = np.sort(z[z > 0])
    if z.size == 0:
        return np.nan, np.nan
    lo = LAMBDA_MIN * z[int(np.floor(C_MIN * z.size))]
    hi = LAMBDA_MAX * z[int(np.floor(C_MAX * z.size))]
    return float(lo), float(hi)


def read_cams(cam_dir, names):
    """The converter's per-image cam files, widened as Quorum's loader widens."""
    out = {}
    for i, nm in enumerate(names):
        p = Path(cam_dir) / f"{i:08d}_cam.txt"
        if not p.exists():
            continue
        tok = p.read_text().split()
        j = tok.index("extrinsic") if "extrinsic" in tok else 0
        # depth_min and depth_interval follow the intrinsic block's 9 values
        k = tok.index("intrinsic")
        vals = tok[k + 10:k + 14]
        lo = float(vals[0])
        hi = float(vals[3]) if len(vals) > 3 else np.nan
        out[nm] = (lo * QUORUM_WIDEN_MIN, hi * QUORUM_WIDEN_MAX)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cumvs_dir", help="CUMVS initializer output (input_images.json etc.)")
    ap.add_argument("mlp", help="the scene's scan_alignment.mlp")
    ap.add_argument("--prepared", help="Quorum's prepared folder, for its cams/")
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    d = json.load(open(Path(args.cumvs_dir) / "input_images.json"))
    poses = json.load(open(Path(args.cumvs_dir) / "global_poses.json"))
    P = sparse_points(Path(args.cumvs_dir) / "point_cloud_sparse.ply")

    G = []
    for ply, M in load_mlp(args.mlp):
        off, n, dt = ply_vertex_layout(ply)
        a = np.memmap(ply, dtype=dt, mode="r", offset=off, shape=(n,))[::args.stride]
        Q = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
        G.append(Q @ M[:3, :3].T + M[:3, 3])
    G = np.concatenate(G)
    print(f"sparse {P.shape[0]:,}  scan sample {G.shape[0]:,}\n")

    pose = {p["id"]: p for p in poses["global_poses"]}
    names = [Path(e["filename"]).name for e in d["input_images"]]
    cams = read_cams(Path(args.prepared) / "cams", sorted(names)) if args.prepared else {}

    print(f"{'image':<16}{'CUMVS lo':>10}{'hi':>9}{'quorum lo':>11}{'hi':>9}"
          f"{'GT<lo C':>9}{'GT<lo Q':>9}{'GT>hi C':>9}{'GT>hi Q':>9}")
    rows = []
    for e in d["input_images"]:
        pid = e["id"]
        nm = Path(e["filename"]).name
        pm = pose.get(pid)
        if pm is None:
            continue
        R = np.array(pm["R"]["data"], dtype=np.float64).reshape(3, 3)
        t = np.array(pm["t"]["data"], dtype=np.float64).reshape(3)
        clo, chi = cumvs_range(P, R, t)
        # The frustum test is not optional. Without it this counts every scan
        # point in front of the camera PLANE -- including everything beside and
        # behind the view -- and near-camera points outside the field of view
        # then read as "below the near plane", which is how a first run of this
        # put Quorum at 19% excluded.
        Xc = to_camera(G, R, t)
        K = np.array(e["K"]["data"], dtype=np.float64).reshape(3, 3)
        front = Xc[:, 2] > 0
        Xc = Xc[front]
        uv = Xc @ K.T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]
        w, h = e.get("width"), e.get("height")
        if not w or not h:
            w, h = int(2 * K[0, 2]), int(2 * K[1, 2])
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        z = Xc[inside, 2]
        if z.size == 0:
            continue
        qlo, qhi = cams.get(nm, (np.nan, np.nan))
        r = (nm, clo, chi, qlo, qhi,
             float((z < clo).mean()), float((z < qlo).mean()) if qlo == qlo else np.nan,
             float((z > chi).mean()), float((z > qhi).mean()) if qhi == qhi else np.nan)
        rows.append(r)
        print(f"{r[0]:<16}{r[1]:>10.2f}{r[2]:>9.2f}{r[3]:>11.2f}{r[4]:>9.2f}"
              f"{r[5]:>9.4f}{r[6]:>9.4f}{r[7]:>9.4f}{r[8]:>9.4f}")

    a = np.array([[r[5], r[6], r[7], r[8]] for r in rows], dtype=np.float64)
    print(f"\nmean over {len(rows)} views:  below-range CUMVS {np.nanmean(a[:,0]):.4f}  "
          f"quorum {np.nanmean(a[:,1]):.4f}   above-range CUMVS {np.nanmean(a[:,2]):.4f}  "
          f"quorum {np.nanmean(a[:,3]):.4f}")
    print(f"total outside:            CUMVS {np.nanmean(a[:,0]+a[:,2]):.4f}   "
          f"quorum {np.nanmean(a[:,1]+a[:,3]):.4f}")


if __name__ == "__main__":
    main()
