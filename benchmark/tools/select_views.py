"""Build a `pair.txt` with either the shared converter's view selection or CUMVS's.

The two are the same algorithm. Both score an image pair by the number of
shared sparse tracks and reject the pair outright when the 75th-percentile
triangulation angle over those tracks is below one degree; the ACM lineage's
Gaussian angle weighting is present in `colmap2mvsnet_acm_perf.py:328` only as
a commented-out line, replaced by `score += 1`.

They differ in one place. The converter takes the top `neighbours` by score
with no `score > 0` filter (`colmap2mvsnet_acm_perf.py:492`), so a reference
whose gate leaves fewer than that many survivors is **padded with pairs the
gate rejected**. CUMVS filters first (`app_initialize_ETH3D.cpp:199`) and emits
a shorter list, dropping the reference entirely when fewer than `min_neighbors`
survive. So the selection difference measured between the two methods is not a
difference of rule but of what happens after the rule says no.

This exists so that difference can be applied to any method the harness drives,
on any scene, without running CUMVS -- and so the `cumvs` mode can be validated
against CUMVS's own `view_id_sets.json` rather than assumed faithful.

That validation is joined by image NAME, never by index. The two pipelines
number images in different orders, and on the ETH3D scenes measured here those
orders are reversed (see `read_colmap`), so an index-to-index comparison is a
comparison of two different images that happens to typecheck. Every consumer of
this module's output is responsible for the same join.
"""
import argparse
import numpy as np
from pathlib import Path


def read_colmap(cal):
    """Camera centres and per-image sparse track ids, in COLMAP image-id order.

    This is the converter's order (`colmap2mvsnet_acm_perf.py:402` sorts on
    image id). CUMVS does NOT share it: it indexes by the order images appear in
    `images.txt` (`app_initialize_ETH3D.cpp:20`), and ETH3D writes that file in
    DESCENDING image id, so on playground and courtyard CUMVS index k is this
    module's index (n - 1 - k). An index is therefore meaningless across the two
    pipelines; join on the image name, which `names` carries for that purpose.
    """
    lines = (Path(cal) / "images.txt").read_text().splitlines()
    ids, names, centres, tracks = [], [], [], []
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#") or not l.strip():
            i += 1
            continue
        f = l.split()
        if len(f) != 10:
            i += 1
            continue
        iid = int(f[0])
        qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
        n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])
        obs = lines[i+1].split()
        pid = np.array([int(obs[k+2]) for k in range(0, len(obs), 3)], dtype=np.int64)
        ids.append(iid)
        names.append(f[9])
        centres.append(-R.T @ np.array([tx, ty, tz]))
        tracks.append(pid[pid >= 0])
        i += 2
    order = np.argsort(ids)
    return ([names[i] for i in order], np.array([centres[i] for i in order]),
            [np.unique(tracks[i]) for i in order])


def read_points(cal):
    xyz = {}
    for l in (Path(cal) / "points3D.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        xyz[int(f[0])] = (float(f[1]), float(f[2]), float(f[3]))
    keys = np.fromiter(xyz.keys(), dtype=np.int64)
    lut = np.full(keys.max() + 1, -1, dtype=np.int64)
    lut[keys] = np.arange(keys.size)
    return lut, np.array([xyz[k] for k in keys], dtype=np.float64)


def score_matrix(centres, tracks, lut, pts, angle_floor=1.0, pct=0.75):
    n = len(tracks)
    S = np.zeros((n, n), dtype=np.int64)
    for a in range(n - 1):
        for b in range(a + 1, n):
            inter = np.intersect1d(tracks[a], tracks[b], assume_unique=True)
            if inter.size == 0:
                continue
            P = pts[lut[inter]]
            v1, v2 = centres[a] - P, centres[b] - P
            cos = np.einsum("nk,nk->n", v1, v2) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
            ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            ang.sort()
            # Index, not interpolation: both implementations take the element at
            # int(0.75*n) of the sorted angles, so a percentile function with a
            # different convention would disagree on small intersections.
            if ang[int(ang.size * pct)] < angle_floor:
                continue
            S[a, b] = S[b, a] = inter.size
    return S


def select(S, mode, neighbours, min_neighbors=2):
    n = S.shape[0]
    out = []
    for i in range(n):
        order = np.argsort(-S[i], kind="stable")
        if mode == "cumvs":
            keep = [k for k in order if S[i, k] > 0][:neighbours]
            # CUMVS sorts by score only when it has to truncate, so a reference
            # with few enough survivors keeps them in image-id order. Reproduced
            # because the order reaches the matcher as the neighbour order.
            if len(keep) < neighbours:
                keep = sorted(keep)
            if len(keep) < min_neighbors:
                keep = []
        else:
            keep = list(order[:min(neighbours, n - 1)])
        out.append([(int(k), int(S[i, k])) for k in keep])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration", help="COLMAP dslr_calibration_undistorted directory")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--mode", choices=("cumvs", "acm"), default="cumvs")
    ap.add_argument("--neighbours", type=int, default=20)
    ap.add_argument("--min-neighbors", type=int, default=2)
    a = ap.parse_args()

    names, centres, tracks = read_colmap(a.calibration)
    lut, pts = read_points(a.calibration)
    S = score_matrix(centres, tracks, lut, pts)
    sel = select(S, a.mode, a.neighbours, a.min_neighbors)

    with open(a.out, "w") as f:
        f.write(f"{len(names)}\n")
        for i, row in enumerate(sel):
            f.write(f"{i}\n{len(row)} ")
            for k, s in row:
                f.write(f"{k} {s} ")
            f.write("\n")
    sizes = [len(r) for r in sel]
    zeros = sum(1 for r in sel for _, s in r if s == 0)
    print(f"{a.out}  mode={a.mode}  {len(names)} images  "
          f"set size min {min(sizes)} mean {np.mean(sizes):.1f} max {max(sizes)}  "
          f"gate-failing entries included: {zeros}")


if __name__ == "__main__":
    main()
