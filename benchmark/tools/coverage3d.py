"""Reproduce ETH3D completeness independently, and locate the misses in 3D.

The per-view classifier answers a different question from the evaluator: it asks
whether the NEAREST surface along a ray is within tolerance, so a spurious
foreground point makes a pixel miss even when the ground-truth surface is
covered from elsewhere. The evaluator asks whether any reconstruction point lies
within tolerance of a ground-truth point, in 3D. Those two can and do order
methods differently, so the hole has to be measured the evaluator's way.

Doing it here rather than reading the evaluator's number also prices the
possibility that the hole is an artefact of the evaluator's ASCII PLY path
(F-074), on which CUMVS has failed once already.
"""
import sys, json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/pelesz/thesis/benchmark/tools')
from render_cloud import ply_layout, read_ascii
from gt_depth import load_mlp, ply_vertex_layout


def gt_points(mlp, stride=1):
    out = []
    for ply, M in load_mlp(mlp):
        off, n, dt = ply_vertex_layout(ply)
        arr = np.memmap(ply, dtype=dt, mode='r', offset=off, shape=(n,))
        arr = arr[::stride]
        P = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
        out.append(P @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(out).astype(np.float32)


def cloud_points(ply, chunk=4_000_000):
    off, n, dt, fmt = ply_layout(ply)
    names = [nm for nm, _ in dt]
    ci = {nm: i for i, nm in enumerate(names)}
    if fmt == 'ascii':
        blocks = [b[:, [ci['x'], ci['y'], ci['z']]].copy()
                  for b in read_ascii(ply, off, n, len(names))]
    else:
        arr = np.memmap(ply, dtype=np.dtype(dt), mode='r', offset=off, shape=(n,))
        blocks = [np.stack([arr[a:a+chunk]['x'], arr[a:a+chunk]['y'], arr[a:a+chunk]['z']],
                           axis=1).astype(np.float32) for a in range(0, n, chunk)]
    return np.concatenate(blocks)


if __name__ == '__main__':
    mlp, ply, label = sys.argv[1], sys.argv[2], sys.argv[3]
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    G = gt_points(mlp, stride)
    C = cloud_points(ply)
    tree = cKDTree(C)
    d, _ = tree.query(G, k=1, workers=-1, distance_upper_bound=1.0)
    res = {'label': label, 'gt_points': int(G.shape[0]), 'cloud_points': int(C.shape[0])}
    for t in (0.01, 0.02, 0.05, 0.10, 0.20):
        res[f'comp@{t}'] = float((d <= t).mean())
    miss = ~np.isfinite(d) | (d > 0.20)
    res['unreached@20cm'] = float(miss.mean())
    if miss.sum():
        M = G[miss]
        res['miss_bbox'] = [[float(x) for x in M.min(0)], [float(x) for x in M.max(0)]]
        res['miss_centroid'] = [float(x) for x in M.mean(0)]
        res['miss_z_pct'] = [float(np.percentile(M[:, 2], p)) for p in (10, 50, 90)]
    print(json.dumps(res))
