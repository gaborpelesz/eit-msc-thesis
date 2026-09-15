"""Rasterize an ETH3D laser scan into a depth map for one camera.

The campaign measures completeness after fusion, against the scan as a point
set. That cannot say whether a method's coverage deficit is in its depth maps
or in its fusion, because by then the two are indistinguishable. This renders
the ground truth into a single view so a method's own depth map -- or a cloud
rasterized the same way -- can be compared pixel by pixel.

The scans are NOT in the evaluation frame on disk: scan_alignment.mlp carries a
per-scan 4x4 that the official evaluator applies (bench/evaluate.py:119 passes
it the .mlp, not the plys), so this applies it too. Skipping it puts playground's
scan1 0.73 m off along x and silently scores every method against a shifted
world.

Occlusion is handled by z-buffering at 2x the requested resolution and then
min-pooling, which lets a densely sampled foreground surface close the gaps its
own sampling leaves before a background point can show through them. That is a
mitigation, not a guarantee: at a depth discontinuity the scan's own sampling
density decides, so `count` ships alongside `depth` and the consumer is expected
to threshold on it.
"""
import argparse, re
import numpy as np
from pathlib import Path
from xml.etree import ElementTree

FMT = {'float': ('f4', 4), 'float32': ('f4', 4), 'double': ('f8', 8),
       'uchar': ('u1', 1), 'uint8': ('u1', 1), 'char': ('i1', 1),
       'int': ('i4', 4), 'uint': ('u4', 4), 'short': ('i2', 2), 'ushort': ('u2', 2)}


def ply_vertex_layout(path):
    """Offset, count and dtype of the vertex element only.

    ETH3D's scans carry a trailing `element camera`; its properties must not be
    folded into the vertex stride or every coordinate after the first is
    garbage.
    """
    with open(path, 'rb') as f:
        n, props, elements = 0, [], 0
        while True:
            line = f.readline()
            if not line:
                break
            t = line.decode('ascii', 'replace').strip().split()
            if not t:
                continue
            if t[0] == 'element':
                elements += 1
                if t[1] == 'vertex':
                    n = int(t[2])
            elif t[0] == 'property' and elements == 1:
                props.append((t[2], FMT[t[1]][0]))
            elif t[0] == 'end_header':
                break
        return f.tell(), n, np.dtype(props)


def load_mlp(mlp_path):
    """(ply path, 4x4 world transform) for each mesh the alignment names."""
    root = ElementTree.parse(mlp_path).getroot()
    base = Path(mlp_path).parent
    out = []
    for mesh in root.iter('MLMesh'):
        M = np.eye(4)
        node = mesh.find('MLMatrix44')
        if node is not None and node.text:
            vals = [float(v) for v in node.text.split()]
            if len(vals) == 16:
                M = np.array(vals, dtype=np.float64).reshape(4, 4)
        out.append((base / mesh.get('filename'), M))
    return out


def read_cam(path):
    txt = Path(path).read_text().split()
    i = txt.index('extrinsic')
    E = np.array(txt[i + 1:i + 17], dtype=np.float64).reshape(4, 4)
    j = txt.index('intrinsic')
    K = np.array(txt[j + 1:j + 10], dtype=np.float64).reshape(3, 3)
    return E[:3, :3], E[:3, 3], K


def render(mlp, cam, width, height, scale, chunk=8_000_000):
    """Depth (inf where the scan saw nothing) and point count, at `scale` down."""
    R, t, K = read_cam(cam)
    ss = 2                                   # supersample, then min-pool
    W, H = int(round(width / scale)) * ss, int(round(height / scale)) * ss
    s = 1.0 / scale * ss
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s

    zbuf = np.full(W * H, np.inf, dtype=np.float32)
    cnt = np.zeros(W * H, dtype=np.int32)
    Rf, tf = R.astype(np.float32), t.astype(np.float32)

    for ply, M in load_mlp(mlp):
        off, n, dt = ply_vertex_layout(ply)
        arr = np.memmap(ply, dtype=dt, mode='r', offset=off, shape=(n,))
        Mf = M.astype(np.float32)
        for a in range(0, n, chunk):
            blk = arr[a:min(a + chunk, n)]
            P = np.stack([blk['x'], blk['y'], blk['z']], axis=1).astype(np.float32)
            P = P @ Mf[:3, :3].T + Mf[:3, 3]      # scan -> world (the .mlp)
            P = P @ Rf.T + tf                      # world -> camera
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
            idx = v[ok] * W + u[ok]
            zo = z[ok]
            np.minimum.at(zbuf, idx, zo)
            np.add.at(cnt, idx, 1)

    Z = zbuf.reshape(H, W)
    C = cnt.reshape(H, W)
    # Min-pool the supersampled buffer: the nearest surface in each 2x2 wins,
    # so a foreground gap is closed by its own neighbours rather than by
    # whatever lies behind it.
    Z = np.minimum(np.minimum(Z[0::2, 0::2], Z[1::2, 0::2]),
                   np.minimum(Z[0::2, 1::2], Z[1::2, 1::2]))
    C = C[0::2, 0::2] + C[1::2, 0::2] + C[0::2, 1::2] + C[1::2, 1::2]
    return Z, C.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlp")
    ap.add_argument("cam")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--scale", type=float, default=4.0)
    a = ap.parse_args()
    Z, C = render(a.mlp, a.cam, a.width, a.height, a.scale)
    np.savez_compressed(a.out, depth=Z, count=C)
    known = np.isfinite(Z)
    print(f"{a.out}  {Z.shape[1]}x{Z.shape[0]}  "
          f"{known.mean()*100:.1f}% of pixels have scan support  "
          f"median {np.median(C[known]):.0f} pts/px  "
          f"depth {np.nanmin(Z[known]):.2f}..{np.nanmax(Z[known]):.2f} m")


if __name__ == '__main__':
    main()
