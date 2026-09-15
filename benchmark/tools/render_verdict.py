"""Render an evaluator classification cloud with its verdict colours intact.

No shading and no auto-exposure: the point of the image is which verdict a
surface got, and any shading term would mix the three verdict colours into
shades that no longer read as categories. Nearest point wins the pixel, so what
is shown is the verdict of the surface actually facing the camera.
"""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/pelesz/thesis/benchmark/tools')
from render_cloud import ply_layout, read_cam

ACC = np.array([0.35, 0.78, 0.42], np.float32)     # accurate
INA = np.array([0.92, 0.26, 0.24], np.float32)     # inaccurate
UNO = np.array([0.30, 0.52, 0.90], np.float32)     # unobserved (not scored)


def main(ply, cam, out, width=1400, ss=2, src_width=3200, chunk=4_000_000):
    R, t, K = read_cam(cam)
    s = (width * ss) / src_width
    fx, fy, cx, cy = K[0, 0]*s, K[1, 1]*s, K[0, 2]*s, K[1, 2]*s
    W = width * ss
    H = max(int(round(2 * cy)), 8)
    off, n, dt, fmt = ply_layout(ply)
    a = np.memmap(ply, dtype=np.dtype(dt), mode='r', offset=off, shape=(n,))
    zbuf = np.full(W * H, np.inf, np.float32)
    cbuf = np.zeros((W * H, 3), np.float32)
    Rf, tf = R.astype(np.float32), t.astype(np.float32)
    tally = np.zeros(3, np.int64)

    for i in range(0, n, chunk):
        blk = a[i:min(i + chunk, n)]
        P = np.stack([blk['x'], blk['y'], blk['z']], axis=1).astype(np.float32)
        g, r = np.asarray(blk['green']), np.asarray(blk['red'])
        cls = np.where(g > 128, 0, np.where(r > 128, 1, 2)).astype(np.int8)
        P = P @ Rf.T + tf
        z = P[:, 2]
        ok = z > 1e-3
        P, z, cls = P[ok], z[ok], cls[ok]
        u = (fx * P[:, 0] / z + cx).astype(np.int32)
        v = (fy * P[:, 1] / z + cy).astype(np.int32)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        z, u, v, cls = z[ok], u[ok], v[ok], cls[ok]
        o = np.argsort(-z, kind='stable')
        z, u, v, cls = z[o], u[o], v[o], cls[o]
        col = np.stack([ACC, INA, UNO])[cls]
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ii = np.clip(v + dv, 0, H-1) * W + np.clip(u + du, 0, W-1)
            zc = np.full(W * H, np.inf, np.float32)
            cc = np.zeros((W * H, 3), np.float32)
            zc[ii] = z
            cc[ii] = col
            m = zc < zbuf
            zbuf[m] = zc[m]
            cbuf[m] = cc[m]

    hit = np.isfinite(zbuf).reshape(H, W)
    rgb = cbuf.reshape(H, W, 3)
    rgb = np.where(hit[..., None], rgb, np.array([0.06, 0.07, 0.09], np.float32))
    # Tally what is actually visible, which is what the reader is being shown.
    vis = rgb[hit]
    for k, c in enumerate((ACC, INA, UNO)):
        tally[k] = int((np.abs(vis - c).sum(axis=1) < 1e-3).sum())
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img.resize((W // ss, H // ss), Image.LANCZOS).save(out, quality=92)
    tot = max(1, tally.sum())
    print(f"{out}  visible: accurate {tally[0]*100.0/tot:4.1f}%  "
          f"inaccurate {tally[1]*100.0/tot:4.1f}%  unobserved {tally[2]*100.0/tot:4.1f}%")


main(sys.argv[1], sys.argv[2], sys.argv[3],
     width=int(sys.argv[4]) if len(sys.argv) > 4 else 1400)
