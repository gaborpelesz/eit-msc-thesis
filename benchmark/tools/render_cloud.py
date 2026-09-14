"""Splat a binary PLY into a shaded image from an ACMMP-format camera.

Geometry-only shading (screen-space normals + eye-dome lighting): the LeanMVS
clouds carry no colour or normals, so any colour-based render would compare
methods on different data. One point per sample, no splat radius, so holes in
the reconstruction stay holes on screen.
"""
import sys, struct, numpy as np
from PIL import Image

FMT = {'float': ('f4', 4), 'float32': ('f4', 4), 'double': ('f8', 8),
       'uchar': ('u1', 1), 'uint8': ('u1', 1), 'char': ('i1', 1),
       'int': ('i4', 4), 'uint': ('u4', 4), 'short': ('i2', 2), 'ushort': ('u2', 2)}


def ply_layout(path):
    with open(path, 'rb') as f:
        hdr, n, props = b'', 0, []
        while True:
            line = f.readline()
            hdr += line
            t = line.decode('ascii', 'replace').strip().split()
            if t and t[0] == 'element' and t[1] == 'vertex':
                n = int(t[2])
            elif t and t[0] == 'property':
                props.append((t[1], t[2]))
            elif t and t[0] == 'end_header':
                break
        return f.tell(), n, [(nm, FMT[ty][0]) for ty, nm in props]


def read_cam(path):
    txt = open(path).read().split()
    i = txt.index('extrinsic')
    E = np.array(txt[i + 1:i + 17], dtype=np.float64).reshape(4, 4)
    j = txt.index('intrinsic')
    K = np.array(txt[j + 1:j + 10], dtype=np.float64).reshape(3, 3)
    return E[:3, :3], E[:3, 3], K


def render(ply, cam, out, width=1400, ss=2, src_width=3200, chunk=4_000_000, colour=True):
    R, t, K = read_cam(cam)
    s = (width * ss) / src_width
    fx, fy = K[0, 0] * s, K[1, 1] * s
    cx, cy = K[0, 2] * s, K[1, 2] * s
    W = width * ss
    H = int(round(2 * cy)) if cy > 0 else W * 2 // 3
    H = max(H, 8)
    off, n, dt = ply_layout(ply)
    arr = np.memmap(ply, dtype=np.dtype(dt), mode='r', offset=off, shape=(n,))

    names = {n for n, _ in dt}
    has_rgb = colour and {'red', 'green', 'blue'} <= names

    zbuf = np.full(W * H, np.inf, dtype=np.float32)
    pbuf = np.zeros((W * H, 3), dtype=np.float32)
    cbuf = np.zeros((W * H, 3), dtype=np.float32) if has_rgb else None
    Rf, tf = R.astype(np.float32), t.astype(np.float32)

    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        blk = arr[a:b]
        P = np.stack([blk['x'], blk['y'], blk['z']], axis=1).astype(np.float32)
        C = (np.stack([blk['red'], blk['green'], blk['blue']], axis=1).astype(np.float32) / 255.0
             if has_rgb else None)
        P = P @ Rf.T + tf
        z = P[:, 2]
        ok = z > 1e-3
        P, z = P[ok], z[ok]
        if has_rgb: C = C[ok]
        u = (fx * P[:, 0] / z + cx).astype(np.int32)
        v = (fy * P[:, 1] / z + cy).astype(np.int32)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        P, z, u, v = P[ok], z[ok], u[ok], v[ok]
        if has_rgb: C = C[ok]
        order = np.argsort(-z, kind='stable')          # nearest written last
        zo, Po, uo, vo = z[order], P[order], u[order], v[order]
        Co = C[order] if has_rgb else None
        # A point covers 2x2 supersampled pixels, i.e. exactly one output pixel
        # after the downsample below. At one pixel per point the background
        # shows through the gaps as speckle, and a sparser cloud then reads as
        # darker rather than as sparser -- which would invert the very
        # comparison these renders exist to make. The splat fills the gap
        # between samples; it never extends a surface past its own points,
        # because every written pixel still has to win the depth test.
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ii = np.clip(vo + dv, 0, H - 1) * W + np.clip(uo + du, 0, W - 1)
            zc = np.full(W * H, np.inf, dtype=np.float32)
            pc = np.zeros((W * H, 3), dtype=np.float32)
            zc[ii] = zo
            pc[ii] = Po
            if has_rgb:
                cc = np.zeros((W * H, 3), dtype=np.float32)
                cc[ii] = Co
            m = zc < zbuf
            zbuf[m] = zc[m]
            pbuf[m] = pc[m]
            if has_rgb: cbuf[m] = cc[m]

    Z = zbuf.reshape(H, W)
    Pb = pbuf.reshape(H, W, 3)
    hit = np.isfinite(Z)

    # Normals from a 3x3 mean of the hit positions: a per-pixel cross product
    # on raw samples turns depth noise into black speckle and swamps the
    # completeness difference the comparison is about. Holes stay holes -
    # smoothing runs over hits only and never fills an empty pixel.
    w = hit.astype(np.float32)
    acc = np.zeros_like(Pb); cnt = np.zeros_like(Z)
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            acc += np.roll(np.roll(Pb * w[..., None], sy, axis=0), sx, axis=1)
            cnt += np.roll(np.roll(w, sy, axis=0), sx, axis=1)
    Pb = np.divide(acc, cnt[..., None], out=Pb.copy(), where=cnt[..., None] > 0)

    dx = np.zeros_like(Pb); dy = np.zeros_like(Pb)
    dx[:, 1:-1] = Pb[:, 2:] - Pb[:, :-2]
    dy[1:-1, :] = Pb[2:, :] - Pb[:-2, :]
    N = np.cross(dx, dy)
    ln = np.linalg.norm(N, axis=2, keepdims=True)
    N = np.divide(N, ln, out=np.zeros_like(N), where=ln > 1e-12)
    if N[..., 2].sum() > 0:
        N = -N

    L = np.array([0.35, -0.55, -0.76], dtype=np.float32)
    L /= np.linalg.norm(L)
    lam = np.clip((N * L).sum(axis=2), 0.0, 1.0)

    # Eye-dome lighting: darken where a neighbour is markedly nearer.
    lz = np.log(np.where(hit, Z, np.nan))
    edl = np.zeros_like(lz)
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(np.roll(lz, sy, axis=0), sx, axis=1)
        edl += np.nan_to_num(np.maximum(0.0, lz - sh), nan=0.0)
    edl = np.exp(-7.0 * edl)

    shade = (0.30 + 0.70 * lam) * (0.55 + 0.45 * edl)
    shade = np.clip(shade, 0.0, 1.0)

    if has_rgb:
        # Albedo carries the image; the shading term only modulates it, so a
        # surface still reads as lit without the geometry swamping the colour.
        albedo = cbuf.reshape(H, W, 3)
        rgb = albedo * (0.40 + 0.75 * shade[..., None])
    else:
        ink = np.array([0.14, 0.17, 0.22], dtype=np.float32)
        lit = np.array([0.97, 0.95, 0.90], dtype=np.float32)
        rgb = ink + (lit - ink) * shade[..., None]
    # Auto-exposure over the hit pixels only. ETH3D frames vary by several stops
    # between scenes -- an indoor office against an outdoor facade -- and a fixed
    # gain renders one of them unreadable. Anchoring the 97th percentile rather
    # than the maximum keeps a few specular points from darkening the whole
    # frame. Background is excluded so a sparse cloud is not brightened merely
    # for covering less of the image, which would flatter exactly the clouds
    # this report is meant to distinguish.
    if hit.any():
        lum = rgb[hit] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        ref = np.percentile(lum, 97)
        if ref > 1e-6:
            rgb = rgb * (0.82 / ref)
    rgb = np.clip(rgb, 0, 1) ** (1 / 1.15)

    bg = np.array([0.055, 0.062, 0.075], dtype=np.float32)
    rgb = np.where(hit[..., None], rgb, bg)

    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((W // ss, H // ss), Image.LANCZOS)
    img.save(out, quality=90)
    print(f"{out}  {n} pts  {hit.sum() * 100.0 / hit.size:.1f}% coverage  "
          f"{'rgb' if has_rgb else 'geometry-only'}")


if __name__ == '__main__':
    render(sys.argv[1], sys.argv[2], sys.argv[3],
           width=int(sys.argv[4]) if len(sys.argv) > 4 else 1400)
