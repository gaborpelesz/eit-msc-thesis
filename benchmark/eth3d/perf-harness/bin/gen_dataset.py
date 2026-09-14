#!/usr/bin/env python3
"""Synthetic ETH3D-shaped dataset generator for the multi-view-evaluation harness.

Produces, into <out>/:
    scan1.ply ... scanN.ply   ground-truth terrestrial laser scans, scanner-local
                              coordinates, binary_little_endian float32 x/y/z
    scan_alignment.mlp        MeshLab project with one MLMesh per scan, carrying
                              the scanner's global_T_mesh (yaw-only rotation +
                              translation, i.e. a levelled tripod -- the accuracy
                              code assumes scan-local z is up)
    reconstruction.ply        the "MVS output": the union of the GT scans mapped to
                              global coordinates, with a fraction of points dropped,
                              Gaussian measurement noise added, and a small fraction
                              displaced as gross outliers
    manifest.json             seed, parameters, point counts and SHA-256 of every file

Why a ray-cast scene instead of random points: ComputeAccuracy bins every scan
point into a 2048x1024 azimuth/inclination grid and walks an angular neighbourhood
of that grid per reconstruction point. Only points that actually lie on rays
emanating from the scanner origin reproduce the real occupancy of that grid, and
hence the real cost of the hot loop. Uniform noise in a box would put a wildly
unrepresentative number of points in each cell.
"""
import argparse, hashlib, json, os, struct, sys
import numpy as np


# ---------------------------------------------------------------- scene geometry

def _ray_box_inside(orig, dirs, lo, hi):
    """Nearest positive exit t of rays starting inside an axis-aligned box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - orig) * inv
        t2 = (hi - orig) * inv
    tmax = np.minimum(np.maximum(t1, t2), np.inf).min(axis=1)
    return np.where(tmax > 0, tmax, np.inf)


def _ray_box_outside(orig, dirs, lo, hi):
    """Nearest positive entry t of rays against an axis-aligned box, from outside."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - orig) * inv
        t2 = (hi - orig) * inv
        tn = np.maximum(t1, t2).min(axis=1)
        tf = np.minimum(t1, t2).max(axis=1)
    hit = (tn >= tf) & (tf > 0)
    return np.where(hit, tf, np.inf)


def _ray_sphere(orig, dirs, centre, radius):
    oc = orig - centre                      # (3,) or (n,3)
    b = np.einsum("ij,j->i", dirs, oc) if oc.ndim == 1 else np.einsum("ij,ij->i", dirs, oc)
    c = float(oc @ oc) - radius * radius
    disc = b * b - c
    ok = disc > 0
    t = np.full(dirs.shape[0], np.inf)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    tn = -b - sq
    tf = -b + sq
    t_hit = np.where(tn > 0, tn, tf)
    return np.where(ok & (t_hit > 0), t_hit, np.inf)


def build_scene(rng, extent):
    """A room plus interior clutter. Returns (room_lo, room_hi, boxes, spheres)."""
    w, d, h = extent
    room_lo = np.array([-w / 2, -d / 2, 0.0])
    room_hi = np.array([w / 2, d / 2, h])
    boxes, spheres = [], []
    for _ in range(9):                      # furniture-sized boxes on the floor
        sz = rng.uniform([0.4, 0.4, 0.4], [1.8, 1.8, 2.2])
        ctr = rng.uniform(room_lo + 1.0, room_hi - 1.0)
        ctr[2] = sz[2] / 2
        boxes.append((ctr - sz / 2, ctr + sz / 2))
    for _ in range(6):                      # spheres/columns: curved surfaces
        r = rng.uniform(0.25, 0.9)
        ctr = rng.uniform(room_lo + 1.5, room_hi - 1.5)
        ctr[2] = rng.uniform(0.6, h - 0.8)
        spheres.append((ctr, r))
    return room_lo, room_hi, boxes, spheres


def cast_scan(scene, origin, yaw, n_az, n_inc, rng, range_noise_m, dropout, max_range):
    """Cast a regular azimuth x inclination lattice. Returns scanner-LOCAL points."""
    room_lo, room_hi, boxes, spheres = scene
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False, dtype=np.float64)
    # Avoid the exact poles: a real scanner has a blind zenith/nadir cone.
    inc = np.linspace(0.08 * np.pi, 0.94 * np.pi, n_inc, dtype=np.float64)
    AZ, INC = np.meshgrid(az, inc, indexing="ij")
    AZ = AZ.ravel(); INC = INC.ravel()
    s = np.sin(INC)
    d_local = np.stack([s * np.cos(AZ), s * np.sin(AZ), np.cos(INC)], axis=1)

    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    d_global = d_local @ R.T

    t = _ray_box_inside(origin, d_global, room_lo, room_hi)
    for lo, hi in boxes:
        t = np.minimum(t, _ray_box_outside(origin, d_global, lo, hi))
    for ctr, r in spheres:
        t = np.minimum(t, _ray_sphere(origin, d_global, ctr, r))

    keep = np.isfinite(t) & (t <= max_range)
    if dropout > 0:                          # non-returns: dark/specular surfaces
        keep &= rng.random(t.shape[0]) >= dropout
    t = t[keep]
    d_local = d_local[keep]
    t = t + rng.normal(0.0, range_noise_m, t.shape[0])   # scanner range precision
    return (d_local * t[:, None]).astype(np.float32), R


# ---------------------------------------------------------------------- ply / mlp

def write_ply_xyz(path, pts):
    assert pts.dtype == np.float32 and pts.ndim == 2 and pts.shape[1] == 3
    hdr = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(np.ascontiguousarray(pts).tobytes())


def write_mlp(path, entries):
    rows = []
    for label, filename, T in entries:
        m = "\n".join(" ".join(repr(float(T[i][j])) for j in range(4)) for i in range(4))
        rows.append(
            f'  <MLMesh label="{label}" filename="{filename}">\n'
            f"   <MLMatrix44>\n{m}\n</MLMatrix44>\n"
            f"  </MLMesh>\n"
        )
    with open(path, "w") as f:
        f.write('<!DOCTYPE MeshLabDocument>\n<MeshLabProject>\n <MeshGroup>\n')
        f.writelines(rows)
        f.write(" </MeshGroup>\n</MeshLabProject>\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- main

PRESETS = {
    # name:      (n_scans, n_az, n_inc, room extent)
    "tiny":   (2, 700,  350,  (12.0, 10.0, 4.0)),
    "bench":  (3, 1500, 750,  (14.0, 11.0, 4.5)),
    "large":  (4, 2600, 1300, (16.0, 13.0, 5.0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="bench", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--recon-keep", type=float, default=0.88,
                    help="fraction of GT points surviving into the reconstruction")
    ap.add_argument("--recon-noise", type=float, default=0.006,
                    help="sigma [m] of isotropic Gaussian noise on recon points")
    ap.add_argument("--recon-outlier-frac", type=float, default=0.025)
    ap.add_argument("--recon-outlier-sigma", type=float, default=0.35)
    args = ap.parse_args()

    n_scans, n_az, n_inc, extent = PRESETS[args.preset]
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    scene = build_scene(rng, extent)
    w, d, h = extent

    entries, globals_ = [], []
    for i in range(n_scans):
        ang = 2 * np.pi * i / n_scans + 0.37
        origin = np.array([0.32 * w * np.cos(ang), 0.32 * d * np.sin(ang),
                           float(rng.uniform(1.45, 1.75))])
        yaw = float(rng.uniform(-np.pi, np.pi))
        pts_local, R = cast_scan(scene, origin, yaw, n_az, n_inc, rng,
                                 range_noise_m=0.0012, dropout=0.06,
                                 max_range=1.6 * max(w, d))
        name = f"scan{i + 1}.ply"
        write_ply_xyz(os.path.join(args.out, name), pts_local)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = origin
        entries.append((f"scan{i + 1}", name, T))
        globals_.append((pts_local.astype(np.float64) @ R.T) + origin)
        print(f"  {name}: {pts_local.shape[0]:,} points", file=sys.stderr)

    write_mlp(os.path.join(args.out, "scan_alignment.mlp"), entries)

    gt = np.concatenate(globals_, axis=0)
    n_keep = int(round(args.recon_keep * gt.shape[0]))
    idx = rng.permutation(gt.shape[0])[:n_keep]
    recon = gt[idx] + rng.normal(0.0, args.recon_noise, (n_keep, 3))
    n_out = int(round(args.recon_outlier_frac * n_keep))
    if n_out:
        oi = rng.permutation(n_keep)[:n_out]
        recon[oi] += rng.normal(0.0, args.recon_outlier_sigma, (n_out, 3))
    recon = rng.permutation(recon).astype(np.float32)   # MVS order is not scan order
    write_ply_xyz(os.path.join(args.out, "reconstruction.ply"), recon)
    print(f"  reconstruction.ply: {recon.shape[0]:,} points", file=sys.stderr)

    files = sorted(f for f in os.listdir(args.out) if f != "manifest.json")
    manifest = {
        "generator": "gen_dataset.py",
        "preset": args.preset, "seed": args.seed,
        "n_scans": n_scans, "n_az": n_az, "n_inc": n_inc, "room_extent": list(extent),
        "recon_keep": args.recon_keep, "recon_noise_sigma_m": args.recon_noise,
        "recon_outlier_frac": args.recon_outlier_frac,
        "recon_outlier_sigma_m": args.recon_outlier_sigma,
        "gt_points_total": int(gt.shape[0]), "recon_points": int(recon.shape[0]),
        "files": {f: {"bytes": os.path.getsize(os.path.join(args.out, f)),
                      "sha256": sha256(os.path.join(args.out, f))} for f in files},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps({k: manifest[k] for k in
                      ("preset", "seed", "gt_points_total", "recon_points")}))


if __name__ == "__main__":
    main()
