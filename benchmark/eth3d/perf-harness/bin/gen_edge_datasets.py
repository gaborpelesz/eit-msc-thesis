#!/usr/bin/env python3
"""Generate small adversarial datasets for the round-2 edge gate.

The bench and tiny datasets are well-formed: no empty clouds, no non-finite
coordinates, no points sitting exactly on a tolerance boundary, no coordinates
far from the origin, always six tolerances. A variant can therefore pass the
main gate and still change behaviour on any of those. Each case below isolates
one such input class. The reference is whatever the UPSTREAM baseline binary
does with it -- including crashing or printing nan; the point is only that a
variant must do the same thing.
"""
import numpy as np, pathlib, sys

def write_ply_xyz(path, pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {pts.shape[0]}\n"
           "property float x\nproperty float y\nproperty float z\nend_header\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(hdr); f.write(np.ascontiguousarray(pts).tobytes())

def write_mlp(path, entries):
    rows = []
    for label, filename, T in entries:
        m = "\n".join(" ".join(repr(float(T[i][j])) for j in range(4)) for i in range(4))
        rows.append(f'  <MLMesh label="{label}" filename="{filename}">\n'
                    f"   <MLMatrix44>\n{m}\n</MLMatrix44>\n  </MLMesh>\n")
    with open(path, "w") as f:
        f.write('<!DOCTYPE MeshLabDocument>\n<MeshLabProject>\n <MeshGroup>\n')
        f.writelines(rows); f.write(" </MeshGroup>\n</MeshLabProject>\n")

I4 = np.eye(4)
# A small rotation+translation, so the scan frame is not the global frame: the
# transform is on the value path for both metrics.
th = 0.37
R = np.array([[np.cos(th), -np.sin(th), 0.], [np.sin(th), np.cos(th), 0.], [0., 0., 1.]])
T1 = np.eye(4); T1[:3, :3] = R; T1[:3, 3] = [0.31, -0.17, 0.05]

def case(root, name, scans, recon, tolerances="0.01,0.02,0.05,0.1,0.2,0.5"):
    d = root / name; d.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (pts, Tm) in enumerate(scans):
        fn = f"scan{i}.ply"; write_ply_xyz(d / fn, pts); entries.append((f"s{i}", fn, Tm))
    write_mlp(d / "scan_alignment.mlp", entries)
    write_ply_xyz(d / "reconstruction.ply", recon)
    (d / "tolerances.txt").write_text(tolerances + "\n")
    print(f"{name:18s} scans={[len(p) for p,_ in scans]} recon={len(recon)} tol={tolerances}")

def main():
    root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260914)
    base = rng.uniform(-1.0, 1.0, size=(400, 3)).astype(np.float32)

    # 1. Empty reconstruction: ComputeCompleteness has an explicit early return for
    #    it (the kd-tree build crashes on an empty cloud) and ComputeAccuracy does not.
    case(root, "empty-recon", [(base, T1)], np.zeros((0, 3), np.float32))

    # 2. Single reconstruction point: k=1 nearest-neighbour over a one-point index,
    #    and every voxel tally has exactly one cell.
    case(root, "single-recon-point", [(base, T1)], np.array([[0.1, 0.2, 0.3]], np.float32))

    # 3. Empty scan cloud alongside a populated one: a cell whose point count is zero
    #    is a division by zero in the completeness average.
    case(root, "empty-scan", [(np.zeros((0, 3), np.float32), I4), (base, T1)],
         base + rng.normal(0, 0.004, base.shape).astype(np.float32))

    # 4. A single tolerance: the inner loop bound `tolerances_count - 2` underflows
    #    size_t at one tolerance, and the histogram stride becomes 1.
    case(root, "single-tolerance", [(base, T1)],
         base + rng.normal(0, 0.004, base.shape).astype(np.float32), tolerances="0.05")

    # 5. Non-finite coordinates in both clouds: NaN and +/-Inf reach the spherical
    #    conversion, the float-to-int voxel conversion and the kd-tree query.
    bad = np.vstack([base[:50],
                     np.array([[np.nan, 0.1, 0.2], [0.3, np.inf, 0.4], [0.5, 0.6, -np.inf],
                               [np.nan, np.nan, np.nan], [0.0, 0.0, 0.0]], np.float32)])
    case(root, "non-finite", [(bad.astype(np.float32), I4)], bad.astype(np.float32))

    # 6. Coordinates ~1e5 m from the origin: the voxel cell index is a float->int
    #    conversion, so the integer cell coordinates get large and the hash spreads
    #    differently; also stresses any assumption about grid extent.
    far = (base + np.float32(1.0e5)).astype(np.float32)
    case(root, "far-from-origin", [(far, I4)],
         (far + rng.normal(0, 0.004, far.shape)).astype(np.float32))

    # 7. Reconstruction points placed at EXACTLY the tolerance distances from scan
    #    points. Every comparison in both metrics is strict '<' against a squared
    #    tolerance, so these land precisely on the branch boundary.
    anchor = np.zeros((6, 3), np.float32)
    for i, t in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
        anchor[i] = [np.float32(i * 3.0), 0.0, 0.0]
    exact = anchor.copy()
    for i, t in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
        exact[i] = [np.float32(i * 3.0) + np.float32(t), 0.0, 0.0]
    case(root, "tolerance-boundary", [(anchor, I4)], exact)

    # 8. Many exactly coincident points: every duplicate hashes to one voxel cell and
    #    every nearest-neighbour distance is an exact tie.
    dup = np.repeat(base[:20], 40, axis=0).astype(np.float32)
    case(root, "coincident", [(dup, I4)], dup)

    # 9. Unsorted / duplicated tolerance list, which std::sort must normalise.
    case(root, "tolerance-dupes", [(base, T1)],
         (base + rng.normal(0, 0.004, base.shape)).astype(np.float32),
         tolerances="0.2,0.01,0.05,0.01,0.5")

    # 10. Beam-boundary case. The accuracy metric classifies a reconstruction point
    #     against a laser beam whose radius at distance d is
    #         beam_start_radius_meters + d * tan(beam_divergence_halfangle_deg)
    #     Defaults: 0.5*0.00225 m and 0.011 deg, so at d = 5 m the radius is
    #     0.001125 + 5*tan(0.011 deg) = 0.0020845 m. Reconstruction points are placed
    #     at lateral offsets straddling that radius, which is what makes this dataset
    #     sensitive to a perturbation of the beam model itself -- the other nine cases
    #     are not, and a negative control that only rescales the beam passes them all.
    d = 5.0
    beam_r = 0.5 * 0.00225 + d * np.tan(np.deg2rad(0.011))
    scan_pts, rec_pts = [], []
    for i in range(60):
        ang = 2.0 * np.pi * i / 60.0
        # One scan point straight out along the ray, at distance d.
        ray = np.array([np.cos(ang), np.sin(ang), 0.0])
        scan_pts.append(ray * d)
        # Lateral direction, perpendicular to the ray and to z.
        lat = np.array([-np.sin(ang), np.cos(ang), 0.0])
        for k, f in enumerate([0.98, 0.995, 1.0, 1.005, 1.015, 1.03]):
            rec_pts.append(ray * d + lat * (beam_r * f))
    case(root, "beam-boundary",
         [(np.array(scan_pts, np.float32), I4)], np.array(rec_pts, np.float32))

if __name__ == "__main__":
    main()
