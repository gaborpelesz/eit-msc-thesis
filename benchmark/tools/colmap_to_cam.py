#!/usr/bin/env python3
"""Write an ACMMP-format camera file for one image of a prepared ETH3D scene.

`bench run` deletes each run's work tree unless `keep_intermediates` is set, and
with it the converter's `cams/` directory -- so the cameras a cloud was
reconstructed through do not survive the campaign that produced it. The COLMAP
calibration under `<scene>_<width>/` does survive, permanently, and carries the
same information. This reconstructs the camera from it, so a cloud can still be
rendered from a real viewpoint months later.

COLMAP's quaternion and translation are already world-to-camera, which is the
convention ACMMP's `extrinsic` block uses, so no inversion is involved.
"""
import argparse, math
from pathlib import Path

def quat_to_R(qw, qx, qy, qz):
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return [
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--root", default="/mnt/samsung2tb/eth3d")
    ap.add_argument("--width", type=int, default=3200)
    ap.add_argument("--image", help="image name; default is the middle one")
    ap.add_argument("--index", type=int, help="pick by position instead")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    cal = (Path(a.root) / f"{a.scene}_{a.width}" /
           f"{a.scene}_dslr_undistorted/{a.scene}/dslr_calibration_undistorted")
    cams = {}
    for line in (cal / "cameras.txt").read_text().splitlines():
        f = line.split()
        if line.startswith("#") or len(f) < 8: continue
        cams[f[0]] = tuple(map(float, f[4:8]))

    poses = []
    for line in (cal / "images.txt").read_text().splitlines():
        f = line.split()
        # 10-field pose lines only; the observation lines between them come in
        # triples, and 10 is not a multiple of 3.
        if line.startswith("#") or len(f) != 10: continue
        poses.append((f[9], tuple(map(float, f[1:8])), f[8]))
    poses.sort()
    if a.image:
        sel = [p for p in poses if p[0].endswith(a.image)]
        if not sel: raise SystemExit(f"no image matching {a.image}; have e.g. {poses[0][0]}")
        name, pose, cid = sel[0]
    else:
        name, pose, cid = poses[a.index if a.index is not None else len(poses)//2]

    qw, qx, qy, qz, tx, ty, tz = pose
    R = quat_to_R(qw, qx, qy, qz)
    fx, fy, cx, cy = cams[cid]
    rows = [f"{R[i][0]} {R[i][1]} {R[i][2]} {(tx,ty,tz)[i]}" for i in range(3)]
    Path(a.out).write_text(
        "extrinsic\n" + "\n".join(rows) + "\n0.0 0.0 0.0 1.0\n\n"
        f"intrinsic\n{fx} 0.0 {cx}\n0.0 {fy} {cy}\n0.0 0.0 1.0\n")
    print(f"{a.scene}: {name} (camera {cid}) -> {a.out}")

main()
