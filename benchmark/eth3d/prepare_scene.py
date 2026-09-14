#!/usr/bin/env python3
"""Produce `<scene>_<width>/` from an ETH3D high-res scene.

The five scenes the campaign already uses were prepared before this script
existed; its rules were recovered from them so that scenes prepared now are
interchangeable with those:

  * `H' = floor(H * width / W)` -- `office` is the case that distinguishes
    floor from round (2132.647 -> 2132).
  * Intrinsics scale ANISOTROPICALLY: `fx`,`cx` by `width/W`, `fy`,`cy` by
    `H'/H`. A single uniform factor reproduces `fx`/`cx` and gets `fy`/`cy`
    wrong, which is easy to miss because the error is under a quarter pixel.
  * `images.txt` and `points3D.txt` are copied verbatim. The 2D observations
    in `images.txt` are NOT rescaled -- the converters read poses and the
    point cloud, never those coordinates.
  * JPEG is re-encoded at quality 100, 4:4:4. Recovered from file size: the
    prepared frames sit within 2% of that setting and 2-10x off every other.

The resampling filter is NOT recoverable -- the prepared frames are re-encoded
JPEGs, and every candidate filter agrees with them to under one grey level on
average, below the JPEG noise floor. HAMMING is the closest of the six and is
what this script uses; the choice is a disclosed difference between scene sets.
"""
import argparse, math, shutil, sys
from pathlib import Path
from PIL import Image

def scale_cameras(src, dst, width):
    out, geom = [], {}
    for line in src.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            out.append(line); continue
        f = line.split()
        cid, model, W, H = f[0], f[1], int(f[2]), int(f[3])
        fx, fy, cx, cy = map(float, f[4:8])
        Hp = math.floor(H * width / W)
        sx, sy = width / W, Hp / H
        geom[cid] = (width, Hp)
        out.append(f"{cid} {model} {width} {Hp} {fx*sx} {fy*sy} {cx*sx} {cy*sy}")
    dst.write_text("\n".join(out) + "\n")
    return geom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--root", default="/mnt/samsung2tb/eth3d")
    ap.add_argument("--width", type=int, default=3200)
    ap.add_argument("--filter", default="HAMMING")
    a = ap.parse_args()
    root, s, w = Path(a.root), a.scene, a.width
    src, dst = root / s, root / f"{s}_{w}"
    if not src.is_dir(): sys.exit(f"no such scene: {src}")
    if dst.exists(): sys.exit(f"refusing to overwrite {dst}")

    cal_rel = Path(f"{s}_dslr_undistorted/{s}")
    src_cal, dst_cal = src / cal_rel, dst / cal_rel
    (dst_cal / "dslr_calibration_undistorted").mkdir(parents=True)
    geom = scale_cameras(src_cal / "dslr_calibration_undistorted/cameras.txt",
                         dst_cal / "dslr_calibration_undistorted/cameras.txt", w)
    for f in ("images.txt", "points3D.txt"):
        shutil.copy2(src_cal / "dslr_calibration_undistorted" / f,
                     dst_cal / "dslr_calibration_undistorted" / f)

    # Each image takes the size of ITS camera: a scene may carry several, and on
    # six of the thirteen ETH3D scenes they differ (F-021).
    cam_of = {}
    for line in (src_cal / "dslr_calibration_undistorted/images.txt").read_text().splitlines():
        f = line.split()
        # Two lines per image: the 10-field pose line, then the observations,
        # which come in triples. 10 is not a multiple of 3, so the field count
        # alone separates them.
        if line.startswith("#") or len(f) != 10: continue
        cam_of[f[9]] = f[8]

    filt = getattr(Image, a.filter)
    n = 0
    for name, cid in sorted(cam_of.items()):
        i, o = src_cal / "images" / name, dst_cal / "images" / name
        o.parent.mkdir(parents=True, exist_ok=True)
        Image.open(i).convert("RGB").resize(geom[cid], filt).save(
            o, "JPEG", quality=100, subsampling=0)
        n += 1
    shutil.copytree(src / f"{s}_dslr_scan_eval", dst / f"{s}_dslr_scan_eval")
    sizes = sorted(set(geom.values()))
    print(f"{s}: {n} images -> {dst}  sizes={sizes}")

main()
