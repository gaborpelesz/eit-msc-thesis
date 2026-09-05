"""ETH3D undistorted scenes: distinct image dimensions per scene and how many
images use each. Reads COLMAP cameras.txt / images.txt only."""
import glob, os, sys, collections
root = sys.argv[1]
print(f"{'scene':<14}{'imgs':>5}  {'cams':>4}  dimensions (W x H : #images)")
for cal in sorted(glob.glob(f"{root}/*/*_dslr_undistorted/*/dslr_calibration_undistorted")):
    scene = cal.split("/")[-2]
    cams = {}
    for l in open(f"{cal}/cameras.txt"):
        if l.startswith("#"): continue
        f = l.split(); cams[f[0]] = (int(f[2]), int(f[3]))
    use = collections.Counter()
    lines = [l for l in open(f"{cal}/images.txt") if not l.startswith("#") and l.strip()]
    for l in lines[0::2]:
        use[cams[l.split()[8]]] += 1
    dims = ", ".join(f"{w}x{h}:{n}" for (w, h), n in sorted(use.items(), key=lambda x: -x[1]))
    print(f"{scene:<14}{sum(use.values()):>5}  {len(cams):>4}  {dims}")
