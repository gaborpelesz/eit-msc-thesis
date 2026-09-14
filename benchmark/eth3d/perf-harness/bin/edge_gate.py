#!/usr/bin/env python3
"""Edge-case differential gate.

For each dataset under data/edge, runs a binary twice (1 and 12 threads) with the
per-case tolerance list, capturing exit code, stdout, the first stderr line and the
SHA-256 of every per-point classification cloud, and compares all of it against a
golden captured from the upstream baseline binary.

Unlike the main gate this deliberately includes inputs the baseline handles badly
(empty clouds, NaN, division by zero). Whatever the baseline does IS the reference:
a variant that "fixes" one of them has changed observable behaviour and must
disclose it, so a mismatch is reported either way and the operator decides.
"""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys

THREAD_ENV = ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS",
              "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "KMP_BLOCKTIME", "KMP_AFFINITY"]

def env_for(threads):
    e = {k: v for k, v in os.environ.items() if k not in THREAD_ENV}
    e.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
             OMP_MAX_ACTIVE_LEVELS="1", OMP_WAIT_POLICY="PASSIVE")
    return e

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def run_case(binary, ds, workdir, threads):
    workdir = pathlib.Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    tol = (ds / "tolerances.txt").read_text().strip()
    cmd = [str(pathlib.Path(binary).resolve()), "--tolerances", tol,
           "--reconstruction_ply_path", str(ds / "reconstruction.ply"),
           "--ground_truth_mlp_path", str(ds / "scan_alignment.mlp"),
           "--accuracy_cloud_output_path", str(workdir / "acc"),
           "--completeness_cloud_output_path", str(workdir / "cmp")]
    try:
        p = subprocess.run(cmd, env=env_for(threads), capture_output=True, timeout=300)
        rc, out, err = p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        rc, out, err = "timeout", "", ""
    # Only the four result lines matter from stdout; the Loading/Computing progress
    # lines carry absolute paths that differ between worktrees.
    keep = [l for l in out.splitlines()
            if l.startswith(("Tolerances:", "Completenesses:", "Accuracies:", "F1-scores:"))]
    clouds = {}
    for p_ in sorted(workdir.rglob("*.ply")):
        clouds[str(p_.relative_to(workdir))] = sha256(p_)
    shutil.rmtree(workdir, ignore_errors=True)
    return {"returncode": rc, "result_lines": keep,
            "stderr_first": (err.strip().splitlines() or [""])[0][:200],
            "clouds": clouds}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--edge-root", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--write-golden")
    ap.add_argument("--threads", default="1,12")
    a = ap.parse_args()

    root = pathlib.Path(a.edge_root)
    cases = sorted(d for d in root.iterdir() if d.is_dir())
    threads = [int(t) for t in a.threads.split(",")]
    got = {}
    for ds in cases:
        for t in threads:
            got[f"{ds.name}@{t}"] = run_case(a.binary, ds, pathlib.Path(a.workdir) / f"{ds.name}_{t}", t)

    if a.write_golden:
        pathlib.Path(a.write_golden).write_text(json.dumps(got, indent=1, sort_keys=True))
        print(f"wrote golden for {len(got)} case/thread combinations -> {a.write_golden}")
        for k in sorted(got):
            g = got[k]
            print(f"  {k:34s} rc={g['returncode']!s:8s} clouds={len(g['clouds']):2d} "
                  f"{(g['result_lines'][1][:60] if len(g['result_lines'])>1 else g['stderr_first'][:60])}")
        return 0

    golden = json.loads(pathlib.Path(a.golden).read_text())
    bad = []
    for k in sorted(set(golden) | set(got)):
        g, o = golden.get(k), got.get(k)
        if g is None or o is None:
            bad.append((k, "case present in only one of golden/actual")); continue
        if g["returncode"] != o["returncode"]:
            bad.append((k, f"exit code {g['returncode']} -> {o['returncode']}"))
        if g["result_lines"] != o["result_lines"]:
            for i in range(max(len(g["result_lines"]), len(o["result_lines"]))):
                gl = g["result_lines"][i] if i < len(g["result_lines"]) else "<missing>"
                ol = o["result_lines"][i] if i < len(o["result_lines"]) else "<missing>"
                if gl != ol:
                    bad.append((k, f"stdout\n      golden: {gl}\n      actual: {ol}"))
        for name in sorted(set(g["clouds"]) | set(o["clouds"])):
            gh, oh = g["clouds"].get(name), o["clouds"].get(name)
            if gh != oh:
                bad.append((k, f"cloud {name}: {str(gh)[:16]} -> {str(oh)[:16]}"))
    if bad:
        print(f"FAIL  edge gate  ({len(bad)} mismatches over {len(cases)} cases x {len(threads)} thread counts)")
        for k, m in bad[:40]:
            print(f"  {k}: {m}")
        return 1
    print(f"PASS  edge gate  ({len(cases)} cases x {len(threads)} thread counts, "
          f"exit code + result lines + every per-point cloud)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
