"""Re-score a campaign's stored point clouds with a different evaluator build.

A campaign must not mix evaluator builds (methods.yaml, global_deviations), but
`bench run --rerun` re-runs the method on the GPU as well -- hours of compute to
change a scoring step that reads an artefact already on disk. This re-scores
every stored cloud in one campaign with one evaluator and records the result
next to the original.

Nothing is overwritten. `run.json` keeps the numbers the campaign produced;
this writes `rescore.json` beside it and, for any run the original scored too,
records whether the two agree. Disagreement is reported, never silently
resolved: two evaluator builds differing on the same bytes is a finding, not a
retry.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench import evaluate as ev  # noqa: E402
from bench import spec as sp  # noqa: E402

CELLS = ("accuracy", "completeness", "f1")


def cells_equal(a, b):
    """Exact equality on every printed cell, keyed by tolerance.

    Exact, not approximate: the evaluator prints six significant digits and the
    two builds are claimed to be equivalent, so any visible difference is the
    thing worth knowing about. A tolerance present in one and not the other is
    a mismatch rather than a skipped comparison.

    A run whose evaluation failed stores `quality: []`, not null, so emptiness
    has to mean "nothing to compare against" here. Reading it as a set of zero
    tolerances would report every newly scoreable run as a disagreement, which
    is the opposite of what it is.
    """
    if not a or not b:
        return None
    ka = {round(r["tolerance"], 6): r for r in a}
    kb = {round(r["tolerance"], 6): r for r in b}
    if ka.keys() != kb.keys():
        return False
    return all(ka[t][c] == kb[t][c] for t in ka for c in CELLS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--image", required=True, help="evaluator image tag to score with")
    ap.add_argument("--evaluator-sha", required=True)
    ap.add_argument("--only", help="substring filter on the run key")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = sp.load(args.campaign_dir / "spec.yaml")
    spec.image = args.image

    runs = sorted(d for d in args.campaign_dir.iterdir() if d.is_dir() and (d / "run.json").is_file())
    if args.only:
        runs = [d for d in runs if args.only in d.name]

    summary, failures, mismatches = [], [], []
    for d in runs:
        rec = json.loads((d / "run.json").read_text())
        ply = rec.get("point_cloud_path")
        if not ply or not Path(ply).is_file():
            print(f"SKIP  {d.name}: no stored cloud")
            continue
        if args.dry_run:
            print(f"would score {d.name} <- {ply}")
            continue

        print(f"scoring {d.name} ... ", end="", flush=True)
        try:
            quality = ev.evaluate(spec, ply, rec["scene"], rec["width"],
                                  log_dir=None, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            print(f"FAILED: {exc}")
            failures.append((d.name, str(exc)))
            (d / "rescore.json").write_text(json.dumps(
                {"evaluator_sha": args.evaluator_sha, "image": args.image,
                 "status": "eval_failed", "error": str(exc), "quality": None}, indent=2, sort_keys=True))
            continue

        agree = cells_equal(rec.get("quality"), quality)
        prim = next((q for q in quality if abs(q["tolerance"] - rec["primary_tolerance"]) < 1e-9), None)
        out = {
            "evaluator_sha": args.evaluator_sha,
            "image": args.image,
            "status": "ok",
            "quality": quality,
            "f1_primary": prim["f1"] if prim else None,
            "accuracy_primary": prim["accuracy"] if prim else None,
            "completeness_primary": prim["completeness"] if prim else None,
            "original_evaluator_sha": rec.get("evaluator_sha"),
            "original_status": rec.get("status"),
            "agrees_with_original": agree,
        }
        (d / "rescore.json").write_text(json.dumps(out, indent=2, sort_keys=True))
        tag = {True: "same", False: "DIFFERS", None: "new"}[agree]
        print(f"f1={out['f1_primary']}  [{tag}]")
        if agree is False:
            mismatches.append(d.name)
        summary.append(out | {"run": d.name})

    if args.dry_run:
        return 0

    (args.campaign_dir / "rescore-summary.json").write_text(json.dumps(
        {"evaluator_sha": args.evaluator_sha, "image": args.image,
         "runs": summary, "mismatches": mismatches, "failures": failures},
        indent=2, sort_keys=True))
    print(f"\n{len(summary)} scored, {len(mismatches)} differing, {len(failures)} failed")
    if mismatches:
        print("DIFFERING:", ", ".join(mismatches))
    return 1 if (mismatches or failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
