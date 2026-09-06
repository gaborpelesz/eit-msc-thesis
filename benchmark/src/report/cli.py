"""`report` -- generate the manuscript's result tables and figures.

    uv run python -m report render CAMPAIGN_DIR [CAMPAIGN_DIR ...]
                                  [--out DIR] [--manifest PATH] [--no-figures]
    uv run python -m report summary CAMPAIGN_DIR

`render` writes, per campaign, into `thesis/generated/`:

    results-<campaign>-summary.tex        provenance, n, wall time, F1, memory,
                                          status counts
    results-<campaign>-phases.tex         exclusive phase share per method
    results-<campaign>-f1-tolerances.tex  F1 at every evaluated tolerance
    figures/results-<campaign>-phase-shares.tex   stacked bars (pgfplots)
    figures/results-<campaign>-wall-f1.tex        wall time vs F1 (pgfplots)

`summary` prints the same aggregates as text, for checking a campaign without
touching the manuscript tree.

The package has no console-script entry in `pyproject.toml` on purpose:
`benchmark/pyproject.toml` and `benchmark/uv.lock` are hashed into the Docker
image's build manifest and `bench run` refuses to start a campaign whose
working tree differs from the image it is running, so registering the script
would halt the fleet until the image was rebuilt. `src/` is on `sys.path`
through `eval.pth`, so `python -m report` works today.
"""

import argparse
import sys
from pathlib import Path

from deviations import manifest as mf

from . import figures as fg
from . import load as ld
from . import tables as tb

THESIS_GENERATED = "thesis/generated"


def default_out_dir(manifest_path):
    return mf.repo_root(Path(manifest_path).parent) / THESIS_GENERATED


def render_campaign(campaign, out_dir, with_figures=True, command=tb.COMMAND):
    """Write every artefact for one campaign. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, disagreements = ld.phase_rows(campaign)

    written = [
        (out_dir / f"results-{campaign.name}-summary.tex", tb.summary(campaign, command)),
        (
            out_dir / f"results-{campaign.name}-phases.tex",
            tb.phases(campaign, rows, disagreements, command),
        ),
        (
            out_dir / f"results-{campaign.name}-f1-tolerances.tex",
            tb.tolerances(campaign, command),
        ),
    ]
    if with_figures:
        figure_dir = out_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        written += [
            (
                figure_dir / f"results-{campaign.name}-phase-shares.tex",
                fg.phase_shares(campaign, rows, command),
            ),
            (
                figure_dir / f"results-{campaign.name}-wall-f1.tex",
                fg.wall_vs_f1(campaign, command),
            ),
        ]
    for path, text in written:
        path.write_text(text)
    return [path for path, _ in written], disagreements


def cmd_render(args):
    manifest = mf.load(args.manifest)
    out_dir = Path(args.out) if args.out else default_out_dir(args.manifest)
    for directory in args.campaign_dir:
        campaign = ld.Campaign(directory, manifest)
        paths, disagreements = render_campaign(
            campaign, out_dir, with_figures=not args.no_figures
        )
        for path in paths:
            print(f"wrote {path}")
        counts = {}
        for record in campaign.records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        print(
            f"  {campaign.name}: {len(campaign.records)} runs, "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        for run_key, names in disagreements:
            print(
                f"  warning: {run_key} phases.txt disagrees with the stored totals "
                f"for {', '.join(names)}",
                file=sys.stderr,
            )
    return 0


def cmd_summary(args):
    manifest = mf.load(args.manifest)
    campaign = ld.Campaign(args.campaign_dir, manifest)
    counts = campaign.status_counts()
    print(campaign.footer(tb.COMMAND))
    print()
    header = (
        f"{'method':<20} {'provenance':<26} {'conf':<8} {'n':>6} {'wall s':>10} "
        f"{'F1':>8} {'peak MiB':>10}  status"
    )
    print(header)
    print("-" * len(header))
    rows = {(r["method"], r["configuration"]): r for r in campaign.summary_rows()}

    def cell(value, width, digits=1, scale=1.0):
        return "--".rjust(width) if value is None else f"{value / scale:>{width},.{digits}f}"

    for method in campaign.methods:
        for configuration in campaign.configurations:
            row = rows.get((method, configuration))
            if row is None:
                continue
            statuses = ", ".join(
                f"{k}={v}"
                for k, v in sorted((counts.get((method, configuration)) or {}).items())
            )
            completed = f"{row['n_ok']}/{row['runs']}"
            print(
                f"{method:<20} {campaign.provenance(method):<26} {configuration:<8} "
                f"{completed:>6} "
                f"{cell(row['wall_mean'], 10)} "
                f"{cell(row['f1_mean'], 8, digits=4)} "
                f"{cell(row['peak_device_bytes'], 10, digits=0, scale=ld.MIB)}  "
                f"{statuses}"
            )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="report", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=str(mf.default_manifest_path()),
        help="path to methods.yaml (default: benchmark/methods/methods.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render", help="write the result tables and figures into thesis/generated/"
    )
    render_parser.add_argument("campaign_dir", nargs="+")
    render_parser.add_argument(
        "--out", help=f"output directory (default: {THESIS_GENERATED}/)"
    )
    render_parser.add_argument(
        "--no-figures", action="store_true", help="tables only, no pgfplots sources"
    )

    summary_parser = subparsers.add_parser(
        "summary", help="print a campaign's aggregates as text"
    )
    summary_parser.add_argument("campaign_dir")

    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            return cmd_render(args)
        return cmd_summary(args)
    except ld.ReportError as exc:
        print(f"report: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
