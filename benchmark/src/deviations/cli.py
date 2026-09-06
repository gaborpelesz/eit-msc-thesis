"""`deviations` -- manifest and fork-deviation tooling.

    deviations verify [--manifest PATH] [--arch 75|120] [--binaries DIR | --image NAME]
    deviations list   [--manifest PATH]
    deviations render [--manifest PATH] [--methods-dir DIR] [--thesis-dir DIR]

`render` writes the four generated artefacts -- `benchmark/methods/DEVIATIONS.md`,
`benchmark/methods/deviations.json`, `thesis/generated/deviations.tex` and
`thesis/generated/methods-provenance.tex` -- from the manifest and the forks'
`upstream-base..HEAD` logs. They are never hand-edited; re-run `render` instead.
"""

import argparse
import sys

from . import manifest as mf
from . import render as rd
from . import verify as vf


def add_manifest_argument(parser):
    parser.add_argument(
        "--manifest",
        default=str(mf.default_manifest_path()),
        help="path to methods.yaml (default: benchmark/methods/methods.yaml)",
    )


def main():
    parser = argparse.ArgumentParser(prog="deviations", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="check the manifest against the forks; gates every benchmark run"
    )
    add_manifest_argument(verify_parser)
    verify_parser.add_argument(
        "--arch",
        type=int,
        choices=[75, 120],
        help="CUDA architecture the method kernels must carry; requires --binaries or --image",
    )
    where = verify_parser.add_mutually_exclusive_group()
    where.add_argument(
        "--binaries",
        help="host directory holding the built method binaries (needs cuobjdump on the host)",
    )
    where.add_argument(
        "--image",
        help="Docker image whose /sota holds the built binaries; cuobjdump runs inside it",
    )

    list_parser = subparsers.add_parser(
        "list", help="print the deviation table read from the forks' git logs"
    )
    add_manifest_argument(list_parser)

    render_parser = subparsers.add_parser(
        "render",
        help="write DEVIATIONS.md, deviations.json, deviations.tex and "
        "methods-provenance.tex from the manifest and the forks' git logs",
    )
    add_manifest_argument(render_parser)
    render_parser.add_argument(
        "--methods-dir",
        help="where DEVIATIONS.md and deviations.json go (default: next to the manifest)",
    )
    render_parser.add_argument(
        "--thesis-dir",
        help=f"where the .tex tables go (default: {rd.THESIS_GENERATED}/)",
    )

    args = parser.parse_args()

    if args.command == "verify":
        if (args.arch is None) != (args.binaries is None and args.image is None):
            parser.error("--arch must be given together with --binaries or --image")
        return vf.verify(args.manifest, arch=args.arch, binaries=args.binaries, image=args.image)

    if args.command == "list":
        return vf.list_deviations(args.manifest)

    return rd.render(
        args.manifest, methods_dir=args.methods_dir, thesis_dir=args.thesis_dir
    )


if __name__ == "__main__":
    sys.exit(main())
