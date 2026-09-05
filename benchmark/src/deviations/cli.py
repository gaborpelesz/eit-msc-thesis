"""`deviations` -- manifest and fork-deviation tooling.

    deviations verify [--manifest PATH] [--arch 75|120] [--binaries DIR]
    deviations list   [--manifest PATH]
    deviations render [--manifest PATH]
"""

import argparse
import sys

from . import manifest as mf
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
        help="CUDA architecture the binaries must carry; requires --binaries",
    )
    verify_parser.add_argument(
        "--binaries",
        help="directory holding the built method binaries (the image's /sota)",
    )

    list_parser = subparsers.add_parser(
        "list", help="print the deviation table read from the forks' git logs"
    )
    add_manifest_argument(list_parser)

    render_parser = subparsers.add_parser(
        "render", help="write DEVIATIONS.md, deviations.json and deviations.tex"
    )
    add_manifest_argument(render_parser)

    args = parser.parse_args()

    if args.command == "verify":
        if (args.arch is None) != (args.binaries is None):
            parser.error("--arch and --binaries must be given together")
        return vf.verify(args.manifest, arch=args.arch, binaries=args.binaries)

    if args.command == "list":
        return vf.list_deviations(args.manifest)

    print("deviations render: not implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
