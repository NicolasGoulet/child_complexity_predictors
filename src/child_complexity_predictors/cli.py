"""Command-line interface for complexity predictor extraction."""

from __future__ import annotations

import argparse
import json
import sys

from .manifest import ComplexityManifest
from .predictors import run_extract
from .trajectory import run_trajectory


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    manifest = ComplexityManifest.from_path(args.manifest)
    if args.check_inputs:
        manifest.validate_existing_inputs()
    print(json.dumps({"status": "ok", "run_id": manifest.run_id}, indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    manifest = ComplexityManifest.from_path(args.manifest)
    audit = run_extract(manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "audit_json": str(manifest.audit_json),
                "row_count": audit["row_count"],
            },
            indent=2,
        )
    )
    return 0


def cmd_trajectory(args: argparse.Namespace) -> int:
    manifest = ComplexityManifest.from_path(args.manifest)
    audit = run_trajectory(manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "trajectory_rows": audit["trajectory_row_count"],
                "aggregate_rows": audit["aggregate_row_count"],
                "trajectory_output_csv": audit["trajectory_output_csv"],
                "aggregate_output_csv": audit["aggregate_output_csv"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="child-complexity-predictors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--check-inputs", action="store_true")
    validate.set_defaults(func=cmd_validate_manifest)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--manifest", required=True)
    extract.set_defaults(func=cmd_extract)

    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--manifest", required=True)
    trajectory.set_defaults(func=cmd_trajectory)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
