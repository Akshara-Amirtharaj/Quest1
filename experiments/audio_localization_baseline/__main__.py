from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import ManifestError, load_manifest
from .runner import run_benchmark


DEFAULT_OUTPUT = Path("experiments/audio_localization_baseline/results/full-asr-baseline.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the unchanged Quest1 production run_v1 full-ASR baseline."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = run_benchmark(manifest, manifest_path=args.manifest, output_path=args.output)
    except ManifestError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"error": f"Could not write benchmark output: {exc}"}, indent=2), file=sys.stderr)
        return 2
    errors = sum(case["status"] == "error" for case in report["cases"])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "cases": len(report["cases"]),
                "successful": len(report["cases"]) - errors,
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
