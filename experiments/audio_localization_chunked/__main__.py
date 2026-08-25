from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.audio_localization_baseline.manifest import ManifestError

from .manifest import load_chunked_manifest, override_chunk_config
from .runner import load_baseline_results, run_chunked_benchmark


DEFAULT_OUTPUT = Path("experiments/audio_localization_chunked/results/chunked-asr.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark chronological overlapping chunked ASR with early stopping."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-duration",
        type=float,
        help="Override manifest chunk duration in seconds",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        help="Override manifest overlap in seconds",
    )
    parser.add_argument(
        "--transcript-context",
        type=float,
        help="Override carried adjacent-transcript context in seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_chunked_manifest(args.manifest)
        config = override_chunk_config(
            loaded.chunked_asr,
            chunk_duration_seconds=args.chunk_duration,
            overlap_seconds=args.overlap,
            transcript_context_seconds=args.transcript_context,
        )
        baseline_results = load_baseline_results(args.baseline_results)
        report = run_chunked_benchmark(
            loaded.baseline,
            config,
            baseline_results,
            manifest_path=args.manifest,
            baseline_results_path=args.baseline_results,
            output_path=args.output,
        )
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
