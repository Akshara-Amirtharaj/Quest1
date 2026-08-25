from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.audio_localization_baseline.manifest import ManifestError

from .manifest import load_vad_manifest
from .runner import load_strategy_results, run_vad_benchmark


DEFAULT_OUTPUT = Path("experiments/audio_localization_chunked_vad/results/chunked-vad-asr.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark conservative Silero VAD plus chronological chunked ASR."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--chunked-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_vad_manifest(args.manifest)
        baseline = load_strategy_results(args.baseline_results, "full-ASR baseline")
        chunked = load_strategy_results(args.chunked_results, "chunked-ASR")
        report = run_vad_benchmark(
            loaded.chunked.baseline,
            loaded.chunked.chunked_asr,
            loaded.vad,
            baseline,
            chunked,
            manifest_path=args.manifest,
            baseline_results_path=args.baseline_results,
            chunked_results_path=args.chunked_results,
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
