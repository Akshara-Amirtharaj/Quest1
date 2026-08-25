from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_chunked.runner import load_baseline_results

from .manifest import load_kws_manifest
from .runner import run_kws_benchmark


DEFAULT_OUTPUT = Path("experiments/audio_localization_kws/results/kws-accurate-asr.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark sherpa-onnx open-vocabulary KWS plus accurate ASR."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--chunked-results", type=Path, required=True)
    parser.add_argument("--vad-results", type=Path, required=True)
    parser.add_argument("--locator-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_kws_manifest(args.manifest)
        report = run_kws_benchmark(
            loaded.baseline,
            loaded.kws,
            load_baseline_results(args.baseline_results),
            load_baseline_results(args.chunked_results),
            load_baseline_results(args.vad_results),
            load_baseline_results(args.locator_results),
            manifest_path=args.manifest,
            baseline_results_path=args.baseline_results,
            chunked_results_path=args.chunked_results,
            vad_results_path=args.vad_results,
            locator_results_path=args.locator_results,
            output_path=args.output,
        )
    except ManifestError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
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
