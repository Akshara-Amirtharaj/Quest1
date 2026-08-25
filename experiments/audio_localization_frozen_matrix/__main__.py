from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.audio_localization_baseline.manifest import ManifestError

from .fixtures import generate_controlled_fixtures
from .manifest import filter_manifest, load_frozen_manifest
from .runner import (
    prepare_public_media,
    run_baseline_only,
    run_chunked_only,
    run_frozen_matrix,
)


DEFAULT_MANIFEST = Path("experiments/audio_localization_frozen_matrix/manifest.json")
DEFAULT_RESULTS = Path("experiments/audio_localization_frozen_matrix/results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen full-ASR versus 120/5/15 chunked matrix")
    parser.add_argument("action", choices=("validate", "generate-fixtures", "prepare", "run"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--source-kind", choices=("public", "controlled"))
    parser.add_argument("--strategy", choices=("both", "baseline", "chunked"), default="both")
    parser.add_argument("--baseline-results", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_frozen_manifest(args.manifest)
        if args.case_id or args.source_kind:
            manifest = filter_manifest(
                manifest,
                case_ids=tuple(args.case_id),
                source_kind=args.source_kind,
            )
        if args.action == "validate":
            result = {
                "benchmark_id": manifest.benchmark_id,
                "total_cases": len(manifest.cases),
                "audio_cases": len(manifest.audio_cases),
                "ocr_only_cases": len(manifest.cases) - len(manifest.audio_cases),
                "unique_media_sources": manifest.unique_media_count,
                "chunked_asr": manifest.chunked_asr.__dict__,
            }
        elif args.action == "generate-fixtures":
            fixtures = generate_controlled_fixtures()
            result = {"fixtures": {key: value.__dict__ for key, value in fixtures.items()}}
        elif args.action == "prepare":
            result = prepare_public_media(manifest, DEFAULT_RESULTS / "media-preparation.json")
        else:
            run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = DEFAULT_RESULTS / run_id
            if args.strategy == "baseline":
                result = run_baseline_only(manifest, args.manifest, run_dir)
            elif args.strategy == "chunked":
                if args.baseline_results is None:
                    raise ManifestError("--strategy chunked requires --baseline-results.")
                result = run_chunked_only(
                    manifest,
                    args.manifest,
                    args.baseline_results,
                    run_dir,
                )
            else:
                result = run_frozen_matrix(manifest, args.manifest, run_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ManifestError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
