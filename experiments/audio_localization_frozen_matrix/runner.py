from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dialogue_locator.acquisition import acquire_media

from experiments.audio_localization_baseline.runner import run_benchmark
from experiments.audio_localization_chunked.runner import (
    load_baseline_results,
    run_chunked_benchmark,
)

from .evaluation import build_comparison, write_outputs
from .manifest import FrozenMatrixManifest, to_strategy_manifest


def prepare_public_media(
    manifest: FrozenMatrixManifest,
    output_path: Path,
) -> dict[str, Any]:
    urls = sorted({case.url for case in manifest.cases if case.source_kind == "public"})
    records = []
    for url in urls:
        try:
            path, metadata = acquire_media(url, manifest.defaults.work_dir)
            records.append(
                {
                    "url": url,
                    "status": "ok",
                    "media_path": str(path),
                    "media_cache_hit": metadata.get("media_cache_hit", False),
                    "error_reason": None,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "url": url,
                    "status": "error",
                    "media_path": None,
                    "media_cache_hit": False,
                    "error_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unique_public_media": len(urls),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_frozen_matrix(
    manifest: FrozenMatrixManifest,
    manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    strategy_manifest = to_strategy_manifest(manifest)
    missing = [
        str(case.local_media_path)
        for case in strategy_manifest.cases
        if case.local_media_path is not None and not case.local_media_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Controlled fixtures are missing: {', '.join(missing)}")
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = run_dir / "baseline.json"
    chunked_path = run_dir / "chunked.json"
    baseline = run_benchmark(
        strategy_manifest,
        manifest_path=manifest_path,
        output_path=baseline_path,
    )
    baseline_results = load_baseline_results(baseline_path)
    chunked = run_chunked_benchmark(
        strategy_manifest,
        manifest.chunked_asr,
        baseline_results,
        manifest_path=manifest_path,
        baseline_results_path=baseline_path,
        output_path=chunked_path,
    )
    summary = build_comparison(manifest, baseline, chunked)
    summary["run_policy"] = {
        "warmup_runs": 1,
        "measured_runs_per_case_strategy": 1,
        "reason": "Long CPU-only public videos make three complete measured repetitions unreasonable.",
    }
    write_outputs(summary, run_dir)
    return summary


def run_baseline_only(
    manifest: FrozenMatrixManifest,
    manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_benchmark(
        to_strategy_manifest(manifest),
        manifest_path=manifest_path,
        output_path=run_dir / "baseline.json",
    )


def run_chunked_only(
    manifest: FrozenMatrixManifest,
    manifest_path: Path,
    baseline_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_chunked_benchmark(
        to_strategy_manifest(manifest),
        manifest.chunked_asr,
        load_baseline_results(baseline_path),
        manifest_path=manifest_path,
        baseline_results_path=baseline_path,
        output_path=run_dir / "chunked.json",
    )
