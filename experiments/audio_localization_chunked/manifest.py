from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.audio_localization_baseline.manifest import (
    BenchmarkManifest,
    ManifestError,
    load_manifest,
)


@dataclass(frozen=True)
class ChunkedASRConfig:
    chunk_duration_seconds: float = 10.0
    overlap_seconds: float = 2.0

    def validate(self) -> ChunkedASRConfig:
        if not math.isfinite(self.chunk_duration_seconds) or self.chunk_duration_seconds <= 0:
            raise ManifestError("chunked_asr.chunk_duration_seconds must be greater than zero.")
        if not math.isfinite(self.overlap_seconds) or self.overlap_seconds < 0:
            raise ManifestError("chunked_asr.overlap_seconds cannot be negative.")
        if self.overlap_seconds >= self.chunk_duration_seconds:
            raise ManifestError(
                "chunked_asr.overlap_seconds must be smaller than chunk_duration_seconds."
            )
        return self


@dataclass(frozen=True)
class ChunkedBenchmarkManifest:
    baseline: BenchmarkManifest
    chunked_asr: ChunkedASRConfig


def load_chunked_manifest(path: Path) -> ChunkedBenchmarkManifest:
    baseline = load_manifest(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read chunked-ASR configuration: {exc}") from exc
    raw_config = payload.get("chunked_asr", {})
    if not isinstance(raw_config, dict):
        raise ManifestError("'chunked_asr' must be a JSON object.")
    config = ChunkedASRConfig(
        chunk_duration_seconds=_number(
            raw_config.get("chunk_duration_seconds", 10.0),
            "chunked_asr.chunk_duration_seconds",
        ),
        overlap_seconds=_number(
            raw_config.get("overlap_seconds", 2.0),
            "chunked_asr.overlap_seconds",
        ),
    ).validate()
    return ChunkedBenchmarkManifest(baseline=baseline, chunked_asr=config)


def override_chunk_config(
    config: ChunkedASRConfig,
    *,
    chunk_duration_seconds: float | None,
    overlap_seconds: float | None,
) -> ChunkedASRConfig:
    return ChunkedASRConfig(
        chunk_duration_seconds=(
            config.chunk_duration_seconds
            if chunk_duration_seconds is None
            else chunk_duration_seconds
        ),
        overlap_seconds=config.overlap_seconds if overlap_seconds is None else overlap_seconds,
    ).validate()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ManifestError(f"{name} must be a finite number.")
    return result
