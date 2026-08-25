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


DEFAULT_MODEL_DIR = Path(
    ".cache/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
)


@dataclass(frozen=True)
class KWSConfig:
    model_dir: Path = DEFAULT_MODEL_DIR
    max_anchors: int = 3
    grouping_gap_seconds: float = 2.0
    candidate_margin_before_seconds: float = 2.5
    candidate_margin_after_seconds: float = 2.0
    num_threads: int = 1
    keywords_score: float = 2.0
    keywords_threshold: float = 0.15
    num_trailing_blanks: int = 1

    def validate(self) -> KWSConfig:
        if self.max_anchors <= 0:
            raise ManifestError("kws.max_anchors must be greater than zero.")
        if self.num_threads <= 0:
            raise ManifestError("kws.num_threads must be greater than zero.")
        if self.num_trailing_blanks < 0:
            raise ManifestError("kws.num_trailing_blanks cannot be negative.")
        for value, name in (
            (self.grouping_gap_seconds, "grouping_gap_seconds"),
            (self.candidate_margin_before_seconds, "candidate_margin_before_seconds"),
            (self.candidate_margin_after_seconds, "candidate_margin_after_seconds"),
            (self.keywords_score, "keywords_score"),
            (self.keywords_threshold, "keywords_threshold"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ManifestError(f"kws.{name} cannot be negative.")
        if self.keywords_threshold > 1:
            raise ManifestError("kws.keywords_threshold must be at most 1.")
        return self


@dataclass(frozen=True)
class KWSBenchmarkManifest:
    baseline: BenchmarkManifest
    kws: KWSConfig


def load_kws_manifest(path: Path) -> KWSBenchmarkManifest:
    baseline = load_manifest(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read KWS configuration: {exc}") from exc
    raw = payload.get("kws", {})
    if not isinstance(raw, dict):
        raise ManifestError("'kws' must be a JSON object.")
    config = KWSConfig(
        model_dir=Path(_string(raw.get("model_dir", str(DEFAULT_MODEL_DIR)), "kws.model_dir")),
        max_anchors=_integer(raw.get("max_anchors", 3), "kws.max_anchors"),
        grouping_gap_seconds=_number(raw.get("grouping_gap_seconds", 2), "kws.grouping_gap_seconds"),
        candidate_margin_before_seconds=_number(
            raw.get("candidate_margin_before_seconds", 2.5),
            "kws.candidate_margin_before_seconds",
        ),
        candidate_margin_after_seconds=_number(
            raw.get("candidate_margin_after_seconds", 2),
            "kws.candidate_margin_after_seconds",
        ),
        num_threads=_integer(raw.get("num_threads", 1), "kws.num_threads"),
        keywords_score=_number(raw.get("keywords_score", 2), "kws.keywords_score"),
        keywords_threshold=_number(
            raw.get("keywords_threshold", 0.15), "kws.keywords_threshold"
        ),
        num_trailing_blanks=_integer(
            raw.get("num_trailing_blanks", 1), "kws.num_trailing_blanks"
        ),
    ).validate()
    return KWSBenchmarkManifest(baseline, config)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string.")
    return value.strip()


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


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{name} must be an integer.")
    return value
