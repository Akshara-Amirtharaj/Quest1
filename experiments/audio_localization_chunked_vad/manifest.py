from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_chunked.manifest import (
    ChunkedBenchmarkManifest,
    load_chunked_manifest,
)


@dataclass(frozen=True)
class ConservativeVADConfig:
    threshold: float = 0.35
    neg_threshold: float = 0.20
    min_speech_duration_ms: int = 0
    min_silence_duration_ms: int = 1000
    speech_pad_ms: int = 500
    clear_silence_rms_threshold: float = 0.002
    max_removed_fraction_before_fallback: float = 0.98
    fallback_on_no_match: bool = True
    fallback_on_baseline_mismatch: bool = True

    def validate(self) -> ConservativeVADConfig:
        if not 0 < self.threshold < 1:
            raise ManifestError("vad.threshold must be between 0 and 1.")
        if not 0 < self.neg_threshold < self.threshold:
            raise ManifestError("vad.neg_threshold must be positive and below threshold.")
        if self.min_speech_duration_ms < 0:
            raise ManifestError("vad.min_speech_duration_ms cannot be negative.")
        if self.min_silence_duration_ms < 0:
            raise ManifestError("vad.min_silence_duration_ms cannot be negative.")
        if self.speech_pad_ms < 0:
            raise ManifestError("vad.speech_pad_ms cannot be negative.")
        if not math.isfinite(self.clear_silence_rms_threshold) or self.clear_silence_rms_threshold < 0:
            raise ManifestError("vad.clear_silence_rms_threshold must be finite and non-negative.")
        if not 0 <= self.max_removed_fraction_before_fallback <= 1:
            raise ManifestError(
                "vad.max_removed_fraction_before_fallback must be between 0 and 1."
            )
        return self


@dataclass(frozen=True)
class VADBenchmarkManifest:
    chunked: ChunkedBenchmarkManifest
    vad: ConservativeVADConfig


def load_vad_manifest(path: Path) -> VADBenchmarkManifest:
    chunked = load_chunked_manifest(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read VAD configuration: {exc}") from exc
    raw = payload.get("vad", {})
    if not isinstance(raw, dict):
        raise ManifestError("'vad' must be a JSON object.")
    config = ConservativeVADConfig(
        threshold=_float(raw.get("threshold", 0.35), "vad.threshold"),
        neg_threshold=_float(raw.get("neg_threshold", 0.20), "vad.neg_threshold"),
        min_speech_duration_ms=_int(
            raw.get("min_speech_duration_ms", 0), "vad.min_speech_duration_ms"
        ),
        min_silence_duration_ms=_int(
            raw.get("min_silence_duration_ms", 1000), "vad.min_silence_duration_ms"
        ),
        speech_pad_ms=_int(raw.get("speech_pad_ms", 500), "vad.speech_pad_ms"),
        clear_silence_rms_threshold=_float(
            raw.get("clear_silence_rms_threshold", 0.002),
            "vad.clear_silence_rms_threshold",
        ),
        max_removed_fraction_before_fallback=_float(
            raw.get("max_removed_fraction_before_fallback", 0.98),
            "vad.max_removed_fraction_before_fallback",
        ),
        fallback_on_no_match=_bool(
            raw.get("fallback_on_no_match", True), "vad.fallback_on_no_match"
        ),
        fallback_on_baseline_mismatch=_bool(
            raw.get("fallback_on_baseline_mismatch", True),
            "vad.fallback_on_baseline_mismatch",
        ),
    ).validate()
    return VADBenchmarkManifest(chunked=chunked, vad=config)


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ManifestError(f"{name} must be a finite number.")
    return result


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be an integer.") from exc
    if float(value) != result:
        raise ManifestError(f"{name} must be an integer.")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{name} must be a boolean.")
    return value
