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
class LocatorConfig:
    model: str = "tiny.en"
    candidate_margin_before_seconds: float = 1.5
    candidate_margin_after_seconds: float = 2.0

    def validate(self) -> LocatorConfig:
        if not self.model.strip():
            raise ManifestError("lightweight_locator.model must be a non-empty string.")
        for value, name in (
            (self.candidate_margin_before_seconds, "candidate_margin_before_seconds"),
            (self.candidate_margin_after_seconds, "candidate_margin_after_seconds"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ManifestError(f"lightweight_locator.{name} cannot be negative.")
        return self


@dataclass(frozen=True)
class LocatorBenchmarkManifest:
    baseline: BenchmarkManifest
    locator: LocatorConfig


def load_locator_manifest(path: Path) -> LocatorBenchmarkManifest:
    baseline = load_manifest(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read lightweight-locator configuration: {exc}") from exc
    raw = payload.get("lightweight_locator", {})
    if not isinstance(raw, dict):
        raise ManifestError("'lightweight_locator' must be a JSON object.")
    model = raw.get("model", "tiny.en")
    if not isinstance(model, str):
        raise ManifestError("lightweight_locator.model must be a non-empty string.")
    config = LocatorConfig(
        model=model.strip(),
        candidate_margin_before_seconds=_number(
            raw.get("candidate_margin_before_seconds", 1.5),
            "lightweight_locator.candidate_margin_before_seconds",
        ),
        candidate_margin_after_seconds=_number(
            raw.get("candidate_margin_after_seconds", 2.0),
            "lightweight_locator.candidate_margin_after_seconds",
        ),
    ).validate()
    return LocatorBenchmarkManifest(baseline=baseline, locator=config)


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
