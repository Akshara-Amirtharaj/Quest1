from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.audio_localization_baseline.manifest import (
    BenchmarkCase,
    BenchmarkDefaults,
    BenchmarkManifest,
    ManifestError,
    ProductionBaseline,
)
from experiments.audio_localization_chunked.manifest import ChunkedASRConfig


FROZEN_CHUNK_DURATION_SECONDS = 120.0
FROZEN_OVERLAP_SECONDS = 5.0
FROZEN_TRANSCRIPT_CONTEXT_SECONDS = 15.0
FROZEN_MATCHING_LOGIC = "dialogue_locator.matching.find_dialogue_candidates"
AUDIO_STRATEGIES = ("full_asr", "chunked_asr")
TIMESTAMP_PATTERN = re.compile(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$")


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    source_kind: str
    url: str
    local_media_path: Path | None
    target: str
    expected_result: str
    expected_timestamp_seconds: float | None
    tolerance_seconds: float | None
    modality: str
    ground_truth_type: str
    tags: tuple[str, ...]
    applicable_strategies: tuple[str, ...]

    @property
    def is_audio_case(self) -> bool:
        return all(strategy in self.applicable_strategies for strategy in AUDIO_STRATEGIES)

    @property
    def media_identity(self) -> str:
        return str(self.local_media_path) if self.local_media_path is not None else self.url


@dataclass(frozen=True)
class FrozenMatrixManifest:
    benchmark_id: str
    defaults: BenchmarkDefaults
    chunked_asr: ChunkedASRConfig
    cases: tuple[MatrixCase, ...]

    @property
    def audio_cases(self) -> tuple[MatrixCase, ...]:
        return tuple(case for case in self.cases if case.is_audio_case)

    @property
    def unique_media_count(self) -> int:
        return len({case.media_identity for case in self.cases})


def parse_timestamp(value: str) -> float:
    match = TIMESTAMP_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ManifestError(f"Timestamp must use HH:MM:SS.mmm: {value!r}")
    hours, minutes, seconds, milliseconds = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ManifestError(f"Timestamp minutes and seconds must be below 60: {value!r}")
    fraction = int((milliseconds or "0").ljust(3, "0")) / 1000.0
    return int(hours) * 3600.0 + int(minutes) * 60.0 + int(seconds) + fraction


def load_frozen_manifest(path: Path) -> FrozenMatrixManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read frozen benchmark manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ManifestError("Frozen benchmark manifest version must be 1.")
    config = payload.get("frozen_chunked_asr")
    if not isinstance(config, dict):
        raise ManifestError("frozen_chunked_asr must be an object.")
    chunked = ChunkedASRConfig(
        chunk_duration_seconds=_number(config.get("chunk_duration_seconds"), "chunk duration"),
        overlap_seconds=_number(config.get("overlap_seconds"), "overlap"),
        transcript_context_seconds=_number(
            config.get("transcript_context_seconds"), "transcript context"
        ),
    ).validate()
    expected_config = (
        FROZEN_CHUNK_DURATION_SECONDS,
        FROZEN_OVERLAP_SECONDS,
        FROZEN_TRANSCRIPT_CONTEXT_SECONDS,
    )
    actual_config = (
        chunked.chunk_duration_seconds,
        chunked.overlap_seconds,
        chunked.transcript_context_seconds,
    )
    if actual_config != expected_config:
        raise ManifestError(f"Frozen chunk configuration must be {expected_config}, got {actual_config}.")

    raw_defaults = payload.get("defaults", {})
    if not isinstance(raw_defaults, dict):
        raise ManifestError("defaults must be an object.")
    defaults = BenchmarkDefaults(
        work_dir=Path(raw_defaults.get("work_dir", ".cache/media")),
        output_dir=Path(
            raw_defaults.get(
                "output_dir", "experiments/audio_localization_frozen_matrix/results/frames"
            )
        ),
        model_cache=Path(raw_defaults.get("model_cache", ".cache/models")),
        model=str(raw_defaults.get("model", "base.en")),
        device=str(raw_defaults.get("device", "cpu")),
        compute_type=str(raw_defaults.get("compute_type", "int8")),
        language=raw_defaults.get("language", "en"),
        fuzzy_threshold=_number(raw_defaults.get("fuzzy_threshold", 85), "fuzzy threshold"),
        reuse_transcript_cache=False,
    )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 12:
        raise ManifestError("Frozen benchmark manifest must contain exactly 12 cases.")
    cases = tuple(_parse_case(raw, index) for index, raw in enumerate(raw_cases))
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ManifestError("Frozen benchmark case IDs must be unique.")
    return FrozenMatrixManifest(
        benchmark_id=str(payload.get("benchmark_id", "frozen-audio-localization-v1")),
        defaults=defaults,
        chunked_asr=chunked,
        cases=cases,
    )


def to_strategy_manifest(manifest: FrozenMatrixManifest) -> BenchmarkManifest:
    cases = tuple(
        BenchmarkCase(
            case_id=case.case_id,
            url=case.url,
            source_page_url=None,
            target=case.target,
            model=manifest.defaults.model,
            device=manifest.defaults.device,
            compute_type=manifest.defaults.compute_type,
            language=manifest.defaults.language,
            fuzzy_threshold=manifest.defaults.fuzzy_threshold,
            production_baseline=(
                ProductionBaseline(
                    dialogue_start_seconds=case.expected_timestamp_seconds,
                    timestamp_tolerance_seconds=case.tolerance_seconds or 0.0,
                )
                if case.expected_timestamp_seconds is not None
                else None
            ),
            local_media_path=case.local_media_path,
        )
        for case in manifest.audio_cases
    )
    return BenchmarkManifest(defaults=manifest.defaults, cases=cases)


def filter_manifest(
    manifest: FrozenMatrixManifest,
    *,
    case_ids: tuple[str, ...] = (),
    source_kind: str | None = None,
) -> FrozenMatrixManifest:
    selected = tuple(
        case
        for case in manifest.cases
        if (not case_ids or case.case_id in case_ids)
        and (source_kind is None or case.source_kind == source_kind)
    )
    if case_ids:
        missing = sorted(set(case_ids) - {case.case_id for case in selected})
        if missing:
            raise ManifestError(f"Unknown or filtered case IDs: {', '.join(missing)}")
    if not selected:
        raise ManifestError("Benchmark filter selected no cases.")
    return replace(manifest, cases=selected)


def _parse_case(raw: Any, index: int) -> MatrixCase:
    if not isinstance(raw, dict):
        raise ManifestError(f"cases[{index}] must be an object.")
    expected_result = str(raw.get("expected_result", "MATCH"))
    timestamp_value = raw.get("expected_timestamp")
    timestamp = parse_timestamp(timestamp_value) if isinstance(timestamp_value, str) else None
    if expected_result == "MATCH" and timestamp is None:
        raise ManifestError(f"cases[{index}] MATCH requires expected_timestamp.")
    if expected_result == "NO_MATCH" and timestamp is not None:
        raise ManifestError(f"cases[{index}] NO_MATCH must not have expected_timestamp.")
    strategies = tuple(str(item) for item in raw.get("applicable_strategies", AUDIO_STRATEGIES))
    local_path = raw.get("local_media_path")
    return MatrixCase(
        case_id=str(raw["id"]),
        source_kind=str(raw["source_kind"]),
        url=str(raw["url"]),
        local_media_path=Path(local_path) if local_path is not None else None,
        target=str(raw["target"]),
        expected_result=expected_result,
        expected_timestamp_seconds=timestamp,
        tolerance_seconds=(
            _number(raw.get("tolerance_seconds"), f"cases[{index}].tolerance_seconds")
            if raw.get("tolerance_seconds") is not None
            else None
        ),
        modality=str(raw["modality"]),
        ground_truth_type=str(raw["ground_truth_type"]),
        tags=tuple(str(item) for item in raw.get("tags", [])),
        applicable_strategies=strategies,
    )


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ManifestError(f"{name} must be a finite number.")
    return number
