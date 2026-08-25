from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dialogue_locator.acquisition import validate_public_url
from dialogue_locator.errors import V0Error
from dialogue_locator.transcription import DEFAULT_MODEL


MANIFEST_VERSION = 1
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ManifestError(ValueError):
    """Raised when a benchmark manifest is invalid."""


@dataclass(frozen=True)
class ProductionBaseline:
    dialogue_start_seconds: float
    timestamp_tolerance_seconds: float = 0.05
    matched_text: str | None = None


@dataclass(frozen=True)
class BenchmarkDefaults:
    work_dir: Path = Path(".cache/media")
    output_dir: Path = Path("experiments/audio_localization_baseline/results/frames")
    model_cache: Path = Path(".cache/models")
    model: str = DEFAULT_MODEL
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    fuzzy_threshold: float = 85.0
    reuse_transcript_cache: bool = False
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    url: str
    target: str
    model: str
    device: str
    compute_type: str
    language: str | None
    fuzzy_threshold: float
    production_baseline: ProductionBaseline | None = None


@dataclass(frozen=True)
class BenchmarkManifest:
    defaults: BenchmarkDefaults
    cases: tuple[BenchmarkCase, ...]


def load_manifest(path: Path) -> BenchmarkManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"Could not read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("Manifest root must be a JSON object.")
    if payload.get("version") != MANIFEST_VERSION:
        raise ManifestError(f"Manifest version must be {MANIFEST_VERSION}.")

    defaults_payload = payload.get("defaults", {})
    if not isinstance(defaults_payload, dict):
        raise ManifestError("'defaults' must be a JSON object.")
    defaults = _parse_defaults(defaults_payload)

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestError("'cases' must be a non-empty JSON array.")
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ManifestError(f"cases[{index}] must be a JSON object.")
        case = _parse_case(raw_case, defaults, index)
        if case.case_id in seen_ids:
            raise ManifestError(f"Duplicate case id '{case.case_id}'.")
        seen_ids.add(case.case_id)
        cases.append(case)
    return BenchmarkManifest(defaults=defaults, cases=tuple(cases))


def _parse_defaults(data: dict[str, Any]) -> BenchmarkDefaults:
    fuzzy_threshold = _finite_number(
        data.get("fuzzy_threshold", 85.0), "defaults.fuzzy_threshold", minimum=0, maximum=100
    )
    reuse_cache = data.get("reuse_transcript_cache", False)
    if not isinstance(reuse_cache, bool):
        raise ManifestError("defaults.reuse_transcript_cache must be a boolean.")
    return BenchmarkDefaults(
        work_dir=_path(data.get("work_dir", ".cache/media"), "defaults.work_dir"),
        output_dir=_path(
            data.get("output_dir", "experiments/audio_localization_baseline/results/frames"),
            "defaults.output_dir",
        ),
        model_cache=_path(data.get("model_cache", ".cache/models"), "defaults.model_cache"),
        model=_nonempty_string(data.get("model", DEFAULT_MODEL), "defaults.model"),
        device=_choice(data.get("device", "cpu"), "defaults.device", {"cpu", "cuda", "auto"}),
        compute_type=_nonempty_string(data.get("compute_type", "int8"), "defaults.compute_type"),
        language=_optional_string(data.get("language"), "defaults.language"),
        fuzzy_threshold=fuzzy_threshold,
        reuse_transcript_cache=reuse_cache,
        cookies_from_browser=_optional_string(
            data.get("cookies_from_browser"), "defaults.cookies_from_browser"
        ),
        cookies_file=(
            _path(data["cookies_file"], "defaults.cookies_file")
            if data.get("cookies_file") is not None
            else None
        ),
    )


def _parse_case(data: dict[str, Any], defaults: BenchmarkDefaults, index: int) -> BenchmarkCase:
    prefix = f"cases[{index}]"
    case_id = _nonempty_string(data.get("id"), f"{prefix}.id")
    if CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise ManifestError(
            f"{prefix}.id must start with a letter or number and contain only letters, "
            "numbers, '.', '_' or '-'."
        )
    url = _nonempty_string(data.get("url"), f"{prefix}.url")
    try:
        url = validate_public_url(url)
    except V0Error as exc:
        raise ManifestError(f"{prefix}.url: {exc}") from exc
    baseline_data = data.get("production_baseline")
    baseline = None
    if baseline_data is not None:
        if not isinstance(baseline_data, dict):
            raise ManifestError(f"{prefix}.production_baseline must be a JSON object.")
        baseline = ProductionBaseline(
            dialogue_start_seconds=_finite_number(
                baseline_data.get("dialogue_start_seconds"),
                f"{prefix}.production_baseline.dialogue_start_seconds",
                minimum=0,
            ),
            timestamp_tolerance_seconds=_finite_number(
                baseline_data.get("timestamp_tolerance_seconds", 0.05),
                f"{prefix}.production_baseline.timestamp_tolerance_seconds",
                minimum=0,
            ),
            matched_text=_optional_string(
                baseline_data.get("matched_text"),
                f"{prefix}.production_baseline.matched_text",
            ),
        )
    return BenchmarkCase(
        case_id=case_id,
        url=url,
        target=_nonempty_string(data.get("target"), f"{prefix}.target"),
        model=_nonempty_string(data.get("model", defaults.model), f"{prefix}.model"),
        device=_choice(
            data.get("device", defaults.device), f"{prefix}.device", {"cpu", "cuda", "auto"}
        ),
        compute_type=_nonempty_string(
            data.get("compute_type", defaults.compute_type), f"{prefix}.compute_type"
        ),
        language=_optional_string(data.get("language", defaults.language), f"{prefix}.language"),
        fuzzy_threshold=_finite_number(
            data.get("fuzzy_threshold", defaults.fuzzy_threshold),
            f"{prefix}.fuzzy_threshold",
            minimum=0,
            maximum=100,
        ),
        production_baseline=baseline,
    )


def _path(value: Any, name: str) -> Path:
    return Path(_nonempty_string(value, name))


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _choice(value: Any, name: str, choices: set[str]) -> str:
    selected = _nonempty_string(value, name)
    if selected not in choices:
        raise ManifestError(f"{name} must be one of: {', '.join(sorted(choices))}.")
    return selected


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ManifestError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ManifestError(f"{name} must be at least {minimum:g}.")
    if maximum is not None and result > maximum:
        raise ManifestError(f"{name} must be at most {maximum:g}.")
    return result
