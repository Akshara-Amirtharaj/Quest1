from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dialogue_locator import __version__
from dialogue_locator import pipeline
from dialogue_locator.acquisition import acquire_media, validate_public_url
from dialogue_locator.audio import extract_speech_audio
from dialogue_locator.dependencies import require_external_tools
from dialogue_locator.errors import V0Error
from dialogue_locator.frames import resolve_frame_at_timestamp
from dialogue_locator.models import (
    DialogueMatch,
    MediaInfo,
    ResolvedFrame,
    Transcription,
    format_timestamp,
)
from dialogue_locator.transcription import FasterWhisperTranscriber, resolve_model_name

from experiments.audio_localization_baseline.manifest import (
    BenchmarkCase,
    BenchmarkManifest,
    ManifestError,
    ProductionBaseline,
)
from experiments.audio_localization_baseline.metrics import (
    first_occurrence_matches_baseline,
)
from experiments.audio_localization_baseline.runner import PeakRSSSampler, _write_json_atomic

from .chunking import AudioChunk, ChunkSearchResult, generate_chunks, search_chunks
from .manifest import ChunkedASRConfig
from .metrics import percentage_asr_audio_avoided, speedup_ratio, timestamp_delta
from .reusable_audio import ReusableChunkTranscriber, ReusableWaveAudio


Clock = Callable[[], float]
MATCHING_LOGIC_ID = "dialogue_locator.matching.find_dialogue_candidates"


@dataclass
class ChunkedObservation:
    asr_wall_clock_seconds: list[float] = field(default_factory=list)
    asr_audio_seconds: list[float] = field(default_factory=list)
    acquisition_metadata: dict[str, Any] = field(default_factory=dict)
    media_info: MediaInfo | None = None
    media_path: Path | None = None
    media_metadata_cache_hit: bool = False
    audio_extraction_calls: int = 0
    audio_extraction_wall_clock_seconds: float = 0.0


@dataclass(frozen=True)
class ChunkedLocalization:
    match: DialogueMatch
    frame: ResolvedFrame
    search: ChunkSearchResult
    total_chunks: int


def load_baseline_results(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"Could not read baseline results {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Baseline results are not valid JSON: {exc}") from exc
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ManifestError("Baseline results must contain a 'cases' array.")
    indexed: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ManifestError(f"Baseline results cases[{index}] has no valid case_id.")
        enriched = dict(case)
        enriched["_benchmark_environment"] = payload.get("environment")
        enriched["_matching_logic"] = payload.get("matching_logic")
        indexed[case["case_id"]] = enriched
    return indexed


def run_chunked_benchmark(
    manifest: BenchmarkManifest,
    config: ChunkedASRConfig,
    baseline_results: dict[str, dict[str, Any]],
    *,
    manifest_path: Path,
    baseline_results_path: Path,
    output_path: Path,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    records = [
        _run_case(case, manifest, config, baseline_results.get(case.case_id), clock=clock)
        for case in manifest.cases
    ]
    report = {
        "schema_version": 1,
        "benchmark": "chronological-overlapping-chunked-asr-early-stop",
        "matching_logic": MATCHING_LOGIC_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "baseline_results_path": str(baseline_results_path.resolve()),
        "chunked_asr": asdict(config),
        "environment": _current_environment(),
        "cases": records,
    }
    _write_json_atomic(output_path, report)
    return report


def _run_case(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    config: ChunkedASRConfig,
    baseline: dict[str, Any] | None,
    *,
    clock: Clock,
) -> dict[str, Any]:
    observation = ChunkedObservation()
    localization: ChunkedLocalization | None = None
    search_result: ChunkSearchResult | None = None
    total_chunks = 0
    error_reason: str | None = None
    started = clock()
    with PeakRSSSampler() as memory:
        try:
            localization, observation = _localize_chunked(
                case,
                manifest,
                config,
                observation,
                clock=clock,
            )
            search_result = localization.search
            total_chunks = localization.total_chunks
        except _NoChunkedMatch as exc:
            search_result = exc.search
            total_chunks = exc.total_chunks
            error_reason = (
                "V0Error: Dialogue not found in the chronological chunked-audio transcription "
                f"(fuzzy threshold {case.fuzzy_threshold:g})."
            )
        except Exception as exc:
            error_reason = f"{type(exc).__name__}: {exc}"
    total_wall_clock = max(0.0, clock() - started)

    match = localization.match if localization is not None else None
    baseline_start = _optional_number(baseline, "detected_timestamp_seconds")
    baseline_text = _optional_string(baseline, "matched_text")
    baseline_audio = _optional_number(baseline, "expensive_asr_audio_seconds_processed")
    baseline_total_wall = _optional_number(baseline, "total_wall_clock_seconds")
    baseline_asr_wall = _optional_number(baseline, "asr_wall_clock_seconds")
    processed_audio = sum(observation.asr_audio_seconds)
    asr_wall_clock = sum(observation.asr_wall_clock_seconds)
    comparison = _comparison_baseline(case, baseline_start, baseline_text)
    same_first = first_occurrence_matches_baseline(
        match.start if match is not None else None,
        match.matched_text if match is not None else None,
        comparison,
        target_verified=match is not None,
        earliest_valid_occurrence=match is not None,
    )
    early_stop = (
        search_result is not None
        and search_result.match is not None
        and search_result.processed_chunks < total_chunks
    )
    media_duration = observation.media_info.duration if observation.media_info is not None else None
    audio_duration = _optional_number(baseline, "audio_duration_seconds") or media_duration
    timestamp_difference = timestamp_delta(
        match.start if match is not None else None,
        baseline_start,
    )
    unique_coverage = (
        search_result.processed_unique_audio_seconds if search_result is not None else 0.0
    )
    overlap_overhead = max(0.0, processed_audio - unique_coverage)
    configuration_parity = benchmark_configuration_parity(
        case,
        baseline,
        observation.media_path,
    )

    return {
        "case_id": case.case_id,
        "url": case.url,
        "media_path": str(observation.media_path) if observation.media_path is not None else None,
        "source_page_url": case.source_page_url,
        "target": case.target,
        "status": "ok" if localization is not None else "error",
        "strategy": "chronological_chunked_asr_with_overlap_and_early_stopping",
        "chunk_duration_seconds": config.chunk_duration_seconds,
        "overlap_seconds": config.overlap_seconds,
        "transcript_context_seconds": config.transcript_context_seconds,
        "total_wall_clock_seconds": total_wall_clock,
        "total_wall_clock_hms": format_timestamp(total_wall_clock),
        "asr_wall_clock_seconds": asr_wall_clock,
        "asr_wall_clock_hms": format_timestamp(asr_wall_clock),
        "media_duration_seconds": media_duration,
        "media_duration_hms": format_timestamp(media_duration),
        "audio_duration_seconds": audio_duration,
        "audio_duration_hms": format_timestamp(audio_duration),
        "expensive_asr_audio_seconds_processed": processed_audio,
        "expensive_asr_audio_processed_hms": format_timestamp(processed_audio),
        "unique_audio_coverage_seconds": unique_coverage,
        "total_overlap_overhead_seconds": overlap_overhead,
        "total_overlap_overhead_percentage": (
            overlap_overhead / unique_coverage * 100.0 if unique_coverage > 0 else 0.0
        ),
        "audio_extraction_calls": observation.audio_extraction_calls,
        "audio_extraction_wall_clock_seconds": observation.audio_extraction_wall_clock_seconds,
        "chunks_processed": search_result.processed_chunks if search_result is not None else 0,
        "chunks_total": total_chunks,
        "early_stop_triggered": early_stop,
        "early_stop_chunk_index": (
            search_result.stopped_on_chunk_index if early_stop and search_result is not None else None
        ),
        "early_stop_coverage_timestamp_seconds": (
            search_result.stopped_at_coverage_seconds
            if early_stop and search_result is not None
            else None
        ),
        "early_stop_coverage_timestamp_hms": format_timestamp(
            search_result.stopped_at_coverage_seconds
            if early_stop and search_result is not None
            else None
        ),
        "detected_timestamp_seconds": match.start if match is not None else None,
        "detected_timestamp_hms": format_timestamp(match.start if match is not None else None),
        "timestamp_delta_vs_baseline_seconds": timestamp_difference,
        "matched_text": match.matched_text if match is not None else None,
        "match_type": match.match_type if match is not None else None,
        "match_score": match.score if match is not None else None,
        "fuzzy_threshold": case.fuzzy_threshold,
        "same_first_occurrence_as_baseline": same_first,
        "baseline_configuration_parity": configuration_parity,
        "baseline_detected_timestamp_seconds": baseline_start,
        "baseline_expensive_asr_audio_seconds": baseline_audio,
        "percentage_asr_audio_avoided": percentage_asr_audio_avoided(
            processed_audio,
            baseline_audio,
        ),
        "total_wall_clock_speedup_ratio": speedup_ratio(
            baseline_total_wall,
            total_wall_clock,
        ),
        "asr_wall_clock_speedup_ratio": speedup_ratio(
            baseline_asr_wall,
            asr_wall_clock,
        ),
        "peak_rss_bytes": memory.peak_bytes,
        "fallback_used": observation.acquisition_metadata.get("extractor") == "direct-http",
        "fallback_reason": (
            "direct-http acquisition fallback"
            if observation.acquisition_metadata.get("extractor") == "direct-http"
            else None
        ),
        "error_reason": error_reason,
        "media_cache_hit": observation.acquisition_metadata.get("media_cache_hit", False),
        "media_metadata_cache_hit": observation.media_metadata_cache_hit,
        "model": case.model,
        "device": case.device,
        "compute_type": case.compute_type,
        "language": case.language,
    }


def benchmark_configuration_parity(
    case: BenchmarkCase,
    baseline: dict[str, Any] | None,
    optimized_media_path: Path | None,
) -> dict[str, bool]:
    """Verify that a baseline comparison changes only localization strategy."""
    baseline_media = _optional_string(baseline, "media_path")
    same_media = bool(
        baseline_media
        and optimized_media_path is not None
        and Path(baseline_media).resolve() == optimized_media_path.resolve()
    )
    same_asr = bool(
        baseline
        and baseline.get("model") == case.model
        and baseline.get("device") == case.device
        and baseline.get("compute_type") == case.compute_type
        and baseline.get("language") == case.language
    )
    same_input = bool(
        baseline
        and baseline.get("url") == case.url
        and baseline.get("target") == case.target
    )
    same_matching = bool(
        baseline
        and baseline.get("_matching_logic") == MATCHING_LOGIC_ID
        and _optional_number(baseline, "fuzzy_threshold") == case.fuzzy_threshold
    )
    same_environment = bool(
        baseline and baseline.get("_benchmark_environment") == _current_environment()
    )
    checks = {
        "media_identical": same_media,
        "input_identical": same_input,
        "asr_settings_identical": same_asr,
        "matching_logic_and_threshold_identical": same_matching,
        "hardware_environment_identical": same_environment,
    }
    return {**checks, "all_identical": all(checks.values())}


def _current_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dialogue_locator_version": __version__,
    }


class _NoChunkedMatch(Exception):
    def __init__(self, search: ChunkSearchResult, total_chunks: int) -> None:
        super().__init__("dialogue not found")
        self.search = search
        self.total_chunks = total_chunks


def _localize_chunked(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    config: ChunkedASRConfig,
    observation: ChunkedObservation,
    *,
    clock: Clock,
) -> tuple[ChunkedLocalization, ChunkedObservation]:
    defaults = manifest.defaults
    tools = require_external_tools()
    if case.local_media_path is not None:
        url = case.url
        media_path = case.local_media_path.resolve()
        if not media_path.is_file():
            raise V0Error(f"Controlled fixture media does not exist: {media_path}")
        metadata = {"extractor": "local-fixture", "media_cache_hit": True}
    else:
        url = validate_public_url(case.url)
        media_path, metadata = acquire_media(
            url,
            defaults.work_dir,
            cookies_from_browser=defaults.cookies_from_browser,
            cookie_file=defaults.cookies_file,
        )
    observation.acquisition_metadata = dict(metadata)
    observation.media_path = media_path
    media, metadata_cache_hit = pipeline._inspect_media_cached(
        media_path,
        tools.ffprobe,
        defaults.model_cache.parent / "pipeline-cache",
    )
    observation.media_info = media
    observation.media_metadata_cache_hit = metadata_cache_hit
    pipeline._require_audio_video(media)
    if media.duration is None or media.duration <= 0:
        raise V0Error("Chunked ASR requires a positive ffprobe media duration.")

    resolved_model = resolve_model_name(case.model, case.language)
    transcriber = FasterWhisperTranscriber(
        model_name=resolved_model,
        model_cache=defaults.model_cache,
        device=case.device,
        compute_type=case.compute_type,
        language=case.language,
    )
    defaults.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="chunked-asr-",
        dir=defaults.work_dir,
        ignore_cleanup_errors=True,
    ) as temporary:
        temporary_dir = Path(temporary)
        extraction_started = clock()
        full_audio_path = extract_speech_audio(
            media_path,
            temporary_dir / "full-audio.wav",
            tools.ffmpeg,
        )
        observation.audio_extraction_calls = 1
        observation.audio_extraction_wall_clock_seconds = max(
            0.0, clock() - extraction_started
        )
        with ReusableWaveAudio(full_audio_path) as audio_source:
            chunks = generate_chunks(
                audio_source.duration,
                config.chunk_duration_seconds,
                config.overlap_seconds,
            )
            reusable_transcriber = ReusableChunkTranscriber(transcriber, audio_source)

            def transcribe_chunk(chunk: AudioChunk) -> Transcription:
                audio_seconds = max(
                    0.0,
                    min(audio_source.duration, chunk.end)
                    - min(audio_source.duration, chunk.start),
                )
                asr_started = clock()
                try:
                    return reusable_transcriber(chunk)
                finally:
                    observation.asr_wall_clock_seconds.append(max(0.0, clock() - asr_started))
                    observation.asr_audio_seconds.append(audio_seconds)

            search = search_chunks(
                case.target,
                chunks,
                transcribe_chunk,
                fuzzy_threshold=case.fuzzy_threshold,
                audio_start_offset=media.audio_start_time or 0.0,
                overlap_seconds=config.overlap_seconds,
                transcript_context_seconds=config.transcript_context_seconds,
            )
    if search.match is None:
        raise _NoChunkedMatch(search, len(chunks))
    frame = resolve_frame_at_timestamp(
        media_path,
        search.match.start,
        defaults.output_dir / "chunked" / case.case_id,
    )
    return ChunkedLocalization(search.match, frame, search, len(chunks)), observation


def _comparison_baseline(
    case: BenchmarkCase,
    baseline_start: float | None,
    baseline_text: str | None,
) -> ProductionBaseline | None:
    if baseline_start is None:
        return case.production_baseline
    tolerance = (
        case.production_baseline.timestamp_tolerance_seconds
        if case.production_baseline is not None
        else 0.05
    )
    return ProductionBaseline(
        dialogue_start_seconds=baseline_start,
        timestamp_tolerance_seconds=tolerance,
        matched_text=baseline_text,
    )


def _optional_number(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None or isinstance(record.get(key), bool):
        return None
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value


def _optional_string(record: dict[str, Any] | None, key: str) -> str | None:
    if record is None:
        return None
    value = record.get(key)
    return value if isinstance(value, str) else None
