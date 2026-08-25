from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dialogue_locator import __version__, pipeline
from dialogue_locator.acquisition import acquire_media, validate_public_url
from dialogue_locator.audio import extract_speech_audio
from dialogue_locator.dependencies import require_external_tools
from dialogue_locator.errors import V0Error
from dialogue_locator.frames import resolve_frame_at_timestamp
from dialogue_locator.models import DialogueMatch, MediaInfo, ResolvedFrame, format_timestamp
from dialogue_locator.transcription import FasterWhisperTranscriber, resolve_model_name

from experiments.audio_localization_baseline.manifest import (
    BenchmarkCase,
    BenchmarkManifest,
    ManifestError,
    ProductionBaseline,
)
from experiments.audio_localization_baseline.metrics import first_occurrence_matches_baseline
from experiments.audio_localization_baseline.runner import PeakRSSSampler, _write_json_atomic
from experiments.audio_localization_chunked.chunking import (
    AudioChunk,
    ChunkSearchResult,
    generate_chunks,
    search_chunks,
)
from experiments.audio_localization_chunked.manifest import ChunkedASRConfig
from experiments.audio_localization_chunked.metrics import (
    percentage_asr_audio_avoided,
    speedup_ratio,
    timestamp_delta,
)
from experiments.audio_localization_chunked.runner import load_baseline_results

from .fallbacks import global_fallback_reason
from .manifest import ConservativeVADConfig
from .transcriber import ConservativeVADTranscriber, VADObservation


Clock = Callable[[], float]


@dataclass(frozen=True)
class VADLocalization:
    match: DialogueMatch
    frame: ResolvedFrame
    vad_search: ChunkSearchResult
    fallback_search: ChunkSearchResult | None
    total_chunks: int
    fallback_reasons: tuple[str, ...]
    observation: VADObservation
    media: MediaInfo
    media_metadata_cache_hit: bool
    acquisition_metadata: dict[str, Any]


class _NoVADMatch(Exception):
    def __init__(
        self,
        vad_search: ChunkSearchResult,
        fallback_search: ChunkSearchResult | None,
        total_chunks: int,
        fallback_reasons: tuple[str, ...],
        observation: VADObservation,
        media: MediaInfo,
        media_metadata_cache_hit: bool,
        acquisition_metadata: dict[str, Any],
    ) -> None:
        super().__init__("dialogue not found")
        self.vad_search = vad_search
        self.fallback_search = fallback_search
        self.total_chunks = total_chunks
        self.fallback_reasons = fallback_reasons
        self.observation = observation
        self.media = media
        self.media_metadata_cache_hit = media_metadata_cache_hit
        self.acquisition_metadata = acquisition_metadata


def load_strategy_results(path: Path, label: str) -> dict[str, dict[str, Any]]:
    try:
        return load_baseline_results(path)
    except ManifestError as exc:
        raise ManifestError(f"Invalid {label} results: {exc}") from exc


def run_vad_benchmark(
    manifest: BenchmarkManifest,
    chunk_config: ChunkedASRConfig,
    vad_config: ConservativeVADConfig,
    baseline_results: dict[str, dict[str, Any]],
    chunked_results: dict[str, dict[str, Any]],
    *,
    manifest_path: Path,
    baseline_results_path: Path,
    chunked_results_path: Path,
    output_path: Path,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    records = [
        _run_case(
            case,
            manifest,
            chunk_config,
            vad_config,
            baseline_results.get(case.case_id),
            chunked_results.get(case.case_id),
            clock=clock,
        )
        for case in manifest.cases
    ]
    report = {
        "schema_version": 1,
        "benchmark": "conservative-vad-chronological-chunked-asr-early-stop",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "baseline_results_path": str(baseline_results_path.resolve()),
        "chunked_results_path": str(chunked_results_path.resolve()),
        "chunked_asr": asdict(chunk_config),
        "vad": asdict(vad_config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dialogue_locator_version": __version__,
            "vad_backend": "faster-whisper bundled Silero VAD (ONNX Runtime)",
        },
        "cases": records,
    }
    _write_json_atomic(output_path, report)
    return report


def _run_case(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    chunk_config: ChunkedASRConfig,
    vad_config: ConservativeVADConfig,
    baseline: dict[str, Any] | None,
    chunked: dict[str, Any] | None,
    *,
    clock: Clock,
) -> dict[str, Any]:
    localization: VADLocalization | None = None
    error_reason: str | None = None
    observation = VADObservation()
    media: MediaInfo | None = None
    metadata_cache_hit = False
    acquisition_metadata: dict[str, Any] = {}
    vad_search: ChunkSearchResult | None = None
    fallback_search: ChunkSearchResult | None = None
    total_chunks = 0
    fallback_reasons: tuple[str, ...] = ()
    started = clock()
    with PeakRSSSampler() as memory:
        try:
            localization = _localize_vad(
                case,
                manifest,
                chunk_config,
                vad_config,
                baseline,
                clock=clock,
            )
            observation = localization.observation
            media = localization.media
            metadata_cache_hit = localization.media_metadata_cache_hit
            acquisition_metadata = localization.acquisition_metadata
            vad_search = localization.vad_search
            fallback_search = localization.fallback_search
            total_chunks = localization.total_chunks
            fallback_reasons = localization.fallback_reasons
        except _NoVADMatch as exc:
            observation = exc.observation
            media = exc.media
            metadata_cache_hit = exc.media_metadata_cache_hit
            acquisition_metadata = exc.acquisition_metadata
            vad_search = exc.vad_search
            fallback_search = exc.fallback_search
            total_chunks = exc.total_chunks
            fallback_reasons = exc.fallback_reasons
            error_reason = (
                "V0Error: Dialogue not found after conservative VAD and unfiltered "
                "chronological chunked-ASR fallback."
            )
        except Exception as exc:
            error_reason = f"{type(exc).__name__}: {exc}"
    total_wall = max(0.0, clock() - started)

    match = localization.match if localization is not None else None
    baseline_start = _optional_number(baseline, "detected_timestamp_seconds")
    baseline_text = _optional_string(baseline, "matched_text")
    baseline_audio = _optional_number(baseline, "expensive_asr_audio_seconds_processed")
    baseline_total = _optional_number(baseline, "total_wall_clock_seconds")
    baseline_asr = _optional_number(baseline, "asr_wall_clock_seconds")
    chunked_audio = _optional_number(chunked, "expensive_asr_audio_seconds_processed")
    chunked_total = _optional_number(chunked, "total_wall_clock_seconds")
    chunked_asr = _optional_number(chunked, "asr_wall_clock_seconds")
    asr_audio = sum(observation.expensive_asr_audio_seconds)
    asr_wall = sum(observation.asr_wall_clock_seconds)
    vad_wall = sum(observation.vad_wall_clock_seconds)
    vad_original = sum(observation.original_audio_seconds)
    speech_retained = sum(observation.speech_audio_seconds)
    full_original = _optional_number(baseline, "audio_duration_seconds")
    if full_original is None and media is not None:
        full_original = media.duration
    reference = _reference(case, baseline_start, baseline_text)
    correct = first_occurrence_matches_baseline(
        match.start if match is not None else None,
        match.matched_text if match is not None else None,
        reference,
    )
    fallback_invoked = bool(fallback_reasons)
    final_search = fallback_search or vad_search
    early_stop = (
        final_search is not None
        and final_search.match is not None
        and final_search.processed_chunks < total_chunks
    )

    return {
        "case_id": case.case_id,
        "url": case.url,
        "target": case.target,
        "status": "ok" if localization is not None else "error",
        "strategy": "conservative_vad_chunked_asr_early_stop",
        "total_wall_clock_seconds": total_wall,
        "total_wall_clock_hms": format_timestamp(total_wall),
        "vad_wall_clock_seconds": vad_wall,
        "vad_wall_clock_hms": format_timestamp(vad_wall),
        "asr_wall_clock_seconds": asr_wall,
        "asr_wall_clock_hms": format_timestamp(asr_wall),
        "total_original_audio_duration_seconds": full_original,
        "total_original_audio_duration_hms": format_timestamp(full_original),
        "vad_original_audio_seconds_examined": vad_original,
        "vad_speech_duration_retained_seconds": speech_retained,
        "vad_speech_duration_retained_hms": format_timestamp(speech_retained),
        "percentage_audio_removed_by_vad": _removed_percentage(
            vad_original,
            speech_retained,
        ),
        "expensive_asr_audio_seconds_processed": asr_audio,
        "expensive_asr_audio_processed_hms": format_timestamp(asr_audio),
        "vad_chunks_processed": vad_search.processed_chunks if vad_search is not None else 0,
        "fallback_chunks_processed": (
            fallback_search.processed_chunks if fallback_search is not None else 0
        ),
        "chunks_total": total_chunks,
        "asr_call_count": observation.asr_calls,
        "early_stop_triggered": early_stop,
        "early_stop_chunk_index": (
            final_search.stopped_on_chunk_index if early_stop and final_search is not None else None
        ),
        "detected_timestamp_seconds": match.start if match is not None else None,
        "detected_timestamp_hms": format_timestamp(match.start if match is not None else None),
        "timestamp_delta_vs_baseline_seconds": timestamp_delta(
            match.start if match is not None else None,
            baseline_start,
        ),
        "matched_text": match.matched_text if match is not None else None,
        "match_type": match.match_type if match is not None else None,
        "match_score": match.score if match is not None else None,
        "same_first_occurrence_as_baseline": correct,
        "fallback_invoked": fallback_invoked,
        "fallback_reasons": list(fallback_reasons),
        "percentage_asr_audio_avoided_vs_baseline": percentage_asr_audio_avoided(
            asr_audio,
            baseline_audio,
        ),
        "percentage_asr_audio_avoided_vs_chunked": percentage_asr_audio_avoided(
            asr_audio,
            chunked_audio,
        ),
        "total_wall_clock_speedup_vs_baseline": speedup_ratio(baseline_total, total_wall),
        "total_wall_clock_speedup_vs_chunked": speedup_ratio(chunked_total, total_wall),
        "asr_wall_clock_speedup_vs_baseline": speedup_ratio(baseline_asr, asr_wall),
        "asr_wall_clock_speedup_vs_chunked": speedup_ratio(chunked_asr, asr_wall),
        "baseline_total_wall_clock_seconds": baseline_total,
        "chunked_total_wall_clock_seconds": chunked_total,
        "baseline_expensive_asr_audio_seconds": baseline_audio,
        "chunked_expensive_asr_audio_seconds": chunked_audio,
        "peak_rss_bytes": memory.peak_bytes,
        "media_cache_hit": acquisition_metadata.get("media_cache_hit", False),
        "media_metadata_cache_hit": metadata_cache_hit,
        "fallback_used": fallback_invoked
        or acquisition_metadata.get("extractor") == "direct-http",
        "error_reason": error_reason,
        "model": case.model,
        "device": case.device,
        "compute_type": case.compute_type,
        "language": case.language,
    }


def _localize_vad(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    chunk_config: ChunkedASRConfig,
    vad_config: ConservativeVADConfig,
    baseline: dict[str, Any] | None,
    *,
    clock: Clock,
) -> VADLocalization:
    defaults = manifest.defaults
    url = validate_public_url(case.url)
    tools = require_external_tools()
    media_path, metadata = acquire_media(
        url,
        defaults.work_dir,
        cookies_from_browser=defaults.cookies_from_browser,
        cookie_file=defaults.cookies_file,
    )
    media, metadata_cache_hit = pipeline._inspect_media_cached(
        media_path,
        tools.ffprobe,
        defaults.model_cache.parent / "pipeline-cache",
    )
    pipeline._require_audio_video(media)
    if media.duration is None or media.duration <= 0:
        raise V0Error("VAD chunked ASR requires a positive ffprobe media duration.")
    chunks = generate_chunks(
        media.duration,
        chunk_config.chunk_duration_seconds,
        chunk_config.overlap_seconds,
    )
    resolved_model = resolve_model_name(case.model, case.language)
    base = FasterWhisperTranscriber(
        model_name=resolved_model,
        model_cache=defaults.model_cache,
        device=case.device,
        compute_type=case.compute_type,
        language=case.language,
    )
    observation = VADObservation()
    vad_transcriber = ConservativeVADTranscriber(
        base,
        vad_config,
        observation,
        clock=clock,
    )
    fallback_reasons: list[str] = []
    defaults.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="vad-chunked-asr-",
        dir=defaults.work_dir,
        ignore_cleanup_errors=True,
    ) as temporary:
        temporary_dir = Path(temporary)
        chunk_paths: dict[int, Path] = {}

        def audio_path_for(chunk: AudioChunk) -> Path:
            cached = chunk_paths.get(chunk.index)
            if cached is not None:
                return cached
            path = extract_speech_audio(
                media_path,
                temporary_dir / f"chunk-{chunk.index:04d}.wav",
                tools.ffmpeg,
                start_time=chunk.start,
                duration=chunk.duration,
            )
            chunk_paths[chunk.index] = path
            return path

        vad_search = search_chunks(
            case.target,
            chunks,
            lambda chunk: vad_transcriber(audio_path_for(chunk)),
            fuzzy_threshold=case.fuzzy_threshold,
            audio_start_offset=media.audio_start_time or 0.0,
        )
        fallback_reasons.extend(observation.chunk_fallback_reasons)

        reference = _reference(
            case,
            _optional_number(baseline, "detected_timestamp_seconds"),
            _optional_string(baseline, "matched_text"),
        )
        vad_correct = first_occurrence_matches_baseline(
            vad_search.match.start if vad_search.match is not None else None,
            vad_search.match.matched_text if vad_search.match is not None else None,
            reference,
        )
        global_reason = global_fallback_reason(
            vad_found_match=vad_search.match is not None,
            vad_matches_baseline=vad_correct,
            config=vad_config,
        )
        fallback_search = None
        if global_reason is not None:
            fallback_reasons.append(global_reason)
            fallback_search = search_chunks(
                case.target,
                chunks,
                lambda chunk: vad_transcriber.transcribe_unfiltered(audio_path_for(chunk)),
                fuzzy_threshold=case.fuzzy_threshold,
                audio_start_offset=media.audio_start_time or 0.0,
            )

    final_search = fallback_search or vad_search
    if final_search.match is None:
        raise _NoVADMatch(
            vad_search,
            fallback_search,
            len(chunks),
            tuple(fallback_reasons),
            observation,
            media,
            metadata_cache_hit,
            dict(metadata),
        )
    frame = resolve_frame_at_timestamp(
        media_path,
        final_search.match.start,
        defaults.output_dir / "chunked-vad" / case.case_id,
    )
    return VADLocalization(
        match=final_search.match,
        frame=frame,
        vad_search=vad_search,
        fallback_search=fallback_search,
        total_chunks=len(chunks),
        fallback_reasons=tuple(fallback_reasons),
        observation=observation,
        media=media,
        media_metadata_cache_hit=metadata_cache_hit,
        acquisition_metadata=dict(metadata),
    )


def _reference(
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
    return ProductionBaseline(baseline_start, tolerance, baseline_text)


def _removed_percentage(original: float, retained: float) -> float | None:
    if original <= 0:
        return None
    return max(0.0, min(100.0, (original - retained) / original * 100.0))


def _optional_number(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None or isinstance(record.get(key), bool):
        return None
    try:
        return float(record[key])
    except (KeyError, TypeError, ValueError):
        return None


def _optional_string(record: dict[str, Any] | None, key: str) -> str | None:
    if record is None:
        return None
    value = record.get(key)
    return value if isinstance(value, str) else None
