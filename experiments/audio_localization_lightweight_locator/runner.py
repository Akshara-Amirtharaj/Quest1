from __future__ import annotations

import platform
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dialogue_locator import __version__, pipeline
from dialogue_locator.acquisition import acquire_media, validate_public_url
from dialogue_locator.audio import extract_speech_audio
from dialogue_locator.dependencies import require_external_tools
from dialogue_locator.errors import V0Error
from dialogue_locator.frames import resolve_frame_at_timestamp
from dialogue_locator.models import DialogueMatch, MediaInfo, ResolvedFrame, Transcription, format_timestamp
from dialogue_locator.transcription import FasterWhisperTranscriber, resolve_model_name

from experiments.audio_localization_baseline.manifest import (
    BenchmarkCase,
    BenchmarkManifest,
    ProductionBaseline,
)
from experiments.audio_localization_baseline.metrics import first_occurrence_matches_baseline
from experiments.audio_localization_baseline.runner import PeakRSSSampler, _write_json_atomic
from experiments.audio_localization_chunked.metrics import (
    percentage_asr_audio_avoided,
    speedup_ratio,
    timestamp_delta,
)

from .localization import (
    CandidateWindow,
    LocatorSearchResult,
    generate_candidate_windows,
    verify_candidate_windows,
)
from .manifest import LocatorConfig


Clock = Callable[[], float]


@dataclass
class LocatorObservation:
    locator_wall_seconds: list[float] = field(default_factory=list)
    locator_audio_seconds: list[float] = field(default_factory=list)
    accurate_wall_seconds: list[float] = field(default_factory=list)
    accurate_audio_seconds: list[float] = field(default_factory=list)
    acquisition_metadata: dict[str, Any] = field(default_factory=dict)
    media: MediaInfo | None = None
    media_metadata_cache_hit: bool = False


@dataclass(frozen=True)
class LocatorLocalization:
    match: DialogueMatch
    frame: ResolvedFrame
    search: LocatorSearchResult
    observation: LocatorObservation


class _NoLocatorMatch(Exception):
    def __init__(self, search: LocatorSearchResult, observation: LocatorObservation) -> None:
        super().__init__("dialogue not found")
        self.search = search
        self.observation = observation


def run_locator_benchmark(
    manifest: BenchmarkManifest,
    config: LocatorConfig,
    baseline_results: dict[str, dict[str, Any]],
    chunked_results: dict[str, dict[str, Any]],
    vad_results: dict[str, dict[str, Any]],
    *,
    manifest_path: Path,
    baseline_results_path: Path,
    chunked_results_path: Path,
    vad_results_path: Path,
    output_path: Path,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    records = [
        _run_case(
            case,
            manifest,
            config,
            baseline_results.get(case.case_id),
            chunked_results.get(case.case_id),
            vad_results.get(case.case_id),
            clock=clock,
        )
        for case in manifest.cases
    ]
    report = {
        "schema_version": 1,
        "benchmark": "lightweight-whisper-localization-accurate-asr-verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "baseline_results_path": str(baseline_results_path.resolve()),
        "chunked_results_path": str(chunked_results_path.resolve()),
        "vad_results_path": str(vad_results_path.resolve()),
        "lightweight_locator": asdict(config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dialogue_locator_version": __version__,
            "asr_stack": "faster-whisper",
        },
        "cases": records,
    }
    _write_json_atomic(output_path, report)
    return report


def _run_case(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    config: LocatorConfig,
    baseline: dict[str, Any] | None,
    chunked: dict[str, Any] | None,
    vad: dict[str, Any] | None,
    *,
    clock: Clock,
) -> dict[str, Any]:
    observation = LocatorObservation()
    localization: LocatorLocalization | None = None
    search: LocatorSearchResult | None = None
    error_reason: str | None = None
    started = clock()
    with PeakRSSSampler() as memory:
        try:
            localization = _localize(case, manifest, config, observation, clock=clock)
            search = localization.search
        except _NoLocatorMatch as exc:
            search = exc.search
            observation = exc.observation
            error_reason = (
                "V0Error: Dialogue not found after lightweight localization, accurate "
                "candidate verification, and full-ASR fallback."
            )
        except Exception as exc:
            error_reason = f"{type(exc).__name__}: {exc}"
    total_wall = max(0.0, clock() - started)

    match = localization.match if localization is not None else None
    baseline_start = _optional_number(baseline, "detected_timestamp_seconds")
    baseline_text = _optional_string(baseline, "matched_text")
    baseline_audio = _optional_number(baseline, "expensive_asr_audio_seconds_processed")
    baseline_total = _optional_number(baseline, "total_wall_clock_seconds")
    chunked_total = _optional_number(chunked, "total_wall_clock_seconds")
    vad_total = _optional_number(vad, "total_wall_clock_seconds")
    locator_wall = sum(observation.locator_wall_seconds)
    accurate_wall = sum(observation.accurate_wall_seconds)
    locator_audio = sum(observation.locator_audio_seconds)
    accurate_audio = sum(observation.accurate_audio_seconds)
    reference = _comparison_baseline(case, baseline_start, baseline_text)
    correct = first_occurrence_matches_baseline(
        match.start if match is not None else None,
        match.matched_text if match is not None else None,
        reference,
    )
    candidates = search.candidates if search is not None else ()
    direct_http = observation.acquisition_metadata.get("extractor") == "direct-http"

    return {
        "case_id": case.case_id,
        "url": case.url,
        "source_page_url": getattr(case, "source_page_url", None),
        "target": case.target,
        "status": "ok" if localization is not None else "error",
        "strategy": "lightweight_whisper_locator_accurate_asr_verification",
        "locator_model": config.model,
        "accurate_model": case.model,
        "total_wall_clock_seconds": total_wall,
        "total_wall_clock_hms": format_timestamp(total_wall),
        "locator_asr_wall_clock_seconds": locator_wall,
        "locator_asr_wall_clock_hms": format_timestamp(locator_wall),
        "accurate_verification_asr_wall_clock_seconds": accurate_wall,
        "accurate_verification_asr_wall_clock_hms": format_timestamp(accurate_wall),
        "lightweight_audio_seconds_processed": locator_audio,
        "lightweight_audio_processed_hms": format_timestamp(locator_audio),
        "accurate_audio_seconds_processed": accurate_audio,
        "accurate_audio_processed_hms": format_timestamp(accurate_audio),
        "media_duration_seconds": observation.media.duration if observation.media else None,
        "candidate_count": len(candidates),
        "candidates_verified": search.candidates_verified if search is not None else 0,
        "verified_candidate_index": (
            search.verified_candidate_index if search is not None else None
        ),
        "candidate_windows": [
            {
                "index": candidate.index,
                "start_seconds": candidate.start,
                "end_seconds": candidate.end,
                "duration_seconds": candidate.duration,
                "locator_matched_text": candidate.locator_match.matched_text,
                "locator_match_type": candidate.locator_match.match_type,
                "locator_match_score": candidate.locator_match.score,
            }
            for candidate in candidates
        ],
        "candidate_window_durations_seconds": [candidate.duration for candidate in candidates],
        "fallback_invoked": search.fallback_invoked if search is not None else False,
        "fallback_reason": search.fallback_reason if search is not None else None,
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
        "percentage_expensive_asr_audio_avoided_vs_baseline": percentage_asr_audio_avoided(
            accurate_audio,
            baseline_audio,
        ),
        "total_wall_clock_speedup_vs_baseline": speedup_ratio(baseline_total, total_wall),
        "total_wall_clock_speedup_vs_chunked": speedup_ratio(chunked_total, total_wall),
        "total_wall_clock_speedup_vs_chunked_vad": speedup_ratio(vad_total, total_wall),
        "peak_rss_bytes": memory.peak_bytes,
        "fallback_used": (search.fallback_invoked if search is not None else False) or direct_http,
        "error_reason": error_reason,
        "media_cache_hit": observation.acquisition_metadata.get("media_cache_hit", False),
        "media_metadata_cache_hit": observation.media_metadata_cache_hit,
        "device": case.device,
        "compute_type": case.compute_type,
        "language": case.language,
        "fuzzy_threshold": case.fuzzy_threshold,
    }


def _localize(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    config: LocatorConfig,
    observation: LocatorObservation,
    *,
    clock: Clock,
) -> LocatorLocalization:
    defaults = manifest.defaults
    tools = require_external_tools()
    media_path, metadata = acquire_media(
        validate_public_url(case.url),
        defaults.work_dir,
        cookies_from_browser=defaults.cookies_from_browser,
        cookie_file=defaults.cookies_file,
    )
    observation.acquisition_metadata = dict(metadata)
    media, metadata_cache_hit = pipeline._inspect_media_cached(
        media_path,
        tools.ffprobe,
        defaults.model_cache.parent / "pipeline-cache",
    )
    observation.media = media
    observation.media_metadata_cache_hit = metadata_cache_hit
    pipeline._require_audio_video(media)

    locator_model_name = resolve_model_name(config.model, case.language)
    accurate_model_name = resolve_model_name(case.model, case.language)
    locator = FasterWhisperTranscriber(
        model_name=locator_model_name,
        model_cache=defaults.model_cache,
        device=case.device,
        compute_type=case.compute_type,
        language=case.language,
    )
    accurate = FasterWhisperTranscriber(
        model_name=accurate_model_name,
        model_cache=defaults.model_cache,
        device=case.device,
        compute_type=case.compute_type,
        language=case.language,
    )
    defaults.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="lightweight-locator-",
        dir=defaults.work_dir,
        ignore_cleanup_errors=True,
    ) as temporary:
        temporary_dir = Path(temporary)
        full_audio = extract_speech_audio(
            media_path,
            temporary_dir / "full-audio.wav",
            tools.ffmpeg,
        )
        audio_duration = _wav_duration(full_audio)
        locator_reason: str | None = None
        try:
            locator_transcription = _measured_transcribe(
                locator,
                full_audio,
                observation.locator_wall_seconds,
                observation.locator_audio_seconds,
                clock,
            )
            candidates = generate_candidate_windows(
                case.target,
                locator_transcription.words,
                audio_duration=audio_duration,
                margin_before=config.candidate_margin_before_seconds,
                margin_after=config.candidate_margin_after_seconds,
                fuzzy_threshold=case.fuzzy_threshold,
            )
        except V0Error as exc:
            candidates = ()
            locator_reason = f"lightweight locator failed: {exc}"

        window_paths: dict[int, Path] = {}

        def transcribe_window(candidate: CandidateWindow) -> Transcription:
            audio_path = window_paths.get(candidate.index)
            if audio_path is None:
                audio_path = extract_speech_audio(
                    media_path,
                    temporary_dir / f"candidate-{candidate.index:04d}.wav",
                    tools.ffmpeg,
                    start_time=candidate.start,
                    duration=candidate.duration,
                )
                window_paths[candidate.index] = audio_path
            return _measured_transcribe(
                accurate,
                audio_path,
                observation.accurate_wall_seconds,
                observation.accurate_audio_seconds,
                clock,
            )

        def transcribe_full() -> Transcription:
            return _measured_transcribe(
                accurate,
                full_audio,
                observation.accurate_wall_seconds,
                observation.accurate_audio_seconds,
                clock,
            )

        search = verify_candidate_windows(
            case.target,
            candidates,
            transcribe_window,
            transcribe_full,
            fuzzy_threshold=case.fuzzy_threshold,
            audio_start_offset=media.audio_start_time or 0.0,
            locator_failure_reason=locator_reason,
        )
    if search.match is None:
        raise _NoLocatorMatch(search, observation)
    frame = resolve_frame_at_timestamp(
        media_path,
        search.match.start,
        defaults.output_dir / "lightweight-locator" / case.case_id,
    )
    return LocatorLocalization(search.match, frame, search, observation)


def _measured_transcribe(
    transcriber: FasterWhisperTranscriber,
    audio_path: Path,
    wall_values: list[float],
    audio_values: list[float],
    clock: Clock,
) -> Transcription:
    audio_seconds = _wav_duration(audio_path)
    started = clock()
    try:
        return transcriber(audio_path)
    finally:
        wall_values.append(max(0.0, clock() - started))
        audio_values.append(audio_seconds)


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
    return ProductionBaseline(baseline_start, tolerance, baseline_text)


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


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid WAV frame rate in {path}.")
        return audio.getnframes() / rate
