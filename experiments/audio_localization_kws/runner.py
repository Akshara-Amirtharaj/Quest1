from __future__ import annotations

import importlib.metadata
import platform
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dialogue_locator import __version__, pipeline
from dialogue_locator.acquisition import acquire_media, validate_public_url
from dialogue_locator.audio import extract_speech_audio
from dialogue_locator.dependencies import require_external_tools
from dialogue_locator.errors import V0Error
from dialogue_locator.frames import resolve_frame_at_timestamp
from dialogue_locator.matching import find_dialogue_candidates
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
from experiments.audio_localization_lightweight_locator.localization import (
    LocatorSearchResult,
    offset_match,
    verify_candidate_windows,
)

from .anchors import generate_phrase_anchors
from .candidates import AnchorDetection, KWSCandidateRegion, group_detections
from .manifest import KWSConfig
from .sherpa_backend import (
    SherpaKeywordSpotter,
    SherpaKWSModel,
    SherpaKWSOptions,
)


Clock = Callable[[], float]


@dataclass
class KWSObservation:
    kws_wall_seconds: float = 0.0
    accurate_wall_seconds: list[float] = field(default_factory=list)
    accurate_audio_seconds: list[float] = field(default_factory=list)
    acquisition_metadata: dict[str, Any] = field(default_factory=dict)
    media: MediaInfo | None = None
    media_metadata_cache_hit: bool = False


@dataclass(frozen=True)
class KWSLocalization:
    match: DialogueMatch
    frame: ResolvedFrame
    search: LocatorSearchResult
    anchors: tuple[str, ...]
    detections: tuple[AnchorDetection, ...]
    regions: tuple[KWSCandidateRegion, ...]
    observation: KWSObservation
    backend_error: str | None


class _NoKWSMatch(Exception):
    def __init__(
        self,
        search: LocatorSearchResult,
        anchors: tuple[str, ...],
        detections: tuple[AnchorDetection, ...],
        regions: tuple[KWSCandidateRegion, ...],
        observation: KWSObservation,
        backend_error: str | None,
    ) -> None:
        super().__init__("dialogue not found")
        self.search = search
        self.anchors = anchors
        self.detections = detections
        self.regions = regions
        self.observation = observation
        self.backend_error = backend_error


def run_kws_benchmark(
    manifest: BenchmarkManifest,
    config: KWSConfig,
    baseline_results: dict[str, dict[str, Any]],
    chunked_results: dict[str, dict[str, Any]],
    vad_results: dict[str, dict[str, Any]],
    locator_results: dict[str, dict[str, Any]],
    *,
    manifest_path: Path,
    baseline_results_path: Path,
    chunked_results_path: Path,
    vad_results_path: Path,
    locator_results_path: Path,
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
            locator_results.get(case.case_id),
            clock=clock,
        )
        for case in manifest.cases
    ]
    report = {
        "schema_version": 1,
        "benchmark": "open-vocabulary-kws-accurate-asr-verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "baseline_results_path": str(baseline_results_path.resolve()),
        "chunked_results_path": str(chunked_results_path.resolve()),
        "vad_results_path": str(vad_results_path.resolve()),
        "locator_results_path": str(locator_results_path.resolve()),
        "kws": {**asdict(config), "model_dir": str(config.model_dir)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dialogue_locator_version": __version__,
            "sherpa_onnx_version": _package_version("sherpa-onnx"),
        },
        "cases": records,
    }
    _write_json_atomic(output_path, report)
    return report


def _run_case(
    case: BenchmarkCase,
    manifest: BenchmarkManifest,
    config: KWSConfig,
    baseline: dict[str, Any] | None,
    chunked: dict[str, Any] | None,
    vad: dict[str, Any] | None,
    locator: dict[str, Any] | None,
    *,
    clock: Clock,
) -> dict[str, Any]:
    observation = KWSObservation()
    localization: KWSLocalization | None = None
    search: LocatorSearchResult | None = None
    anchors: tuple[str, ...] = ()
    detections: tuple[AnchorDetection, ...] = ()
    regions: tuple[KWSCandidateRegion, ...] = ()
    backend_error: str | None = None
    error_reason: str | None = None
    started = clock()
    with PeakRSSSampler() as memory:
        try:
            localization = _localize(
                case, manifest, config, observation, baseline, clock=clock
            )
            search = localization.search
            anchors = localization.anchors
            detections = localization.detections
            regions = localization.regions
            backend_error = localization.backend_error
        except _NoKWSMatch as exc:
            search = exc.search
            anchors = exc.anchors
            detections = exc.detections
            regions = exc.regions
            observation = exc.observation
            backend_error = exc.backend_error
            error_reason = (
                "V0Error: Dialogue not found after KWS candidate verification and "
                "accurate full-ASR fallback."
            )
        except Exception as exc:
            error_reason = f"{type(exc).__name__}: {exc}"
    total_wall = max(0.0, clock() - started)

    match = localization.match if localization is not None else None
    baseline_start = _number_from(baseline, "detected_timestamp_seconds")
    baseline_text = _string_from(baseline, "matched_text")
    baseline_audio = _number_from(baseline, "expensive_asr_audio_seconds_processed")
    accurate_audio = sum(observation.accurate_audio_seconds)
    accurate_wall = sum(observation.accurate_wall_seconds)
    reference = _comparison_baseline(case, baseline_start, baseline_text)
    correct = first_occurrence_matches_baseline(
        match.start if match else None,
        match.matched_text if match else None,
        reference,
    )
    detections_by_anchor = {
        anchor: [
            {"start_seconds": item.start, "end_seconds": item.end}
            for item in detections
            if item.anchor == anchor
        ]
        for anchor in anchors
    }
    direct_http = observation.acquisition_metadata.get("extractor") == "direct-http"
    return {
        "case_id": case.case_id,
        "url": case.url,
        "source_page_url": getattr(case, "source_page_url", None),
        "target": case.target,
        "status": "ok" if localization else "error",
        "strategy": "open_vocabulary_kws_accurate_asr_verification",
        "kws_backend": "sherpa-onnx",
        "kws_model": config.model_dir.name,
        "accurate_model": case.model,
        "total_wall_clock_seconds": total_wall,
        "total_wall_clock_hms": format_timestamp(total_wall),
        "kws_runtime_seconds": observation.kws_wall_seconds,
        "kws_runtime_hms": format_timestamp(observation.kws_wall_seconds),
        "anchor_phrases": list(anchors),
        "detections_per_anchor": detections_by_anchor,
        "all_anchor_detections": [asdict(item) for item in detections],
        "candidate_count": len(regions),
        "candidate_regions": [
            {
                "index": region.index,
                "start_seconds": region.start,
                "end_seconds": region.end,
                "duration_seconds": region.duration,
                "detections": [asdict(item) for item in region.detections],
            }
            for region in regions
        ],
        "total_candidate_audio_duration_seconds": sum(region.duration for region in regions),
        "candidates_verified": search.candidates_verified if search else 0,
        "verified_candidate_index": search.verified_candidate_index if search else None,
        "accurate_verification_asr_wall_clock_seconds": accurate_wall,
        "accurate_asr_audio_seconds_processed": accurate_audio,
        "fallback_invoked": search.fallback_invoked if search else False,
        "fallback_reason": search.fallback_reason if search else None,
        "kws_backend_error": backend_error,
        "detected_timestamp_seconds": match.start if match else None,
        "detected_timestamp_hms": format_timestamp(match.start if match else None),
        "timestamp_delta_vs_baseline_seconds": timestamp_delta(
            match.start if match else None,
            baseline_start,
        ),
        "matched_text": match.matched_text if match else None,
        "match_type": match.match_type if match else None,
        "match_score": match.score if match else None,
        "same_first_occurrence_as_baseline": correct,
        "percentage_expensive_asr_audio_avoided_vs_baseline": percentage_asr_audio_avoided(
            accurate_audio, baseline_audio
        ),
        "total_wall_clock_speedup_vs_baseline": speedup_ratio(
            _number_from(baseline, "total_wall_clock_seconds"), total_wall
        ),
        "total_wall_clock_speedup_vs_chunked": speedup_ratio(
            _number_from(chunked, "total_wall_clock_seconds"), total_wall
        ),
        "total_wall_clock_speedup_vs_chunked_vad": speedup_ratio(
            _number_from(vad, "total_wall_clock_seconds"), total_wall
        ),
        "total_wall_clock_speedup_vs_lightweight_locator": speedup_ratio(
            _number_from(locator, "total_wall_clock_seconds"), total_wall
        ),
        "peak_rss_bytes": memory.peak_bytes,
        "fallback_used": (search.fallback_invoked if search else False) or direct_http,
        "error_reason": error_reason,
        "media_duration_seconds": observation.media.duration if observation.media else None,
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
    config: KWSConfig,
    observation: KWSObservation,
    baseline: dict[str, Any] | None,
    *,
    clock: Clock,
) -> KWSLocalization:
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
        media_path, tools.ffprobe, defaults.model_cache.parent / "pipeline-cache"
    )
    observation.media = media
    observation.media_metadata_cache_hit = metadata_cache_hit
    pipeline._require_audio_video(media)
    anchors = generate_phrase_anchors(case.target, config.max_anchors)
    accurate = FasterWhisperTranscriber(
        model_name=resolve_model_name(case.model, case.language),
        model_cache=defaults.model_cache,
        device=case.device,
        compute_type=case.compute_type,
        language=case.language,
    )
    spotter = SherpaKeywordSpotter(
        SherpaKWSModel(config.model_dir),
        SherpaKWSOptions(
            num_threads=config.num_threads,
            keywords_score=config.keywords_score,
            keywords_threshold=config.keywords_threshold,
            num_trailing_blanks=config.num_trailing_blanks,
        ),
    )
    defaults.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="kws-localizer-", dir=defaults.work_dir, ignore_cleanup_errors=True
    ) as temporary:
        temporary_dir = Path(temporary)
        full_audio = extract_speech_audio(
            media_path, temporary_dir / "full-audio.wav", tools.ffmpeg
        )
        audio_duration = _wav_duration(full_audio)
        backend_error: str | None = None
        kws_started = clock()
        try:
            detections = spotter.detect(full_audio, anchors, temporary_dir)
        except Exception as exc:
            detections = ()
            backend_error = f"{type(exc).__name__}: {exc}"
        finally:
            observation.kws_wall_seconds = max(0.0, clock() - kws_started)
        regions = group_detections(
            detections,
            audio_duration=audio_duration,
            grouping_gap=config.grouping_gap_seconds,
            margin_before=config.candidate_margin_before_seconds,
            margin_after=config.candidate_margin_after_seconds,
        )
        windows = tuple(region.verification_window() for region in regions)

        def transcribe_window(window) -> Transcription:
            path = extract_speech_audio(
                media_path,
                temporary_dir / f"candidate-{window.index:04d}.wav",
                tools.ffmpeg,
                start_time=window.start,
                duration=window.duration,
            )
            return _measure_accurate(accurate, path, observation, clock)

        def transcribe_full() -> Transcription:
            return _measure_accurate(accurate, full_audio, observation, clock)

        fallback_reason = backend_error
        if fallback_reason is None and not windows:
            fallback_reason = "KWS found no anchor detections"
        search = verify_candidate_windows(
            case.target,
            windows,
            transcribe_window,
            transcribe_full,
            fuzzy_threshold=case.fuzzy_threshold,
            audio_start_offset=media.audio_start_time or 0.0,
            locator_failure_reason=fallback_reason,
        )
        reference = _comparison_baseline(
            case,
            _number_from(baseline, "detected_timestamp_seconds"),
            _string_from(baseline, "matched_text"),
        )
        agrees = first_occurrence_matches_baseline(
            search.match.start if search.match else None,
            search.match.matched_text if search.match else None,
            reference,
        )
        if search.match is not None and agrees is False and not search.fallback_invoked:
            try:
                full = transcribe_full()
                relative = find_dialogue_candidates(
                    case.target, full.words, case.fuzzy_threshold
                )[0]
                baseline_match = offset_match(relative, media.audio_start_time or 0.0)
            except V0Error:
                baseline_match = None
            search = replace(
                search,
                match=baseline_match,
                verified_candidate_index=None,
                fallback_invoked=True,
                fallback_reason="verified KWS candidate disagreed with baseline first occurrence",
            )
    if search.match is None:
        raise _NoKWSMatch(
            search, anchors, detections, regions, observation, backend_error
        )
    frame = resolve_frame_at_timestamp(
        media_path,
        search.match.start,
        defaults.output_dir / "kws" / case.case_id,
    )
    return KWSLocalization(
        search.match,
        frame,
        search,
        anchors,
        detections,
        regions,
        observation,
        backend_error,
    )


def _measure_accurate(
    transcriber: FasterWhisperTranscriber,
    path: Path,
    observation: KWSObservation,
    clock: Clock,
) -> Transcription:
    duration = _wav_duration(path)
    started = clock()
    try:
        return transcriber(path)
    finally:
        observation.accurate_wall_seconds.append(max(0.0, clock() - started))
        observation.accurate_audio_seconds.append(duration)


def _comparison_baseline(
    case: BenchmarkCase,
    timestamp: float | None,
    text: str | None,
) -> ProductionBaseline | None:
    if timestamp is None:
        return case.production_baseline
    tolerance = (
        case.production_baseline.timestamp_tolerance_seconds
        if case.production_baseline else 0.05
    )
    return ProductionBaseline(timestamp, tolerance, text)


def _number_from(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None or isinstance(record.get(key), bool):
        return None
    try:
        return float(record[key])
    except (KeyError, TypeError, ValueError):
        return None


def _string_from(record: dict[str, Any] | None, key: str) -> str | None:
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
