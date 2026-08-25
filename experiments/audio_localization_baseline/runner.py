from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import wave
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

from dialogue_locator import __version__
from dialogue_locator import pipeline
from dialogue_locator.models import MediaInfo, Transcription, V1Result

from .manifest import BenchmarkCase, BenchmarkManifest
from .metrics import calculate_asr_cost_metrics, first_occurrence_matches_baseline


Clock = Callable[[], float]


@dataclass
class RuntimeObservation:
    asr_call_seconds: list[float] = field(default_factory=list)
    asr_audio_seconds: list[float] = field(default_factory=list)
    audio_duration_seconds: float | None = None
    media_info: MediaInfo | None = None
    acquisition_metadata: dict[str, Any] = field(default_factory=dict)


class _MeasuredTranscriber:
    def __init__(
        self,
        transcriber: Callable[[Path], Transcription],
        observation: RuntimeObservation,
        clock: Clock,
    ) -> None:
        self.transcriber = transcriber
        self.observation = observation
        self.clock = clock

    def __call__(self, audio_path: Path) -> Transcription:
        audio_seconds = _wav_duration(audio_path)
        started = self.clock()
        try:
            return self.transcriber(audio_path)
        finally:
            self.observation.asr_call_seconds.append(max(0.0, self.clock() - started))
            self.observation.asr_audio_seconds.append(audio_seconds)


class _UncachedTranscriber:
    """Experiment-only replacement for transcript result caching."""

    def __init__(
        self,
        transcriber: Callable[[Path], Transcription],
        _cache: object,
        _identity: str,
    ) -> None:
        self.transcriber = transcriber
        self.last_cache_hit = False
        self.last_transcription: Transcription | None = None

    def __call__(self, audio_path: Path) -> Transcription:
        self.last_transcription = self.transcriber(audio_path)
        return self.last_transcription


class PeakRSSSampler:
    """Best-effort current-process-tree RSS sampler; unavailable means null output."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any | None = None

    def __enter__(self) -> PeakRSSSampler:
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except (ImportError, OSError):
            return self
        self._sample()
        self._thread = threading.Thread(target=self._run, name="benchmark-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            processes = [self._process, *self._process.children(recursive=True)]
            current = sum(process.memory_info().rss for process in processes if process.is_running())
        except Exception:
            return
        self.peak_bytes = max(self.peak_bytes or 0, current)


def run_benchmark(
    manifest: BenchmarkManifest,
    *,
    manifest_path: Path,
    output_path: Path,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    records = [
        _run_case(case, manifest, clock=clock)
        for case in manifest.cases
    ]
    report = {
        "schema_version": 1,
        "benchmark": "production-run_v1-full-audio-asr-baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dialogue_locator_version": __version__,
        },
        "cases": records,
    }
    _write_json_atomic(output_path, report)
    return report


def _run_case(case: BenchmarkCase, manifest: BenchmarkManifest, *, clock: Clock) -> dict[str, Any]:
    defaults = manifest.defaults
    observation = RuntimeObservation()
    result: V1Result | None = None
    error_reason: str | None = None
    started = clock()
    with PeakRSSSampler() as memory:
        try:
            with _instrument_production_baseline(
                observation,
                clock=clock,
                reuse_transcript_cache=defaults.reuse_transcript_cache,
            ):
                result = pipeline.run_v1(
                    case.url,
                    case.target,
                    defaults.work_dir,
                    defaults.output_dir / case.case_id,
                    defaults.model_cache,
                    model_name=case.model,
                    device=case.device,
                    compute_type=case.compute_type,
                    fuzzy_threshold=case.fuzzy_threshold,
                    language=case.language,
                    cookies_from_browser=defaults.cookies_from_browser,
                    cookie_file=defaults.cookies_file,
                    precision_mode="default",
                )
        except Exception as exc:
            error_reason = f"{type(exc).__name__}: {exc}"
    total_seconds = max(0.0, clock() - started)
    asr_cost = calculate_asr_cost_metrics(
        observation.asr_call_seconds,
        observation.asr_audio_seconds,
    )
    fallback_reason = _fallback_reason(observation, result)
    record: dict[str, Any] = {
        "case_id": case.case_id,
        "url": case.url,
        "target": case.target,
        "status": "ok" if result is not None else "error",
        "total_wall_clock_seconds": total_seconds,
        "asr_wall_clock_seconds": asr_cost.wall_clock_seconds,
        "media_duration_seconds": (
            observation.media_info.duration if observation.media_info is not None else None
        ),
        "audio_duration_seconds": observation.audio_duration_seconds,
        "expensive_asr_audio_seconds_processed": asr_cost.expensive_audio_seconds_processed,
        "asr_call_count": asr_cost.call_count,
        "detected_timestamp_seconds": result.match.start if result is not None else None,
        "matched_text": result.match.matched_text if result is not None else None,
        "match_type": result.match.match_type if result is not None else None,
        "match_score": result.match.score if result is not None else None,
        "first_occurrence_matches_production_baseline": first_occurrence_matches_baseline(
            result.match.start if result is not None else None,
            result.match.matched_text if result is not None else None,
            case.production_baseline,
        ),
        "production_baseline": (
            asdict(case.production_baseline) if case.production_baseline is not None else None
        ),
        "peak_rss_bytes": memory.peak_bytes,
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "error_reason": error_reason,
        "transcript_cache_hit": result.transcript_cache_hit if result is not None else False,
        "media_metadata_cache_hit": (
            result.media_metadata_cache_hit if result is not None else False
        ),
        "media_cache_hit": observation.acquisition_metadata.get("media_cache_hit", False),
        "model": case.model,
        "device": case.device,
        "compute_type": case.compute_type,
        "language": case.language,
    }
    return record


@contextmanager
def _instrument_production_baseline(
    observation: RuntimeObservation,
    *,
    clock: Clock,
    reuse_transcript_cache: bool,
) -> Iterator[None]:
    original_transcriber = pipeline.FasterWhisperTranscriber
    original_extract_audio = pipeline.extract_speech_audio
    original_acquire = pipeline.acquire_media
    original_inspect = pipeline._inspect_media_cached

    def measured_transcriber(*args: Any, **kwargs: Any) -> _MeasuredTranscriber:
        return _MeasuredTranscriber(original_transcriber(*args, **kwargs), observation, clock)

    def measured_extract(*args: Any, **kwargs: Any) -> Path:
        audio_path = original_extract_audio(*args, **kwargs)
        observation.audio_duration_seconds = _wav_duration(audio_path)
        return audio_path

    def measured_acquire(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        media_path, metadata = original_acquire(*args, **kwargs)
        observation.acquisition_metadata = dict(metadata)
        return media_path, metadata

    def measured_inspect(*args: Any, **kwargs: Any) -> tuple[MediaInfo, bool]:
        media_info, cache_hit = original_inspect(*args, **kwargs)
        observation.media_info = media_info
        return media_info, cache_hit

    with ExitStack() as stack:
        stack.enter_context(patch.object(pipeline, "FasterWhisperTranscriber", measured_transcriber))
        stack.enter_context(patch.object(pipeline, "extract_speech_audio", measured_extract))
        stack.enter_context(patch.object(pipeline, "acquire_media", measured_acquire))
        stack.enter_context(patch.object(pipeline, "_inspect_media_cached", measured_inspect))
        if not reuse_transcript_cache:
            stack.enter_context(patch.object(pipeline, "CachedTranscriber", _UncachedTranscriber))
        yield


def _fallback_reason(observation: RuntimeObservation, result: V1Result | None) -> str | None:
    reasons: list[str] = []
    if observation.acquisition_metadata.get("extractor") == "direct-http":
        reasons.append("direct-http acquisition fallback")
    if result is not None and result.precision_fallback_reason:
        reasons.append(result.precision_fallback_reason)
    return "; ".join(reasons) if reasons else None


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        frame_rate = audio.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid WAV frame rate in {path}.")
        return audio.getnframes() / frame_rate


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
