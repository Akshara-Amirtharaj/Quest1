from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from dialogue_locator.errors import V0Error
from dialogue_locator.models import TranscriptWord, Transcription
from dialogue_locator.transcription import FasterWhisperTranscriber

from .manifest import ConservativeVADConfig
from .vad_filter import (
    SAMPLE_RATE,
    prepare_vad_audio,
    requires_unfiltered_chunk_fallback,
    restore_vad_word_timestamps,
)


Clock = Callable[[], float]


@dataclass
class VADObservation:
    original_audio_seconds: list[float] = field(default_factory=list)
    speech_audio_seconds: list[float] = field(default_factory=list)
    expensive_asr_audio_seconds: list[float] = field(default_factory=list)
    asr_wall_clock_seconds: list[float] = field(default_factory=list)
    vad_wall_clock_seconds: list[float] = field(default_factory=list)
    chunk_fallback_reasons: list[str] = field(default_factory=list)
    chunks_examined: int = 0
    asr_calls: int = 0


class ConservativeVADTranscriber:
    """Use bundled Silero VAD, restoring filtered word times to each source chunk."""

    def __init__(
        self,
        base: FasterWhisperTranscriber,
        config: ConservativeVADConfig,
        observation: VADObservation,
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.base = base
        self.config = config
        self.observation = observation
        self.clock = clock
        self._unfiltered_cache: dict[Path, Transcription] = {}

    def __call__(self, audio_path: Path) -> Transcription:
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        vad_started = self.clock()
        try:
            speech_regions = get_speech_timestamps(
                audio,
                build_vad_options(self.config),
                sampling_rate=SAMPLE_RATE,
            )
        finally:
            self.observation.vad_wall_clock_seconds.append(
                max(0.0, self.clock() - vad_started)
            )
        prepared = prepare_vad_audio(audio, speech_regions, sampling_rate=SAMPLE_RATE)
        self.observation.chunks_examined += 1
        self.observation.original_audio_seconds.append(prepared.original_audio_seconds)
        self.observation.speech_audio_seconds.append(prepared.speech_audio_seconds)

        if requires_unfiltered_chunk_fallback(
            prepared,
            clear_silence_rms_threshold=self.config.clear_silence_rms_threshold,
            max_removed_fraction_before_fallback=(
                self.config.max_removed_fraction_before_fallback
            ),
        ):
            self.observation.chunk_fallback_reasons.append(
                "VAD removal was uncertain for a non-silent chunk; used unfiltered chunk ASR."
            )
            return self._transcribe_unfiltered(audio_path, audio)

        if prepared.speech_audio_seconds <= 0:
            return Transcription("", [], self.base.language, None)

        filtered = self._transcribe(
            prepared.filtered_audio,
            speech_regions=prepared.speech_regions,
        )
        if filtered.words:
            return filtered

        self.observation.chunk_fallback_reasons.append(
            "Filtered speech produced no timestamped words; used unfiltered chunk ASR."
        )
        return self._transcribe_unfiltered(audio_path, audio)

    def transcribe_unfiltered(self, audio_path: Path) -> Transcription:
        cached = self._unfiltered_cache.get(audio_path)
        if cached is not None:
            return cached
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        return self._transcribe_unfiltered(audio_path, audio)

    def _transcribe_unfiltered(self, audio_path: Path, audio) -> Transcription:
        transcription = self._transcribe(audio, speech_regions=None)
        self._unfiltered_cache[audio_path] = transcription
        return transcription

    def _transcribe(
        self,
        audio,
        *,
        speech_regions: tuple[dict[str, int], ...] | None,
    ) -> Transcription:
        if len(audio) == 0:
            return Transcription("", [], self.base.language, None)
        started = self.clock()
        self.observation.asr_calls += 1
        self.observation.expensive_asr_audio_seconds.append(len(audio) / SAMPLE_RATE)
        try:
            model = self.base._load_model()
            language = "en" if self.base.model_name.endswith(".en") else self.base.language
            segment_iterator, info = model.transcribe(
                audio,
                beam_size=5,
                language=language,
                word_timestamps=True,
                vad_filter=False,
            )
            segments = list(segment_iterator)
        except Exception as exc:
            raise V0Error(
                f"Speech recognition failed with model '{self.base.model_name}': {exc}"
            ) from exc
        finally:
            self.observation.asr_wall_clock_seconds.append(
                max(0.0, self.clock() - started)
            )

        words = [
            TranscriptWord(
                text=word.word,
                start=float(word.start),
                end=float(word.end),
                probability=_float_or_none(getattr(word, "probability", None)),
            )
            for segment in segments
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        if speech_regions is not None:
            words = restore_vad_word_timestamps(
                words,
                speech_regions,
                sampling_rate=SAMPLE_RATE,
            )
        return Transcription(
            text="".join(segment.text for segment in segments).strip(),
            words=words,
            language=getattr(info, "language", None),
            language_probability=_float_or_none(getattr(info, "language_probability", None)),
        )


def _float_or_none(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_vad_options(config: ConservativeVADConfig) -> VadOptions:
    return VadOptions(
        threshold=config.threshold,
        neg_threshold=config.neg_threshold,
        min_speech_duration_ms=config.min_speech_duration_ms,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
    )
