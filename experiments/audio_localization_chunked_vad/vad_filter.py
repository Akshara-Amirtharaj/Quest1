from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from faster_whisper.vad import SpeechTimestampsMap

from dialogue_locator.models import TranscriptWord


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VADAudio:
    filtered_audio: np.ndarray
    speech_regions: tuple[dict[str, int], ...]
    original_audio_seconds: float
    speech_audio_seconds: float
    removed_fraction: float
    rms: float


def prepare_vad_audio(
    audio: np.ndarray,
    speech_regions: Iterable[dict[str, Any]],
    *,
    sampling_rate: int = SAMPLE_RATE,
) -> VADAudio:
    if audio.ndim != 1:
        raise ValueError("VAD audio must be a one-dimensional waveform.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    regions = _validated_regions(speech_regions, len(audio))
    filtered = (
        np.concatenate([audio[region["start"] : region["end"]] for region in regions])
        if regions
        else np.array([], dtype=np.float32)
    )
    original_seconds = len(audio) / sampling_rate
    speech_seconds = len(filtered) / sampling_rate
    removed_fraction = (
        max(0.0, min(1.0, 1.0 - speech_seconds / original_seconds))
        if original_seconds > 0
        else 0.0
    )
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    return VADAudio(
        filtered_audio=filtered,
        speech_regions=regions,
        original_audio_seconds=original_seconds,
        speech_audio_seconds=speech_seconds,
        removed_fraction=removed_fraction,
        rms=rms,
    )


def restore_vad_word_timestamps(
    words: Iterable[TranscriptWord],
    speech_regions: Iterable[dict[str, int]],
    *,
    sampling_rate: int = SAMPLE_RATE,
) -> list[TranscriptWord]:
    regions = tuple(speech_regions)
    word_list = list(words)
    if not word_list or not regions:
        return word_list
    timestamp_map = SpeechTimestampsMap(list(regions), sampling_rate, time_precision=3)
    restored: list[TranscriptWord] = []
    for word in word_list:
        middle = (word.start + word.end) / 2
        region_index = timestamp_map.get_chunk_index(middle)
        restored.append(
            TranscriptWord(
                text=word.text,
                start=timestamp_map.get_original_time(word.start, region_index),
                end=timestamp_map.get_original_time(word.end, region_index),
                probability=word.probability,
            )
        )
    return restored


def requires_unfiltered_chunk_fallback(
    vad_audio: VADAudio,
    *,
    clear_silence_rms_threshold: float,
    max_removed_fraction_before_fallback: float,
) -> bool:
    """Treat aggressive removal of nontrivial audio as uncertain, not silence."""
    if vad_audio.rms <= clear_silence_rms_threshold:
        return False
    return (
        vad_audio.speech_audio_seconds <= 0
        or vad_audio.removed_fraction > max_removed_fraction_before_fallback
    )


def _validated_regions(
    speech_regions: Iterable[dict[str, Any]],
    audio_samples: int,
) -> tuple[dict[str, int], ...]:
    validated: list[dict[str, int]] = []
    previous_end = 0
    for index, raw in enumerate(speech_regions):
        try:
            start = int(raw["start"])
            end = int(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid VAD speech region at index {index}.") from exc
        if start < previous_end or end <= start or end > audio_samples:
            raise ValueError(f"Invalid or overlapping VAD speech region at index {index}.")
        validated.append({"start": start, "end": end})
        previous_end = end
    return tuple(validated)
