from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio import extract_speech_audio
from .config import V2Config
from .errors import V0Error
from .matching import find_dialogue
from .models import CaptionCandidate, DialogueMatch, Transcription


LOGGER = logging.getLogger(__name__)
Transcriber = Callable[[Path], Transcription]
AudioExtractor = Callable[..., Path]


@dataclass(frozen=True)
class CaptionVerification:
    match: DialogueMatch
    audio_processed_seconds: float
    candidate: CaptionCandidate


def verify_caption_candidates(
    candidates: list[CaptionCandidate],
    query: str,
    media_path: Path,
    media_duration: float | None,
    audio_start_time: float,
    temporary_dir: Path,
    ffmpeg: str,
    transcriber: Transcriber,
    config: V2Config,
    audio_extractor: AudioExtractor = extract_speech_audio,
) -> tuple[CaptionVerification | None, float]:
    processed_seconds = 0.0
    attempted_windows: set[tuple[int, int]] = set()
    audio_path = temporary_dir / "verification.wav"

    for candidate in candidates:
        for margin in config.verification_margins:
            window_start = max(0.0, candidate.start - margin)
            window_end = candidate.end + margin
            if media_duration is not None:
                window_end = min(media_duration, window_end)
            if window_end <= window_start:
                continue
            window_key = (round(window_start * 1000), round(window_end * 1000))
            if window_key in attempted_windows:
                continue
            attempted_windows.add(window_key)
            window_duration = window_end - window_start
            processed_seconds += window_duration
            try:
                audio_extractor(
                    media_path,
                    audio_path,
                    ffmpeg,
                    start_time=window_start,
                    duration=window_duration,
                )
                try:
                    transcription = transcriber(audio_path)
                finally:
                    asr_call_count = max(
                        1,
                        int(getattr(transcriber, "last_asr_call_count", 1)),
                    )
                    processed_seconds += window_duration * (asr_call_count - 1)
                relative_match = find_dialogue(
                    query,
                    transcription.words,
                    config.verification_fuzzy_threshold,
                )
            except V0Error as exc:
                LOGGER.info(
                    "Caption candidate %.3f-%.3f failed ASR verification with margin %.3f: %s",
                    candidate.start,
                    candidate.end,
                    margin,
                    exc,
                )
                continue

            absolute_offset = window_start + audio_start_time
            return (
                CaptionVerification(
                    match=DialogueMatch(
                        matched_text=relative_match.matched_text,
                        start=relative_match.start + absolute_offset,
                        end=relative_match.end + absolute_offset,
                        match_type=relative_match.match_type,
                        score=relative_match.score,
                    ),
                    audio_processed_seconds=processed_seconds,
                    candidate=candidate,
                ),
                processed_seconds,
            )
    return None, processed_seconds
