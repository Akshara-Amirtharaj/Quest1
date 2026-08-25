from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from dialogue_locator.errors import V0Error
from dialogue_locator.models import TranscriptWord, Transcription
from dialogue_locator.transcription import FasterWhisperTranscriber

from .chunking import AudioChunk


SAMPLE_RATE = 16_000


class ReusableWaveAudio:
    """Read arbitrary chronological slices from one extracted PCM WAV."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._audio = wave.open(str(path), "rb")
        if (
            self._audio.getnchannels() != 1
            or self._audio.getsampwidth() != 2
            or self._audio.getframerate() != SAMPLE_RATE
        ):
            self._audio.close()
            raise V0Error("Reusable chunk audio must be mono 16 kHz 16-bit PCM WAV.")
        self.total_frames = self._audio.getnframes()

    @property
    def duration(self) -> float:
        return self.total_frames / SAMPLE_RATE

    def close(self) -> None:
        self._audio.close()

    def __enter__(self) -> ReusableWaveAudio:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self, chunk: AudioChunk) -> np.ndarray:
        start_frame = min(self.total_frames, max(0, round(chunk.start * SAMPLE_RATE)))
        end_frame = min(self.total_frames, max(start_frame, round(chunk.end * SAMPLE_RATE)))
        self._audio.setpos(start_frame)
        raw = self._audio.readframes(end_frame - start_frame)
        if not raw:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class ReusableChunkTranscriber:
    """Reuse one faster-whisper model and one decoded-audio source for all chunks."""

    def __init__(self, base: FasterWhisperTranscriber, source: ReusableWaveAudio) -> None:
        self.base = base
        self.source = source

    def __call__(self, chunk: AudioChunk) -> Transcription:
        audio = self.source.read(chunk)
        if len(audio) == 0:
            return Transcription("", [], self.base.language, None)
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
        return Transcription(
            text="".join(segment.text for segment in segments).strip(),
            words=words,
            language=getattr(info, "language", None),
            language_probability=_float_or_none(getattr(info, "language_probability", None)),
        )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
