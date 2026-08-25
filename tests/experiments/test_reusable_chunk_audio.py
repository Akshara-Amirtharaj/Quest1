from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.audio_localization_chunked.chunking import AudioChunk
from experiments.audio_localization_chunked.reusable_audio import (
    ReusableChunkTranscriber,
    ReusableWaveAudio,
    SAMPLE_RATE,
)


def _write_wave(path: Path, seconds: float = 2.0) -> Path:
    samples = np.arange(round(seconds * SAMPLE_RATE), dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())
    return path


def test_reusable_wave_reads_multiple_slices_from_one_open_source(tmp_path: Path) -> None:
    path = _write_wave(tmp_path / "audio.wav")

    with ReusableWaveAudio(path) as source:
        first = source.read(AudioChunk(0, 0.0, 0.5))
        second = source.read(AudioChunk(1, 1.0, 1.25))

    assert len(first) == SAMPLE_RATE // 2
    assert len(second) == SAMPLE_RATE // 4
    assert first.dtype == np.float32


def test_chunk_transcriber_reuses_loaded_model_across_slices(tmp_path: Path) -> None:
    path = _write_wave(tmp_path / "audio.wav")

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio: np.ndarray, **_kwargs: object):
            self.calls += 1
            if not np.any(audio):
                return iter(()), SimpleNamespace(language="en", language_probability=1.0)
            word = SimpleNamespace(word=" target", start=0.1, end=0.4, probability=0.9)
            segment = SimpleNamespace(text=" target", words=[word])
            return iter((segment,)), SimpleNamespace(language="en", language_probability=1.0)

    class FakeBase:
        model_name = "base.en"
        language = "en"

        def __init__(self) -> None:
            self.model = FakeModel()
            self.loads = 0

        def _load_model(self) -> FakeModel:
            if self.loads == 0:
                self.loads += 1
            return self.model

    base = FakeBase()
    with ReusableWaveAudio(path) as source:
        transcribe = ReusableChunkTranscriber(base, source)  # type: ignore[arg-type]
        first = transcribe(AudioChunk(0, 0.1, 0.5))
        second = transcribe(AudioChunk(1, 0.5, 0.9))

    assert first.words and second.words
    assert base.loads == 1
    assert base.model.calls == 2
