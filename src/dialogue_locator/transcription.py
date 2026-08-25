from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .errors import V0Error
from .models import TranscriptWord, Transcription


DEFAULT_MODEL = "base.en"
DEFAULT_MULTILINGUAL_MODEL = "base"
LOGGER = logging.getLogger(__name__)


def language_base(language: str | None) -> str | None:
    if not language:
        return None
    return language.casefold().replace("_", "-").split("-", 1)[0]


def resolve_model_name(model_name: str, language: str | None) -> str:
    base_language = language_base(language)
    if base_language and base_language != "en" and model_name.endswith(".en"):
        if model_name == DEFAULT_MODEL:
            return DEFAULT_MULTILINGUAL_MODEL
        raise V0Error(
            f"Model '{model_name}' is English-only. Select a multilingual faster-whisper "
            f"model for language '{language}'."
        )
    return model_name


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_cache: Path | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_cache = model_cache
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise V0Error(
                "faster-whisper is not installed. Install project dependencies with "
                "'python -m pip install -e .'."
            ) from exc
        try:
            if self.model_cache is not None:
                self.model_cache.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_cache) if self.model_cache is not None else None,
            )
        except Exception as exc:
            raise V0Error(f"Could not load speech model '{self.model_name}': {exc}") from exc
        return self._model

    def __call__(self, audio_path: Path) -> Transcription:
        model = self._load_model()
        language = "en" if self.model_name.endswith(".en") else self.language
        try:
            segment_iterator, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language=language,
                word_timestamps=True,
            )
            segments = list(segment_iterator)
        except Exception as exc:
            raise V0Error(f"Speech recognition failed with model '{self.model_name}': {exc}") from exc

        words: list[TranscriptWord] = []
        for segment in segments:
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                words.append(
                    TranscriptWord(
                        text=word.word,
                        start=float(word.start),
                        end=float(word.end),
                        probability=_float_or_none(getattr(word, "probability", None)),
                    )
                )
        if not words:
            raise V0Error("Speech recognition produced no timestamped words.")

        return Transcription(
            text="".join(segment.text for segment in segments).strip(),
            words=words,
            language=getattr(info, "language", None),
            language_probability=_float_or_none(getattr(info, "language_probability", None)),
        )


class WhisperXAligner:
    def __init__(
        self,
        model_cache: Path | None,
        device: str,
        language: str | None,
    ) -> None:
        self.model_cache = model_cache
        self.device = "cpu" if device == "auto" else device
        self.language = language
        self._models: dict[str, tuple[Any, Any]] = {}

    def __call__(self, audio_path: Path, transcription: Transcription) -> Transcription:
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX precision mode is not installed. Install the optional "
                "dependency with 'python -m pip install -e .[precision]'."
            ) from exc

        language = language_base(self.language or transcription.language) or "en"
        if language not in self._models:
            if self.model_cache is not None:
                self.model_cache.mkdir(parents=True, exist_ok=True)
            model, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_dir=str(self.model_cache) if self.model_cache is not None else None,
            )
            self._models[language] = model, metadata
        model, metadata = self._models[language]
        audio = whisperx.load_audio(str(audio_path))
        segments = [
            {
                "text": transcription.text,
                "start": transcription.words[0].start,
                "end": transcription.words[-1].end,
            }
        ]
        aligned = whisperx.align(
            segments,
            model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        words = []
        for word in aligned.get("word_segments", []):
            if word.get("start") is None or word.get("end") is None:
                continue
            words.append(
                TranscriptWord(
                    text=str(word.get("word", "")),
                    start=float(word["start"]),
                    end=float(word["end"]),
                    probability=_float_or_none(word.get("score")),
                )
            )
        if not words:
            raise RuntimeError("WhisperX alignment produced no timestamped words.")
        return replace(
            transcription,
            words=words,
            alignment_source="whisperx",
            precision_fallback_reason=None,
        )


class OptionalWhisperXTranscriber:
    def __init__(
        self,
        base_transcriber: Callable[[Path], Transcription],
        aligner: Callable[[Path, Transcription], Transcription],
    ) -> None:
        self.base_transcriber = base_transcriber
        self.aligner = aligner

    def __call__(self, audio_path: Path) -> Transcription:
        transcription = self.base_transcriber(audio_path)
        try:
            return self.aligner(audio_path, transcription)
        except Exception as exc:
            LOGGER.warning("WhisperX precision alignment unavailable; using faster-whisper: %s", exc)
            return replace(
                transcription,
                alignment_source="faster-whisper",
                precision_fallback_reason=str(exc),
            )


def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_MODEL,
    model_cache: Path | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
) -> Transcription:
    return FasterWhisperTranscriber(
        model_name=model_name,
        model_cache=model_cache,
        device=device,
        compute_type=compute_type,
        language=language,
    )(audio_path)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
