from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from .errors import V0Error
from .matching import best_dialogue_match, find_dialogue
from .models import Transcription


LOGGER = logging.getLogger(__name__)
DEFAULT_PRECISION_MODEL = "distil-large-v3"
# Validation placed the known recoverable Barley miss at 78.05 and clearly
# unrelated windows at or below the low 40s. Keep a large recall-first margin.
DEFAULT_PRECISION_TRIGGER_THRESHOLD = 45.0
Transcriber = Callable[[Path], Transcription]
TranscriberFactory = Callable[[], Transcriber]


class PrecisionFallbackTranscriber:
    """Try the default ASR first and lazily invoke precision ASR only after rejection."""

    def __init__(
        self,
        *,
        query: str,
        fuzzy_threshold: float,
        precision_trigger_threshold: float = DEFAULT_PRECISION_TRIGGER_THRESHOLD,
        base_transcriber: Transcriber,
        base_model_name: str,
        precision_transcriber_factory: TranscriberFactory,
        precision_model_name: str = DEFAULT_PRECISION_MODEL,
        scope: str,
    ) -> None:
        if scope not in {"candidate_window", "full_audio"}:
            raise ValueError("precision fallback scope must be candidate_window or full_audio")
        if not 0 <= precision_trigger_threshold <= fuzzy_threshold <= 100:
            raise ValueError(
                "precision_trigger_threshold must be between 0 and the match threshold"
            )
        self.query = query
        self.fuzzy_threshold = fuzzy_threshold
        self.precision_trigger_threshold = precision_trigger_threshold
        self.base_transcriber = base_transcriber
        self.base_model_name = base_model_name
        self.precision_transcriber_factory = precision_transcriber_factory
        self.precision_model_name = precision_model_name
        self.scope = scope
        self._precision_transcriber: Transcriber | None = None
        self.last_transcription: Transcription | None = None
        self.last_cache_hit = False
        self.asr_model_used = base_model_name
        self.precision_fallback_used = False
        self.precision_scope: str | None = None
        self.base_match_score: float | None = None
        self.precision_match_score: float | None = None
        self.precision_fallback_eligible: bool | None = None
        self.precision_fallback_skip_reason: str | None = None
        self.precision_fallback_reason: str | None = None
        self.last_asr_call_count = 0

    @property
    def precision_loaded(self) -> bool:
        return self._precision_transcriber is not None

    def __call__(self, audio_path: Path) -> Transcription:
        self._reset_call_state()
        base_error: V0Error | None = None
        base_transcription: Transcription | None = None
        self.last_asr_call_count = 1
        try:
            base_transcription = self.base_transcriber(audio_path)
            self.base_match_score = _best_score(
                self.query,
                base_transcription,
            )
            find_dialogue(
                self.query,
                base_transcription.words,
                self.fuzzy_threshold,
            )
        except V0Error as exc:
            base_error = exc
        else:
            self.precision_fallback_eligible = False
            self._select(base_transcription, self.base_transcriber, self.base_model_name)
            return base_transcription

        if self.scope == "candidate_window" and not precision_fallback_eligible(
            self.base_match_score,
            self.precision_trigger_threshold,
            self.fuzzy_threshold,
        ):
            self.precision_fallback_eligible = False
            self.precision_fallback_skip_reason = "base_match_score_below_precision_trigger"
            LOGGER.info(
                "Precision ASR skipped for this candidate: base score %.2f is below %.2f.",
                self.base_match_score,
                self.precision_trigger_threshold,
            )
            if base_transcription is not None:
                self._select(base_transcription, self.base_transcriber, self.base_model_name)
                return base_transcription
            if base_error is not None:
                raise base_error

        self.precision_fallback_eligible = True
        self.precision_fallback_used = True
        self.precision_scope = self.scope
        try:
            precision_transcriber = self._get_precision_transcriber()
            self.last_asr_call_count += 1
            precision_transcription = precision_transcriber(audio_path)
            self.precision_match_score = _best_score(
                self.query,
                precision_transcription,
            )
        except Exception as exc:
            self.precision_fallback_reason = str(exc)
            LOGGER.warning(
                "Precision ASR fallback unavailable; preserving base ASR behavior: %s",
                exc,
            )
            if base_transcription is not None:
                self._select(base_transcription, self.base_transcriber, self.base_model_name)
                self.precision_fallback_used = True
                self.precision_scope = self.scope
                return base_transcription
            if base_error is not None:
                raise base_error
            raise

        self._select(
            precision_transcription,
            precision_transcriber,
            self.precision_model_name,
        )
        self.precision_fallback_used = True
        self.precision_scope = self.scope
        return precision_transcription

    def _get_precision_transcriber(self) -> Transcriber:
        if self._precision_transcriber is None:
            self._precision_transcriber = self.precision_transcriber_factory()
        return self._precision_transcriber

    def _select(
        self,
        transcription: Transcription,
        transcriber: Transcriber,
        model_name: str,
    ) -> None:
        self.last_transcription = transcription
        self.last_cache_hit = bool(getattr(transcriber, "last_cache_hit", False))
        self.asr_model_used = model_name

    def _reset_call_state(self) -> None:
        self.last_transcription = None
        self.last_cache_hit = False
        self.asr_model_used = self.base_model_name
        self.precision_fallback_used = False
        self.precision_scope = None
        self.base_match_score = None
        self.precision_match_score = None
        self.precision_fallback_eligible = None
        self.precision_fallback_skip_reason = None
        self.precision_fallback_reason = None
        self.last_asr_call_count = 0


def _best_score(query: str, transcription: Transcription) -> float | None:
    try:
        return best_dialogue_match(query, transcription.words).score
    except V0Error:
        return None


def precision_fallback_eligible(
    base_match_score: float | None,
    precision_trigger_threshold: float,
    match_threshold: float,
) -> bool:
    """Return whether an uncertain base result merits bounded precision ASR."""
    if base_match_score is None:
        return True
    return precision_trigger_threshold <= base_match_score < match_threshold
