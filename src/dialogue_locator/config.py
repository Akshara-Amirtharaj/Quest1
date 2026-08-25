from __future__ import annotations

from dataclasses import dataclass

from .precision import DEFAULT_PRECISION_TRIGGER_THRESHOLD


@dataclass(frozen=True)
class V2Config:
    caption_fuzzy_threshold: float = 85.0
    verification_fuzzy_threshold: float = 85.0
    subtitle_window_size: int = 4
    verification_margins: tuple[float, ...] = (2.0, 5.0)
    asr_precision_fallback: bool = True
    precision_asr_model: str = "distil-large-v3"
    precision_trigger_threshold: float = DEFAULT_PRECISION_TRIGGER_THRESHOLD
    full_audio_precision_fallback: bool = False

    def __post_init__(self) -> None:
        if self.subtitle_window_size < 1:
            raise ValueError("subtitle_window_size must be at least 1")
        if not 0 <= self.caption_fuzzy_threshold <= 100:
            raise ValueError("caption_fuzzy_threshold must be between 0 and 100")
        if not 0 <= self.verification_fuzzy_threshold <= 100:
            raise ValueError("verification_fuzzy_threshold must be between 0 and 100")
        if not self.verification_margins or any(margin < 0 for margin in self.verification_margins):
            raise ValueError("verification_margins must contain non-negative values")
        if tuple(sorted(set(self.verification_margins))) != self.verification_margins:
            raise ValueError("verification_margins must be unique and increasing")
        if not self.precision_asr_model.strip():
            raise ValueError("precision_asr_model cannot be empty")
        if not 0 <= self.precision_trigger_threshold <= 100:
            raise ValueError("precision_trigger_threshold must be between 0 and 100")
        if self.precision_trigger_threshold > self.verification_fuzzy_threshold:
            raise ValueError(
                "precision_trigger_threshold must be less than or equal to "
                "verification_fuzzy_threshold"
            )
        if self.full_audio_precision_fallback and not self.asr_precision_fallback:
            raise ValueError(
                "full_audio_precision_fallback requires asr_precision_fallback"
            )


@dataclass(frozen=True)
class V3Config:
    search_margin: float = 1.0
    fuzzy_threshold: float = 85.0

    def __post_init__(self) -> None:
        if self.search_margin < 0:
            raise ValueError("search_margin must be non-negative")
        if not 0 <= self.fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be between 0 and 100")
