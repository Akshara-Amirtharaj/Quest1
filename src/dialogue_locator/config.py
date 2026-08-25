from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V2Config:
    caption_fuzzy_threshold: float = 85.0
    verification_fuzzy_threshold: float = 85.0
    subtitle_window_size: int = 4
    verification_margins: tuple[float, ...] = (2.0, 5.0)

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


@dataclass(frozen=True)
class V3Config:
    search_margin: float = 1.0
    fuzzy_threshold: float = 85.0

    def __post_init__(self) -> None:
        if self.search_margin < 0:
            raise ValueError("search_margin must be non-negative")
        if not 0 <= self.fuzzy_threshold <= 100:
            raise ValueError("fuzzy_threshold must be between 0 and 100")
