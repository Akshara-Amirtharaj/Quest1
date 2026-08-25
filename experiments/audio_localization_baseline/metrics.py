from __future__ import annotations

from dataclasses import dataclass

from dialogue_locator.matching import normalize_text

from .manifest import ProductionBaseline


@dataclass(frozen=True)
class ASRCostMetrics:
    wall_clock_seconds: float
    expensive_audio_seconds_processed: float
    call_count: int


def calculate_asr_cost_metrics(
    call_wall_clock_seconds: list[float],
    call_audio_seconds: list[float],
) -> ASRCostMetrics:
    if len(call_wall_clock_seconds) != len(call_audio_seconds):
        raise ValueError("ASR timing and audio-duration lists must have equal lengths.")
    if any(value < 0 for value in call_wall_clock_seconds + call_audio_seconds):
        raise ValueError("ASR measurements cannot be negative.")
    return ASRCostMetrics(
        wall_clock_seconds=sum(call_wall_clock_seconds),
        expensive_audio_seconds_processed=sum(call_audio_seconds),
        call_count=len(call_wall_clock_seconds),
    )


def first_occurrence_matches_baseline(
    detected_timestamp: float | None,
    matched_text: str | None,
    baseline: ProductionBaseline | None,
) -> bool | None:
    if baseline is None:
        return None
    if detected_timestamp is None:
        return False
    if abs(detected_timestamp - baseline.dialogue_start_seconds) > baseline.timestamp_tolerance_seconds:
        return False
    if baseline.matched_text is None:
        return True
    return normalize_text(matched_text or "") == normalize_text(baseline.matched_text)
