from __future__ import annotations

from dataclasses import dataclass

from dialogue_locator.models import format_timestamp

from .manifest import ProductionBaseline


@dataclass(frozen=True)
class ASRCostMetrics:
    wall_clock_seconds: float
    expensive_audio_seconds_processed: float
    call_count: int


def seconds_to_hms(seconds: float | None) -> str | None:
    """Use the production timestamp format for human-readable benchmark durations."""
    return format_timestamp(seconds)


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
    *,
    target_verified: bool | None = None,
    earliest_valid_occurrence: bool = True,
) -> bool | None:
    """Compare occurrence identity by chronological verification and time.

    ``matched_text`` remains part of the stable call signature because callers
    report it as a diagnostic, but wording differences do not define occurrence
    identity. Independent ASR runs can spell the same spoken phrase differently.
    """
    if baseline is None:
        return None
    verified = bool(matched_text) if target_verified is None else target_verified
    if not verified or not earliest_valid_occurrence or detected_timestamp is None:
        return False
    return bool(
        abs(detected_timestamp - baseline.dialogue_start_seconds)
        <= baseline.timestamp_tolerance_seconds
    )
