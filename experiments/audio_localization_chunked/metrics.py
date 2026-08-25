from __future__ import annotations


def percentage_asr_audio_avoided(
    processed_audio_seconds: float,
    baseline_audio_seconds: float | None,
) -> float | None:
    if processed_audio_seconds < 0:
        raise ValueError("processed_audio_seconds cannot be negative.")
    if baseline_audio_seconds is None:
        return None
    if baseline_audio_seconds <= 0:
        return None
    return (baseline_audio_seconds - processed_audio_seconds) / baseline_audio_seconds * 100.0


def speedup_ratio(
    baseline_wall_clock_seconds: float | None,
    strategy_wall_clock_seconds: float,
) -> float | None:
    if strategy_wall_clock_seconds <= 0:
        return None
    if baseline_wall_clock_seconds is None:
        return None
    if baseline_wall_clock_seconds < 0:
        raise ValueError("baseline_wall_clock_seconds cannot be negative.")
    return baseline_wall_clock_seconds / strategy_wall_clock_seconds


def timestamp_delta(
    detected_timestamp_seconds: float | None,
    baseline_timestamp_seconds: float | None,
) -> float | None:
    if detected_timestamp_seconds is None or baseline_timestamp_seconds is None:
        return None
    return detected_timestamp_seconds - baseline_timestamp_seconds
