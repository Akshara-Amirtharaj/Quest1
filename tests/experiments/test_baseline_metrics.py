from __future__ import annotations

import pytest

from experiments.audio_localization_baseline.manifest import ProductionBaseline
from experiments.audio_localization_baseline.metrics import (
    calculate_asr_cost_metrics,
    first_occurrence_matches_baseline,
    seconds_to_hms,
)


def test_asr_cost_metrics_sum_only_expensive_transcriber_calls() -> None:
    metrics = calculate_asr_cost_metrics([1.25, 0.75], [10.0, 4.5])

    assert metrics.wall_clock_seconds == 2.0
    assert metrics.expensive_audio_seconds_processed == 14.5
    assert metrics.call_count == 2


def test_warm_transcript_cache_has_no_expensive_asr_work() -> None:
    metrics = calculate_asr_cost_metrics([], [])

    assert metrics.wall_clock_seconds == 0.0
    assert metrics.expensive_audio_seconds_processed == 0.0
    assert metrics.call_count == 0


def test_benchmark_seconds_use_production_hms_format() -> None:
    assert seconds_to_hms(7.4060727) == "00:00:07.406"
    assert seconds_to_hms(3661.005) == "01:01:01.005"
    assert seconds_to_hms(None) is None


def test_asr_cost_metrics_reject_misaligned_or_negative_observations() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_asr_cost_metrics([1.0], [])
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_asr_cost_metrics([-1.0], [10.0])


def test_first_occurrence_comparison_uses_verification_and_timestamp_tolerance() -> None:
    baseline = ProductionBaseline(
        dialogue_start_seconds=2.16,
        timestamp_tolerance_seconds=0.1,
        matched_text="Captions, like this!",
    )

    assert first_occurrence_matches_baseline(2.20, "captions like this", baseline) is True
    assert first_occurrence_matches_baseline(2.40, "captions like this", baseline) is False
    assert first_occurrence_matches_baseline(2.16, "different words", baseline) is True
    assert (
        first_occurrence_matches_baseline(
            2.16,
            "captions like this",
            baseline,
            target_verified=False,
        )
        is False
    )
    assert (
        first_occurrence_matches_baseline(
            2.16,
            "captions like this",
            baseline,
            earliest_valid_occurrence=False,
        )
        is False
    )
    assert first_occurrence_matches_baseline(None, None, baseline) is False
    assert first_occurrence_matches_baseline(2.16, "captions like this", None) is None


def test_first_occurrence_ignores_asr_spelling_variation() -> None:
    baseline = ProductionBaseline(
        dialogue_start_seconds=2122.22,
        timestamp_tolerance_seconds=0.1,
        matched_text="Don't worry that is not exclusively a Taruvian custom.",
    )

    assert first_occurrence_matches_baseline(
        2122.22,
        "Don't worry that is not exclusively a turovian custom.",
        baseline,
    ) is True
