from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_frozen_matrix.evaluation import (
    build_comparison,
    evaluate_strategy_record,
    write_outputs,
)
from experiments.audio_localization_frozen_matrix.fixtures import derive_boundary_onset
from experiments.audio_localization_frozen_matrix.manifest import (
    FROZEN_CHUNK_DURATION_SECONDS,
    FROZEN_OVERLAP_SECONDS,
    FROZEN_TRANSCRIPT_CONTEXT_SECONDS,
    load_frozen_manifest,
    parse_timestamp,
)


MANIFEST = Path("experiments/audio_localization_frozen_matrix/manifest.json")


def test_timestamp_parser_preserves_one_hour_thirteen_minutes() -> None:
    assert parse_timestamp("01:13:00.000") == 4380.0
    assert parse_timestamp("00:00:23.000") == 23.0
    with pytest.raises(ManifestError):
        parse_timestamp("00:01:13:00.000")


def test_manifest_inventory_applicability_and_media_deduplication() -> None:
    manifest = load_frozen_manifest(MANIFEST)

    assert len(manifest.cases) == 12
    assert len(manifest.audio_cases) == 11
    assert sum(case.source_kind == "public" for case in manifest.cases) == 8
    assert sum(case.source_kind == "controlled" for case in manifest.cases) == 4
    assert manifest.unique_media_count == 8
    assert [case.case_id for case in manifest.cases if not case.is_audio_case] == [
        "big_buck_bunny_visible_title"
    ]
    cs50 = next(case for case in manifest.cases if case.case_id == "cs50_long_middle")
    assert cs50.expected_timestamp_seconds == 4380.0


def test_frozen_chunk_configuration_is_centralized_and_immutable() -> None:
    manifest = load_frozen_manifest(MANIFEST)

    assert manifest.chunked_asr.chunk_duration_seconds == FROZEN_CHUNK_DURATION_SECONDS == 120
    assert manifest.chunked_asr.overlap_seconds == FROZEN_OVERLAP_SECONDS == 5
    assert (
        manifest.chunked_asr.transcript_context_seconds
        == FROZEN_TRANSCRIPT_CONTEXT_SECONDS
        == 15
    )


def test_boundary_fixture_placement_is_derived_from_frozen_boundary() -> None:
    onset = derive_boundary_onset()

    assert onset == FROZEN_CHUNK_DURATION_SECONDS - 1.5
    assert onset < FROZEN_CHUNK_DURATION_SECONDS


def test_absent_target_evaluation_has_no_fabricated_delta() -> None:
    manifest = load_frozen_manifest(MANIFEST)
    case = next(
        case
        for case in manifest.cases
        if case.case_id == "fixture_no_captions_absent_long_silence"
    )
    evaluated = evaluate_strategy_record(
        case,
        "full_asr",
        {
            "status": "error",
            "detected_timestamp_seconds": None,
            "error_reason": "V0Error: Dialogue not found in the spoken-audio transcription",
            "total_wall_clock_seconds": 1.0,
        },
    )

    assert evaluated["result"] == "NO_MATCH"
    assert evaluated["correct_first_occurrence"] is True
    assert evaluated["absolute_timestamp_delta_seconds"] is None
    assert evaluated["false_positive"] is False


def test_repeated_target_must_match_expected_first_timestamp() -> None:
    manifest = load_frozen_manifest(MANIFEST)
    case = next(case for case in manifest.cases if case.case_id == "fixture_repeated_target")

    first = evaluate_strategy_record(
        case,
        "chunked_asr",
        {"detected_timestamp_seconds": 10.2, "status": "ok"},
    )
    second = evaluate_strategy_record(
        case,
        "chunked_asr",
        {"detected_timestamp_seconds": 45.0, "status": "ok"},
    )

    assert first["correct_first_occurrence"] is True
    assert second["correct_first_occurrence"] is False
    assert second["within_tolerance"] is False


def test_ocr_case_is_excluded_from_audio_aggregates_and_outputs_serialize(
    tmp_path: Path,
) -> None:
    manifest = load_frozen_manifest(MANIFEST)
    baseline_cases = []
    chunked_cases = []
    for case in manifest.audio_cases:
        timestamp = case.expected_timestamp_seconds
        error = (
            "V0Error: Dialogue not found in the spoken-audio transcription"
            if case.expected_result == "NO_MATCH"
            else None
        )
        common = {
            "case_id": case.case_id,
            "status": "ok" if timestamp is not None else "error",
            "detected_timestamp_seconds": timestamp,
            "media_duration_seconds": 100.0,
            "expensive_asr_audio_seconds_processed": 100.0,
            "total_wall_clock_seconds": 10.0,
            "error_reason": error,
        }
        baseline_cases.append(common)
        chunked_cases.append(common | {"total_wall_clock_seconds": 5.0})
    summary = build_comparison(
        manifest,
        {"cases": baseline_cases},
        {"cases": chunked_cases},
    )
    write_outputs(summary, tmp_path)

    assert summary["inventory"]["audio_comparison_cases"] == 11
    assert summary["excluded_from_audio_aggregates"] == [
        "big_buck_bunny_visible_title"
    ]
    assert summary["aggregate"]["median_speedup_ratio"] == 2.0
    assert json.loads((tmp_path / "comparison.json").read_text())["comparisons"]
    assert (tmp_path / "comparison.csv").is_file()
    assert "big_buck_bunny_visible_title" not in (tmp_path / "summary.md").read_text()
