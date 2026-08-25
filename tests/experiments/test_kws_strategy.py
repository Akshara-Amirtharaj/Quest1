from __future__ import annotations

import pytest

from dialogue_locator.models import TranscriptWord, Transcription
from experiments.audio_localization_kws.anchors import generate_phrase_anchors
from experiments.audio_localization_kws.candidates import AnchorDetection, group_detections
from experiments.audio_localization_kws.sherpa_backend import _parse_detections
from experiments.audio_localization_lightweight_locator.localization import (
    verify_candidate_windows,
)


def _transcription(*values: tuple[str, float, float]) -> Transcription:
    words = [TranscriptWord(text, start, end, 0.9) for text, start, end in values]
    return Transcription(" ".join(word.text for word in words), words, "en", 1.0)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("quantum lattice failure", "quantum lattice"),
        ("thank you very much", "thank you"),
        ("twenty million dollars", "twenty million"),
        ("Sherlock Holmes arrived", "sherlock holmes"),
        ("Help", "help"),
        ("the company reported revenue of twenty million dollars", "million dollars"),
    ],
    ids=["distinctive", "common", "numbers", "proper-noun", "short", "long"],
)
def test_anchor_generation_for_target_types(target: str, expected: str) -> None:
    anchors = generate_phrase_anchors(target)

    assert expected in anchors
    assert 1 <= len(anchors) <= 3
    assert anchors == generate_phrase_anchors(target)


def test_kws_can_miss_one_anchor_and_detect_another() -> None:
    regions = group_detections(
        [AnchorDetection("million dollars", 10.0, 10.5)],
        audio_duration=30,
        grouping_gap=2,
        margin_before=2.5,
        margin_after=2,
    )

    assert len(regions) == 1
    assert regions[0].detections[0].anchor == "million dollars"


def test_multiple_detection_groups_create_chronological_candidates() -> None:
    regions = group_detections(
        [
            AnchorDetection("second", 20, 20.4),
            AnchorDetection("first b", 3, 3.2),
            AnchorDetection("first a", 2, 2.2),
        ],
        audio_duration=40,
        grouping_gap=1,
        margin_before=1,
        margin_after=1,
    )

    assert len(regions) == 2
    assert [(item.start, item.end) for item in regions] == [(1, 4.2), (19, 21.4)]


def test_first_kws_candidate_can_fail_before_second_verifies() -> None:
    regions = group_detections(
        [AnchorDetection("target", 2, 2.2), AnchorDetection("target", 20, 20.2)],
        audio_duration=30,
        grouping_gap=1,
        margin_before=1,
        margin_after=2,
    )
    windows = tuple(region.verification_window() for region in regions)
    calls: list[int] = []

    def transcribe(window):
        calls.append(window.index)
        if window.index == 0:
            return _transcription(("wrong", 0.5, 0.9))
        return _transcription(("target", 1.0, 1.3), ("phrase", 1.4, 1.8))

    result = verify_candidate_windows(
        "target phrase",
        windows,
        transcribe,
        lambda: _transcription(("unused", 0, 1)),
        fuzzy_threshold=85,
    )

    assert calls == [0, 1]
    assert result.match is not None
    assert result.match.start == 20
    assert result.verified_candidate_index == 1
    assert not result.fallback_invoked


def test_target_absent_falls_back_and_returns_no_match() -> None:
    result = verify_candidate_windows(
        "target phrase",
        (),
        lambda _: _transcription(("unused", 0, 1)),
        lambda: _transcription(("nothing", 5, 5.5), ("here", 5.6, 6)),
        fuzzy_threshold=85,
        locator_failure_reason="KWS found no anchor detections",
    )

    assert result.fallback_invoked
    assert result.match is None
    assert result.fallback_reason == "KWS found no anchor detections"


def test_target_near_beginning_and_end_are_clamped() -> None:
    regions = group_detections(
        [AnchorDetection("early", 0.2, 0.4), AnchorDetection("late", 29.6, 29.9)],
        audio_duration=30,
        grouping_gap=1,
        margin_before=2,
        margin_after=2,
    )

    assert regions[0].start == 0
    assert regions[-1].end == 30


def test_native_sherpa_json_preserves_anchor_timestamp() -> None:
    detections = _parse_detections(
        '{"start_time":0.0,"keyword":"LIKE THIS","timestamps":[2.92,3.04]}',
        {"like this": "like this"},
    )

    assert detections == [AnchorDetection("like this", 2.92, 3.04)]
