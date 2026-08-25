from __future__ import annotations

import pytest

from dialogue_locator.models import TranscriptWord, Transcription
from experiments.audio_localization_lightweight_locator.localization import (
    generate_candidate_windows,
    verify_candidate_windows,
)


def _words(*values: tuple[str, float, float]) -> list[TranscriptWord]:
    return [TranscriptWord(text, start, end, 0.9) for text, start, end in values]


def _transcription(*values: tuple[str, float, float]) -> Transcription:
    words = _words(*values)
    return Transcription(" ".join(item.text for item in words), words, "en", 1.0)


def _candidates(words: list[TranscriptWord], *, duration: float = 60.0):
    return generate_candidate_windows(
        "target phrase",
        words,
        audio_duration=duration,
        margin_before=1.0,
        margin_after=2.0,
        fuzzy_threshold=85.0,
    )


def test_candidates_are_ordered_chronologically() -> None:
    candidates = _candidates(
        _words(
            ("target", 20.0, 20.4),
            ("phrase", 20.5, 20.9),
            ("target", 3.0, 3.4),
            ("phrase", 3.5, 3.9),
        )
    )

    assert [candidate.locator_match.start for candidate in candidates] == [3.0, 20.0]
    assert [candidate.index for candidate in candidates] == [0, 1]


def test_candidate_margins_expand_and_clamp_to_audio() -> None:
    start = _candidates(_words(("target", 0.2, 0.5), ("phrase", 0.6, 0.9)), duration=10)
    end = _candidates(_words(("target", 8.8, 9.1), ("phrase", 9.2, 9.8)), duration=10)

    assert (start[0].start, start[0].end) == (0.0, 2.9)
    assert end[0].start == pytest.approx(7.8)
    assert end[0].end == 10.0


def test_locator_small_asr_error_produces_fuzzy_candidate() -> None:
    candidates = generate_candidate_windows(
        "captions like this",
        _words(("caption", 2.0, 2.4), ("like", 2.5, 2.8), ("this", 2.9, 3.2)),
        audio_duration=20,
        margin_before=1,
        margin_after=1,
        fuzzy_threshold=85,
    )

    assert len(candidates) == 1
    assert candidates[0].locator_match.match_type == "fuzzy"
    assert candidates[0].locator_match.score >= 85


def test_multiple_locator_candidates_are_preserved() -> None:
    candidates = _candidates(
        _words(
            ("target", 2.0, 2.4),
            ("phrase", 2.5, 2.9),
            ("filler", 10.0, 10.5),
            ("target", 20.0, 20.4),
            ("phrase", 20.5, 20.9),
        )
    )

    assert len(candidates) == 2
    assert [candidate.start for candidate in candidates] == [1.0, 19.0]


def test_earlier_fuzzy_candidate_is_not_hidden_by_later_exact_candidate() -> None:
    candidates = generate_candidate_windows(
        "captions like this",
        _words(
            ("caption", 2.0, 2.4),
            ("like", 2.5, 2.8),
            ("this", 2.9, 3.2),
            ("captions", 20.0, 20.4),
            ("like", 20.5, 20.8),
            ("this", 20.9, 21.2),
        ),
        audio_duration=30,
        margin_before=1,
        margin_after=1,
        fuzzy_threshold=85,
    )

    assert len(candidates) == 2
    assert candidates[0].locator_match.start == 2.0
    assert candidates[0].locator_match.match_type == "fuzzy"
    assert candidates[1].locator_match.match_type == "exact"


def test_first_candidate_can_fail_before_second_verifies() -> None:
    candidates = _candidates(
        _words(
            ("target", 2.0, 2.4),
            ("phrase", 2.5, 2.9),
            ("target", 20.0, 20.4),
            ("phrase", 20.5, 20.9),
        )
    )
    calls: list[int] = []

    def verify(candidate):
        calls.append(candidate.index)
        if candidate.index == 0:
            return _transcription(("different", 1.0, 1.4), ("words", 1.5, 1.9))
        return _transcription(("target", 1.0, 1.4), ("phrase", 1.5, 1.9))

    result = verify_candidate_windows(
        "target phrase",
        candidates,
        verify,
        lambda: _transcription(("unused", 0, 1)),
        fuzzy_threshold=85,
        audio_start_offset=0.5,
    )

    assert calls == [0, 1]
    assert result.match is not None
    assert result.match.start == 20.5
    assert result.verified_candidate_index == 1
    assert not result.fallback_invoked


def test_no_verified_candidate_falls_back_to_full_accurate_asr() -> None:
    candidates = _candidates(_words(("target", 2, 2.4), ("phrase", 2.5, 2.9)))
    fallback_calls = 0

    def full():
        nonlocal fallback_calls
        fallback_calls += 1
        return _transcription(("target", 40, 40.4), ("phrase", 40.5, 40.9))

    result = verify_candidate_windows(
        "target phrase",
        candidates,
        lambda _: _transcription(("wrong", 0.2, 0.5)),
        full,
        fuzzy_threshold=85,
    )

    assert fallback_calls == 1
    assert result.fallback_invoked
    assert result.match is not None
    assert result.match.start == 40


def test_first_occurrence_near_start_is_verified_with_absolute_offset() -> None:
    candidates = _candidates(_words(("target", 0.2, 0.5), ("phrase", 0.6, 0.9)))
    result = verify_candidate_windows(
        "target phrase",
        candidates,
        lambda _: _transcription(("target", 0.2, 0.5), ("phrase", 0.6, 0.9)),
        lambda: _transcription(("unused", 0, 1)),
        fuzzy_threshold=85,
        audio_start_offset=0.25,
    )

    assert result.match is not None
    assert result.match.start == 0.45


def test_first_occurrence_near_end_preserves_window_offset() -> None:
    candidates = _candidates(
        _words(("target", 58.0, 58.4), ("phrase", 58.5, 58.9)),
        duration=60,
    )
    result = verify_candidate_windows(
        "target phrase",
        candidates,
        lambda _: _transcription(("target", 1.0, 1.4), ("phrase", 1.5, 1.9)),
        lambda: _transcription(("unused", 0, 1)),
        fuzzy_threshold=85,
    )

    assert result.match is not None
    assert result.match.start == 58.0


def test_no_match_case_returns_none_after_full_asr_fallback() -> None:
    result = verify_candidate_windows(
        "target phrase",
        (),
        lambda _: _transcription(("unused", 0, 1)),
        lambda: _transcription(("nothing", 5, 5.5), ("relevant", 5.6, 6.0)),
        fuzzy_threshold=85,
    )

    assert result.match is None
    assert result.fallback_invoked
    assert result.fallback_reason == "locator found no candidates"
