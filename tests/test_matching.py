import pytest

from dialogue_locator.errors import V0Error
from dialogue_locator.matching import find_dialogue, normalize_text
from dialogue_locator.models import TranscriptWord


def _words(*items: tuple[str, float, float]) -> list[TranscriptWord]:
    return [TranscriptWord(text, start, end) for text, start, end in items]


def test_normalization_handles_case_punctuation_unicode_and_whitespace() -> None:
    assert normalize_text("  My MIND—rebels,\n at stagnation! ") == "my mind rebels at stagnation"


def test_exact_match_can_span_asr_segments() -> None:
    # Segment boundaries disappear in the chronological word stream.
    words = _words(
        (" My", 1.0, 1.2),
        (" mind", 1.2, 1.5),
        (" rebels", 1.5, 1.9),
        (" at", 2.0, 2.1),
        (" stagnation.", 2.1, 2.7),
    )

    match = find_dialogue("MY mind rebels at stagnation!", words)

    assert match.match_type == "exact"
    assert match.matched_text == "My mind rebels at stagnation."
    assert match.start == 1.0
    assert match.end == 2.7


def test_fuzzy_match_handles_minor_asr_difference() -> None:
    words = _words(
        (" My", 3.0, 3.2),
        (" mind", 3.2, 3.5),
        (" revels", 3.5, 3.9),
        (" at", 3.9, 4.0),
        (" stagnation", 4.0, 4.5),
    )

    match = find_dialogue("my mind rebels at stagnation", words)

    assert match.match_type == "fuzzy"
    assert match.matched_text == "My mind revels at stagnation"
    assert match.score >= 85


def test_fuzzy_match_trims_leading_word_from_same_occurrence() -> None:
    words = _words(
        (" time", 1.0, 2.0),
        (" My", 3.0, 3.2),
        (" mind", 3.2, 3.5),
        (" rebels", 3.5, 3.9),
        (" its", 3.9, 4.1),
        (" stagnation", 4.1, 4.8),
    )

    match = find_dialogue("My mind rebels at stagnation", words)

    assert match.match_type == "fuzzy"
    assert match.matched_text == "My mind rebels its stagnation"
    assert match.start == 3.0
    assert match.end == 4.8


def test_first_exact_occurrence_wins() -> None:
    words = _words(
        (" first", 0.0, 0.2),
        (" phrase", 0.2, 0.5),
        (" filler", 0.5, 1.0),
        (" first", 4.0, 4.2),
        (" phrase", 4.2, 4.5),
    )

    match = find_dialogue("first phrase", words)

    assert match.start == 0.0
    assert match.end == 0.5


def test_dialogue_not_found_is_clear() -> None:
    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue("completely different words", _words(("hello", 0.0, 0.4)))


def test_empty_normalized_query_is_rejected() -> None:
    with pytest.raises(V0Error, match="letter or number"):
        find_dialogue("...", _words(("hello", 0.0, 0.4)))
