import pytest

from dialogue_locator.errors import V0Error
from dialogue_locator.matching import (
    find_dialogue,
    find_dialogue_candidates,
    normalize_for_matching,
    normalize_text,
)
from dialogue_locator.models import TranscriptWord


def _words(*items: tuple[str, float, float]) -> list[TranscriptWord]:
    return [TranscriptWord(text, start, end) for text, start, end in items]


def test_normalization_handles_case_punctuation_unicode_and_whitespace() -> None:
    assert normalize_text("  My MIND—rebels,\n at stagnation! ") == "my mind rebels at stagnation"


@pytest.mark.parametrize(
    ("words", "digits", "canonical"),
    [
        ("twenty", "20", "20"),
        ("twenty million", "20 million", "20000000"),
        ("twenty million dollars", "$20 million", "20000000 usd"),
        ("20 million dollars", "$20 million", "20000000 usd"),
        ("one hundred and fifty", "150", "150"),
        ("$150,000.", "one hundred and fifty thousand dollars", "150000 usd"),
    ],
)
def test_numeric_and_currency_forms_share_a_conservative_matching_canonicalization(
    words: str,
    digits: str,
    canonical: str,
) -> None:
    assert normalize_for_matching(words) == canonical
    assert normalize_for_matching(digits) == canonical


def test_unrelated_numbers_remain_different() -> None:
    assert normalize_for_matching("twenty million dollars") != normalize_for_matching(
        "thirty million dollars"
    )
    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue("twenty", _words((" thirty", 0.0, 0.5)))


def test_currency_canonicalization_preserves_original_transcript_text() -> None:
    words = _words(
        (" The", 1.0, 1.2),
        (" company", 1.2, 1.5),
        (" reported", 1.5, 1.9),
        (" revenue", 1.9, 2.2),
        (" of", 2.2, 2.3),
        (" $20", 2.3, 2.6),
        (" million.", 2.6, 3.0),
    )

    match = find_dialogue(
        "The company reported revenue of twenty million dollars.",
        words,
        fuzzy_threshold=85.0,
    )

    assert match.match_type == "exact"
    assert match.score == 100.0
    assert match.matched_text == "The company reported revenue of $20 million."
    assert match.start == 1.0
    assert match.end == 3.0


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


def test_straight_and_curly_apostrophes_normalize_equally() -> None:
    assert normalize_text("How's it looking?") == normalize_text("How’s it looking?")
    match = find_dialogue(
        "How’s it looking?",
        _words((" How's", 1.0, 1.4), (" it", 1.4, 1.6), (" looking?", 1.6, 2.0)),
    )
    assert match.match_type == "exact"
    assert match.matched_text == "How's it looking?"


def test_target_across_three_segments_uses_first_and_final_segment_timestamps() -> None:
    words = _words(
        (" My mind", 1.0, 1.8),
        (" rebels at", 1.8, 2.4),
        (" stagnation.", 2.4, 3.0),
    )

    match = find_dialogue("my mind rebels at stagnation", words)

    assert match.match_type == "exact"
    assert match.start == 1.0
    assert match.end == 3.0
    assert match.matched_text == "My mind rebels at stagnation."


def test_long_target_spans_neighbouring_segments() -> None:
    words = _words(
        (" The company reported", 2.0, 2.8),
        (" revenue of twenty million", 2.8, 3.8),
        (" dollars during the last", 3.8, 4.8),
        (" financial quarter.", 4.8, 5.3),
    )

    match = find_dialogue(
        "The company reported revenue of $20 million during the last financial quarter",
        words,
    )

    assert match.match_type == "exact"
    assert (match.start, match.end) == (2.0, 5.3)


def test_earlier_acceptable_fuzzy_occurrence_beats_later_exact_occurrence() -> None:
    words = _words(
        (" My", 1.0, 1.2),
        (" mind", 1.2, 1.5),
        (" revels", 1.5, 1.9),
        (" at", 1.9, 2.0),
        (" stagnation.", 2.0, 2.5),
        (" filler", 3.0, 3.5),
        (" My", 8.0, 8.2),
        (" mind", 8.2, 8.5),
        (" rebels", 8.5, 8.9),
        (" at", 8.9, 9.0),
        (" stagnation.", 9.0, 9.5),
    )

    matches = find_dialogue_candidates("my mind rebels at stagnation", words)

    assert matches[0].match_type == "fuzzy"
    assert matches[0].start == 1.0
    assert matches[1].match_type == "exact"
    assert matches[1].start == 8.0


@pytest.mark.parametrize(
    ("query", "unrelated"),
    [("yes", "yesterday"), ("no", "not"), ("okay", "oak"), ("come", "home")],
)
def test_short_common_target_requires_an_acceptable_match(query: str, unrelated: str) -> None:
    exact = find_dialogue(query, _words((f" {query}!", 1.0, 1.2)))
    assert exact.match_type == "exact"
    assert exact.start == 1.0
    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue(query, _words((f" {unrelated}", 2.0, 2.3)))


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_and_whitespace_only_targets_are_rejected(query: str) -> None:
    with pytest.raises(V0Error, match="letter or number"):
        find_dialogue(query, _words(("hello", 0.0, 0.4)))


def test_empty_transcript_and_one_segment_transcript() -> None:
    with pytest.raises(V0Error, match="no matchable words"):
        find_dialogue("hello", [])

    match = find_dialogue("hello world", _words((" Hello, world!", 4.0, 4.8)))
    assert match.matched_text == "Hello, world!"
    assert (match.start, match.end) == (4.0, 4.8)


def test_slightly_incorrect_user_quotation_uses_existing_fuzzy_match() -> None:
    match = find_dialogue(
        "my mind rebelled at stagnation",
        _words(
            (" My", 5.0, 5.2),
            (" mind", 5.2, 5.5),
            (" rebels", 5.5, 5.9),
            (" at", 5.9, 6.0),
            (" stagnation.", 6.0, 6.5),
        ),
    )
    assert match.match_type == "fuzzy"
    assert match.score >= 85.0


def test_homophone_is_accepted_only_with_strong_surrounding_context() -> None:
    match = find_dialogue(
        "the knight rode through the silent valley",
        _words(
            (" The", 1.0, 1.2),
            (" night", 1.2, 1.5),
            (" rode", 1.5, 1.8),
            (" through", 1.8, 2.1),
            (" the", 2.1, 2.2),
            (" silent", 2.2, 2.6),
            (" valley.", 2.6, 3.0),
        ),
    )

    assert match.match_type == "fuzzy"
    assert match.score >= 85.0
    assert (match.start, match.end) == (1.0, 3.0)

def test_stutter_selects_complete_target_after_repeated_leading_word() -> None:
    match = find_dialogue(
        "please open the laboratory door",
        _words(
            (" Please", 1.0, 1.2),
            (" please", 1.2, 1.4),
            (" open", 1.4, 1.7),
            (" the", 1.7, 1.8),
            (" laboratory", 1.8, 2.2),
            (" door.", 2.2, 2.5),
        ),
    )

    assert match.match_type == "exact"
    assert match.matched_text == "please open the laboratory door."
    assert (match.start, match.end) == (1.2, 2.5)


@pytest.mark.parametrize(
    "transcript",
    [
        (" Please", " remember", " close", " the", " laboratory", " door."),
        (" Please", " remember", " and", " close", " the", " laboratory", " door."),
    ],
)
def test_one_omitted_or_substituted_word_uses_existing_fuzzy_matching(
    transcript: tuple[str, ...],
) -> None:
    words = [TranscriptWord(text, index * 0.4, (index + 1) * 0.4) for index, text in enumerate(transcript)]

    match = find_dialogue("Please remember to close the laboratory door", words)

    assert match.match_type == "fuzzy"
    assert match.score >= 85.0


@pytest.mark.parametrize(
    ("query", "near_miss"),
    [
        ("the", "them"),
        ("it", "its"),
        ("it was", "it is"),
        ("yes", "yess"),
        ("no", "not"),
        ("okay", "okays"),
    ],
)
def test_very_short_ambiguous_targets_accept_exact_but_reject_near_miss(
    query: str,
    near_miss: str,
) -> None:
    exact = find_dialogue(query, _words((f" {query}!", 1.0, 1.3)))
    assert exact.match_type == "exact"

    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue(query, _words((f" {near_miss}", 2.0, 2.4)))


def test_short_distinctive_openai_target_accepts_common_asr_spacing() -> None:
    match = find_dialogue(
        "OpenAI",
        _words((" open", 3.0, 3.3), (" AI", 3.3, 3.6)),
    )

    assert match.match_type == "fuzzy"
    assert match.score >= 85.0
    assert match.matched_text == "open AI"
