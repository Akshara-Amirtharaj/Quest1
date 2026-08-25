from dialogue_locator.caption_matching import find_caption_candidates, merge_caption_candidates
from dialogue_locator.models import SubtitleEntry


def test_exact_candidate_spans_adjacent_subtitle_entries() -> None:
    entries = [
        SubtitleEntry("My mind", 10.0, 11.0),
        SubtitleEntry("rebels at stagnation.", 11.0, 12.5),
    ]

    candidates = find_caption_candidates(
        "My mind rebels at stagnation", entries, "en", "manual", max_window_entries=3
    )

    exact = [candidate for candidate in candidates if candidate.match_type == "exact"]
    assert len(exact) == 1
    assert (exact[0].start, exact[0].end) == (10.0, 12.5)


def test_fuzzy_caption_candidate_handles_minor_difference() -> None:
    entries = [SubtitleEntry("My mind revels at stagnation", 4.0, 6.0)]

    candidates = find_caption_candidates(
        "My mind rebels at stagnation", entries, "en", "automatic", fuzzy_threshold=85
    )

    assert candidates[0].match_type == "fuzzy"
    assert candidates[0].score >= 85


def test_substring_candidate_allows_caption_context_around_query() -> None:
    entries = [SubtitleEntry("Narrator: target words appear now", 7.0, 8.0)]

    candidates = find_caption_candidates("target words", entries, "en", "manual")

    assert candidates[0].match_type == "substring"
    assert candidates[0].start == 7.0


def test_candidates_from_tracks_are_chronological() -> None:
    late = find_caption_candidates(
        "target words", [SubtitleEntry("target words", 20.0, 21.0)], "en", "manual"
    )
    early = find_caption_candidates(
        "target words", [SubtitleEntry("target words", 5.0, 6.0)], "en", "automatic"
    )

    merged = merge_caption_candidates([late, early])

    assert [candidate.start for candidate in merged] == [5.0, 20.0]
