from __future__ import annotations

from collections.abc import Callable

import pytest

from dialogue_locator.errors import V0Error
from dialogue_locator.models import TranscriptWord, Transcription
from experiments.audio_localization_chunked.chunking import (
    AudioChunk,
    generate_chunks,
    merge_overlapping_words,
    restore_absolute_timestamps,
    search_chunks,
    stitch_adjacent_words,
)


def _transcription(*words: tuple[str, float, float]) -> Transcription:
    timestamped = [TranscriptWord(text, start, end, 0.9) for text, start, end in words]
    return Transcription(" ".join(word[0] for word in words), timestamped, "en", 1.0)


def _indexed_transcriber(
    values: dict[int, Transcription],
    calls: list[int],
) -> Callable[[AudioChunk], Transcription]:
    def transcribe(chunk: AudioChunk) -> Transcription:
        calls.append(chunk.index)
        return values.get(chunk.index, _transcription(("filler", 0.1, 0.5)))

    return transcribe


def test_chunk_boundary_generation_covers_audio_with_overlap() -> None:
    chunks = generate_chunks(audio_duration=25.0, chunk_duration=10.0, overlap=2.0)

    assert [(chunk.index, chunk.start, chunk.end) for chunk in chunks] == [
        (0, 0.0, 10.0),
        (1, 8.0, 18.0),
        (2, 16.0, 25.0),
    ]
    assert chunks[0].start == 0
    assert chunks[-1].end == 25.0


def test_overlap_handling_deduplicates_same_observation_but_keeps_repetition() -> None:
    existing = [
        TranscriptWord("go", 9.0, 9.4, 0.7),
        TranscriptWord("go", 9.5, 9.8, 0.8),
    ]
    incoming = [TranscriptWord("GO!", 9.05, 9.45, 0.95)]

    merged = merge_overlapping_words(existing, incoming)

    assert len(merged) == 2
    assert [(word.start, word.probability) for word in merged] == [(9.05, 0.95), (9.5, 0.8)]


def test_absolute_timestamp_restoration_includes_chunk_and_stream_offsets() -> None:
    restored = restore_absolute_timestamps(
        [TranscriptWord("target", 1.25, 1.75, 0.9)],
        chunk_start=8.0,
        audio_start_offset=0.5,
    )

    assert restored[0].start == 9.75
    assert restored[0].end == 10.25


def test_first_occurrence_across_chunk_boundary_is_found() -> None:
    chunks = generate_chunks(18.0, 10.0, 2.0)
    calls: list[int] = []
    transcriber = _indexed_transcriber(
        {
            0: _transcription(("cross", 9.35, 9.75)),
            1: _transcription(
                ("cross", 1.38, 1.76),
                ("boundary", 2.05, 2.60),
            ),
        },
        calls,
    )

    result = search_chunks(
        "cross boundary",
        chunks,
        transcriber,
        fuzzy_threshold=85.0,
    )

    assert calls == [0, 1]
    assert result.match is not None
    assert result.match.matched_text == "cross boundary"
    assert result.match.start == 9.35
    assert result.processed_chunks == 2


def test_early_stopping_skips_all_later_chunks() -> None:
    chunks = generate_chunks(30.0, 10.0, 2.0)
    calls: list[int] = []
    transcriber = _indexed_transcriber(
        {0: _transcription(("target", 2.0, 2.4), ("words", 2.5, 2.9))},
        calls,
    )

    result = search_chunks("target words", chunks, transcriber, fuzzy_threshold=85.0)

    assert calls == [0]
    assert result.processed_chunks == 1
    assert result.stopped_on_chunk_index == 0
    assert result.stopped_at_coverage_seconds == 10.0


def test_no_match_processes_every_chunk_and_full_coverage() -> None:
    chunks = generate_chunks(25.0, 10.0, 2.0)
    calls: list[int] = []

    result = search_chunks(
        "zebra phrase",
        chunks,
        _indexed_transcriber({}, calls),
        fuzzy_threshold=85.0,
    )

    assert result.match is None
    assert calls == [0, 1, 2]
    assert result.processed_chunks == len(chunks)
    assert result.processed_planned_audio_seconds == sum(chunk.duration for chunk in chunks)
    assert chunks[-1].end == 25.0


def test_first_occurrence_near_start_stops_in_first_chunk() -> None:
    chunks = generate_chunks(60.0, 12.0, 3.0)
    calls: list[int] = []
    result = search_chunks(
        "early phrase",
        chunks,
        _indexed_transcriber(
            {0: _transcription(("early", 0.25, 0.5), ("phrase", 0.55, 0.9))},
            calls,
        ),
        fuzzy_threshold=85.0,
    )

    assert result.match is not None
    assert result.match.start == 0.25
    assert calls == [0]


def test_first_occurrence_near_end_processes_until_final_chunk() -> None:
    chunks = generate_chunks(25.0, 10.0, 2.0)
    calls: list[int] = []
    result = search_chunks(
        "late phrase",
        chunks,
        _indexed_transcriber(
            {2: _transcription(("late", 7.0, 7.3), ("phrase", 7.4, 7.8))},
            calls,
        ),
        fuzzy_threshold=85.0,
    )

    assert result.match is not None
    assert result.match.start == 23.0
    assert calls == [0, 1, 2]
    assert result.processed_chunks == len(chunks)


def test_silent_chunk_without_timestamped_words_does_not_abort_search() -> None:
    chunks = generate_chunks(18.0, 10.0, 2.0)
    calls: list[int] = []

    def transcribe(chunk: AudioChunk) -> Transcription:
        calls.append(chunk.index)
        if chunk.index == 0:
            raise V0Error("Speech recognition produced no timestamped words.")
        return _transcription(("target", 1.0, 1.4), ("phrase", 1.5, 1.9))

    result = search_chunks("target phrase", chunks, transcribe, fuzzy_threshold=85.0)

    assert calls == [0, 1]
    assert result.match is not None
    assert result.match.start == 9.0
    assert result.processed_chunks == 2


def test_non_silence_transcription_error_is_not_suppressed() -> None:
    chunks = generate_chunks(8.0, 8.0, 2.0)

    def transcribe(_chunk: AudioChunk) -> Transcription:
        raise V0Error("Speech recognition failed with model 'base.en': test failure")

    with pytest.raises(V0Error, match="test failure"):
        search_chunks("target phrase", chunks, transcribe, fuzzy_threshold=85.0)


def test_overlap_wording_variants_are_not_interleaved_at_seam() -> None:
    previous = [
        TranscriptWord("a", 7.5, 7.8, 0.9),
        TranscriptWord("Torovian", 8.7, 9.3, 0.8),
        TranscriptWord("custom", 9.4, 9.9, 0.8),
    ]
    incoming = [
        TranscriptWord("Taruvian", 8.65, 9.25, 0.9),
        TranscriptWord("custom", 9.35, 9.85, 0.9),
        TranscriptWord("after", 10.1, 10.5, 0.9),
    ]

    stitched = stitch_adjacent_words(previous, incoming, seam=9.0)

    assert [word.text for word in stitched] == ["a", "Torovian", "custom", "after"]


def test_exact_first_occurrence_wins_and_middle_target_stops_scan() -> None:
    chunks = generate_chunks(100.0, 30.0, 5.0)
    calls: list[int] = []
    result = search_chunks(
        "middle target",
        chunks,
        _indexed_transcriber(
            {
                1: _transcription(
                    ("middle", 8.0, 8.4),
                    ("target", 8.5, 8.9),
                    ("middle", 18.0, 18.4),
                    ("target", 18.5, 18.9),
                )
            },
            calls,
        ),
        fuzzy_threshold=85.0,
        overlap_seconds=5.0,
    )

    assert result.match is not None
    assert result.match.match_type == "exact"
    assert result.match.start == 33.0
    assert calls == [0, 1]


def test_adjacent_context_recovers_boundary_phrase_despite_wording_difference() -> None:
    chunks = generate_chunks(60.0, 30.0, 5.0)
    calls: list[int] = []
    result = search_chunks(
        "not exclusively a Torovian custom",
        chunks,
        _indexed_transcriber(
            {
                0: _transcription(
                    ("not", 27.0, 27.3),
                    ("exclusively", 27.4, 28.0),
                    ("a", 28.1, 28.2),
                ),
                1: _transcription(
                    ("exclusively", 2.4, 3.0),
                    ("a", 3.1, 3.2),
                    ("Taruvian", 3.3, 3.8),
                    ("custom", 3.9, 4.3),
                ),
            },
            calls,
        ),
        fuzzy_threshold=85.0,
        overlap_seconds=5.0,
        transcript_context_seconds=15.0,
    )

    assert result.match is not None
    assert result.match.match_type == "fuzzy"
    assert result.match.start == 27.0
    assert calls == [0, 1]
