from __future__ import annotations

from collections.abc import Callable

from dialogue_locator.models import TranscriptWord, Transcription
from experiments.audio_localization_chunked.chunking import AudioChunk, generate_chunks, search_chunks


def _transcription(*words: tuple[str, float, float]) -> Transcription:
    items = [TranscriptWord(text, start, end, 0.9) for text, start, end in words]
    return Transcription(" ".join(text for text, _, _ in words), items, "en", 1.0)


def _transcriber(
    values: dict[int, Transcription],
    calls: list[int],
) -> Callable[[AudioChunk], Transcription]:
    def transcribe(chunk: AudioChunk) -> Transcription:
        calls.append(chunk.index)
        return values.get(chunk.index, _transcription(("speech", 0.1, 0.4)))

    return transcribe


def test_vad_dialogue_crossing_chunk_boundary_preserves_overlap_behavior() -> None:
    chunks = generate_chunks(18.0, 10.0, 2.0)
    calls: list[int] = []
    result = search_chunks(
        "cross boundary",
        chunks,
        _transcriber(
            {
                0: _transcription(("cross", 9.3, 9.7)),
                1: _transcription(("cross", 1.3, 1.7), ("boundary", 2.0, 2.5)),
            },
            calls,
        ),
        fuzzy_threshold=85.0,
    )

    assert result.match is not None
    assert result.match.start == 9.3
    assert calls == [0, 1]


def test_vad_no_match_processes_all_chunks_before_fallback_decision() -> None:
    chunks = generate_chunks(25.0, 10.0, 2.0)
    calls: list[int] = []
    result = search_chunks(
        "missing target",
        chunks,
        _transcriber({}, calls),
        fuzzy_threshold=85.0,
    )

    assert result.match is None
    assert calls == [0, 1, 2]
    assert result.processed_chunks == len(chunks)


def test_vad_target_near_beginning_stops_first_chunk() -> None:
    chunks = generate_chunks(30.0, 8.0, 2.0)
    calls: list[int] = []
    result = search_chunks(
        "early target",
        chunks,
        _transcriber(
            {0: _transcription(("early", 0.2, 0.5), ("target", 0.6, 1.0))},
            calls,
        ),
        fuzzy_threshold=85.0,
    )

    assert result.match is not None
    assert result.match.start == 0.2
    assert calls == [0]


def test_vad_target_near_end_processes_final_chunk() -> None:
    chunks = generate_chunks(25.0, 10.0, 2.0)
    calls: list[int] = []
    result = search_chunks(
        "late target",
        chunks,
        _transcriber(
            {2: _transcription(("late", 7.0, 7.3), ("target", 7.4, 7.8))},
            calls,
        ),
        fuzzy_threshold=85.0,
    )

    assert result.match is not None
    assert result.match.start == 23.0
    assert calls == [0, 1, 2]
