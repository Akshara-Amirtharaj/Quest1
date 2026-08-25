from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from dialogue_locator.errors import V0Error
from dialogue_locator.matching import find_dialogue_candidates, normalize_text
from dialogue_locator.models import DialogueMatch, TranscriptWord, Transcription


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ChunkSearchResult:
    match: DialogueMatch | None
    processed_chunks: int
    processed_planned_audio_seconds: float
    stopped_on_chunk_index: int | None
    stopped_at_coverage_seconds: float | None
    merged_words: tuple[TranscriptWord, ...]


ChunkTranscriber = Callable[[AudioChunk], Transcription]


def generate_chunks(
    audio_duration: float,
    chunk_duration: float,
    overlap: float,
) -> tuple[AudioChunk, ...]:
    """Generate chronological half-open windows covering the complete audio."""
    for value, name in (
        (audio_duration, "audio_duration"),
        (chunk_duration, "chunk_duration"),
        (overlap, "overlap"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if audio_duration <= 0:
        raise ValueError("audio_duration must be greater than zero.")
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_duration:
        raise ValueError("overlap must be smaller than chunk_duration.")

    chunks: list[AudioChunk] = []
    step = chunk_duration - overlap
    start = 0.0
    index = 0
    while start < audio_duration - 1e-9:
        end = min(audio_duration, start + chunk_duration)
        chunks.append(AudioChunk(index=index, start=start, end=end))
        if end >= audio_duration - 1e-9:
            break
        start += step
        index += 1
    return tuple(chunks)


def restore_absolute_timestamps(
    words: Iterable[TranscriptWord],
    *,
    chunk_start: float,
    audio_start_offset: float = 0.0,
) -> list[TranscriptWord]:
    offset = chunk_start + audio_start_offset
    return [
        TranscriptWord(
            text=word.text,
            start=word.start + offset,
            end=word.end + offset,
            probability=word.probability,
        )
        for word in words
    ]


def merge_overlapping_words(
    existing: Iterable[TranscriptWord],
    incoming: Iterable[TranscriptWord],
) -> list[TranscriptWord]:
    """Merge overlapping ASR observations without duplicating the same spoken word."""
    merged = list(existing)
    for candidate in incoming:
        duplicate_index = next(
            (
                index
                for index, current in enumerate(merged)
                if normalize_text(current.text) == normalize_text(candidate.text)
                and _substantially_overlaps(current, candidate)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        merged[duplicate_index] = _preferred_observation(merged[duplicate_index], candidate)
    return sorted(merged, key=lambda word: (word.start, word.end))


def search_chunks(
    query: str,
    chunks: Iterable[AudioChunk],
    transcribe_chunk: ChunkTranscriber,
    *,
    fuzzy_threshold: float,
    audio_start_offset: float = 0.0,
) -> ChunkSearchResult:
    """Transcribe/search chronologically and stop on the earliest accepted occurrence."""
    if not normalize_text(query):
        raise V0Error("Target dialogue must contain at least one letter or number.")
    chunk_list = tuple(chunks)
    merged_words: list[TranscriptWord] = []
    processed_audio = 0.0
    for processed_count, chunk in enumerate(chunk_list, start=1):
        transcription = transcribe_chunk(chunk)
        absolute_words = restore_absolute_timestamps(
            transcription.words,
            chunk_start=chunk.start,
            audio_start_offset=audio_start_offset,
        )
        merged_words = merge_overlapping_words(merged_words, absolute_words)
        processed_audio += chunk.duration
        try:
            match = find_dialogue_candidates(query, merged_words, fuzzy_threshold)[0]
        except V0Error:
            continue
        return ChunkSearchResult(
            match=match,
            processed_chunks=processed_count,
            processed_planned_audio_seconds=processed_audio,
            stopped_on_chunk_index=chunk.index,
            stopped_at_coverage_seconds=chunk.end + audio_start_offset,
            merged_words=tuple(merged_words),
        )
    return ChunkSearchResult(
        match=None,
        processed_chunks=len(chunk_list),
        processed_planned_audio_seconds=processed_audio,
        stopped_on_chunk_index=None,
        stopped_at_coverage_seconds=None,
        merged_words=tuple(merged_words),
    )


def _substantially_overlaps(first: TranscriptWord, second: TranscriptWord) -> bool:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shortest = min(max(0.0, first.end - first.start), max(0.0, second.end - second.start))
    return shortest > 0 and intersection / shortest >= 0.5


def _preferred_observation(first: TranscriptWord, second: TranscriptWord) -> TranscriptWord:
    first_probability = first.probability if first.probability is not None else -1.0
    second_probability = second.probability if second.probability is not None else -1.0
    if second_probability > first_probability:
        return second
    if second_probability == first_probability and second.start < first.start:
        return second
    return first
