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
    processed_unique_audio_seconds: float
    stopped_on_chunk_index: int | None
    stopped_at_coverage_seconds: float | None
    merged_words: tuple[TranscriptWord, ...]

ChunkTranscriber = Callable[[AudioChunk], Transcription]
NO_TIMESTAMPED_WORDS_ERROR = "Speech recognition produced no timestamped words."


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


def stitch_adjacent_words(
    existing: Iterable[TranscriptWord],
    incoming: Iterable[TranscriptWord],
    *,
    seam: float,
) -> list[TranscriptWord]:
    """Join adjacent ASR observations without interleaving overlap variants.

    Independently decoded overlap can contain different spellings or even a
    different number of words. Token-level deduplication cannot safely merge
    those observations: both variants remain and corrupt the chronological
    token sequence. A deterministic time seam instead chooses exactly one
    observation on either side of the overlap midpoint.
    """
    existing_words = list(existing)
    before = [word for word in existing_words if _midpoint(word) <= seam]
    after: list[TranscriptWord] = []
    for word in incoming:
        if _midpoint(word) <= seam:
            continue
        duplicate = next(
            (
                current
                for current in existing_words
                if normalize_text(current.text) == normalize_text(word.text)
                and _substantially_overlaps(current, word)
            ),
            None,
        )
        after.append(_preferred_observation(duplicate, word) if duplicate is not None else word)
    return sorted([*before, *after], key=lambda word: (word.start, word.end))


def search_chunks(
    query: str,
    chunks: Iterable[AudioChunk],
    transcribe_chunk: ChunkTranscriber,
    *,
    fuzzy_threshold: float,
    audio_start_offset: float = 0.0,
    overlap_seconds: float | None = None,
    transcript_context_seconds: float = 15.0,
) -> ChunkSearchResult:
    """Transcribe/search chronologically and stop on the earliest accepted occurrence.

    Only the current chunk and a bounded canonical tail are searched. This
    keeps matching cost linear in the number of chunks and lets a phrase cross
    a chunk boundary without retaining/interleaving the complete transcript.
    """
    if not normalize_text(query):
        raise V0Error("Target dialogue must contain at least one letter or number.")
    if overlap_seconds is not None and overlap_seconds < 0:
        raise ValueError("overlap_seconds cannot be negative.")
    if transcript_context_seconds <= 0:
        raise ValueError("transcript_context_seconds must be greater than zero.")
    chunk_list = tuple(chunks)
    canonical_words: list[TranscriptWord] = []
    context_words: list[TranscriptWord] = []
    processed_audio = 0.0
    for processed_count, chunk in enumerate(chunk_list, start=1):
        try:
            transcription = transcribe_chunk(chunk)
        except V0Error as exc:
            if str(exc) != NO_TIMESTAMPED_WORDS_ERROR:
                raise
            # A silent chunk is a valid chronological observation, not a failure
            # of the complete search. Preserve coverage and continue scanning.
            transcription = Transcription("", [], None, None)
        absolute_words = restore_absolute_timestamps(
            transcription.words,
            chunk_start=chunk.start,
            audio_start_offset=audio_start_offset,
        )
        if processed_count == 1:
            canonical_words = list(absolute_words)
        else:
            previous_chunk = chunk_list[processed_count - 2]
            effective_overlap = (
                max(0.0, previous_chunk.end - chunk.start)
                if overlap_seconds is None
                else overlap_seconds
            )
            seam = chunk.start + audio_start_offset + effective_overlap / 2.0
            canonical_words = stitch_adjacent_words(
                canonical_words,
                absolute_words,
                seam=seam,
            )
        processed_audio += chunk.duration
        context_start = chunk.start + audio_start_offset - transcript_context_seconds
        context_words = [word for word in canonical_words if word.end >= context_start]
        matches = _accepted_matches(
            query,
            (absolute_words, context_words),
            fuzzy_threshold,
        )
        if not matches:
            continue
        match = min(matches, key=lambda item: (item.start, -item.score, item.end))
        return ChunkSearchResult(
            match=match,
            processed_chunks=processed_count,
            processed_planned_audio_seconds=processed_audio,
            processed_unique_audio_seconds=chunk.end,
            stopped_on_chunk_index=chunk.index,
            stopped_at_coverage_seconds=chunk.end + audio_start_offset,
            merged_words=tuple(canonical_words),
        )
    return ChunkSearchResult(
        match=None,
        processed_chunks=len(chunk_list),
        processed_planned_audio_seconds=processed_audio,
        processed_unique_audio_seconds=chunk_list[-1].end if chunk_list else 0.0,
        stopped_on_chunk_index=None,
        stopped_at_coverage_seconds=None,
        merged_words=tuple(canonical_words),
    )


def _accepted_matches(
    query: str,
    word_sets: Iterable[list[TranscriptWord]],
    fuzzy_threshold: float,
) -> list[DialogueMatch]:
    matches: list[DialogueMatch] = []
    seen: set[tuple[float, float, str]] = set()
    for words in word_sets:
        try:
            candidates = find_dialogue_candidates(query, words, fuzzy_threshold)
        except V0Error:
            continue
        for match in candidates:
            key = (round(match.start, 6), round(match.end, 6), normalize_text(match.matched_text))
            if key not in seen:
                seen.add(key)
                matches.append(match)
    return matches


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


def _midpoint(word: TranscriptWord) -> float:
    return (word.start + word.end) / 2.0
