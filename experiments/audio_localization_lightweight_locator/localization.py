from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from dialogue_locator.errors import V0Error
from dialogue_locator.matching import find_dialogue_candidates, normalize_text
from dialogue_locator.models import DialogueMatch, TranscriptWord, Transcription


@dataclass(frozen=True)
class CandidateWindow:
    index: int
    start: float
    end: float
    locator_match: DialogueMatch

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class LocatorSearchResult:
    match: DialogueMatch | None
    candidates: tuple[CandidateWindow, ...]
    candidates_verified: int
    verified_candidate_index: int | None
    fallback_invoked: bool
    fallback_reason: str | None


WindowTranscriber = Callable[[CandidateWindow], Transcription]
FullTranscriber = Callable[[], Transcription]


def generate_candidate_windows(
    query: str,
    locator_words: Iterable[TranscriptWord],
    *,
    audio_duration: float,
    margin_before: float,
    margin_after: float,
    fuzzy_threshold: float,
) -> tuple[CandidateWindow, ...]:
    """Turn lightweight-model matches into clamped chronological windows."""
    if not normalize_text(query):
        raise V0Error("Target dialogue must contain at least one letter or number.")
    if audio_duration <= 0:
        raise ValueError("audio_duration must be greater than zero.")
    if margin_before < 0 or margin_after < 0:
        raise ValueError("candidate margins cannot be negative.")
    matches = _find_all_locator_matches(query, list(locator_words), fuzzy_threshold)
    if not matches:
        return ()

    ordered = sorted(matches, key=lambda match: (match.start, match.end))
    return tuple(
        CandidateWindow(
            index=index,
            start=max(0.0, match.start - margin_before),
            end=min(audio_duration, match.end + margin_after),
            locator_match=match,
        )
        for index, match in enumerate(ordered)
    )


def _find_all_locator_matches(
    query: str,
    words: list[TranscriptWord],
    fuzzy_threshold: float,
) -> list[DialogueMatch]:
    """Reuse the production matcher while retaining fuzzy hits before later exact hits."""
    if not words:
        return []
    query_length = len(normalize_text(query).split())
    length_delta = max(1, min(3, math.ceil(query_length * 0.25)))
    minimum = max(1, query_length - length_delta)
    maximum = query_length + length_delta
    found: list[DialogueMatch] = []
    try:
        found.extend(find_dialogue_candidates(query, words, fuzzy_threshold))
    except V0Error:
        pass
    for start in range(len(words)):
        for length in range(minimum, maximum + 1):
            if start + length > len(words):
                break
            try:
                found.extend(
                    find_dialogue_candidates(
                        query,
                        words[start : start + length],
                        fuzzy_threshold,
                    )
                )
            except V0Error:
                continue
    return _deduplicate_matches(found)


def _deduplicate_matches(matches: Iterable[DialogueMatch]) -> list[DialogueMatch]:
    selected: list[DialogueMatch] = []
    for candidate in sorted(matches, key=lambda match: (match.start, match.end, -match.score)):
        duplicate = next(
            (index for index, current in enumerate(selected) if _same_occurrence(current, candidate)),
            None,
        )
        if duplicate is None:
            selected.append(candidate)
            continue
        current = selected[duplicate]
        if (candidate.score, -candidate.start) > (current.score, -current.start):
            selected[duplicate] = candidate
    return sorted(selected, key=lambda match: (match.start, match.end))


def _same_occurrence(first: DialogueMatch, second: DialogueMatch) -> bool:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shortest = min(max(0.0, first.end - first.start), max(0.0, second.end - second.start))
    return shortest > 0 and intersection / shortest >= 0.5


def verify_candidate_windows(
    query: str,
    candidates: Iterable[CandidateWindow],
    transcribe_window: WindowTranscriber,
    transcribe_full_audio: FullTranscriber,
    *,
    fuzzy_threshold: float,
    audio_start_offset: float = 0.0,
    locator_failure_reason: str | None = None,
) -> LocatorSearchResult:
    """Verify candidates with accurate ASR, then use accurate full-ASR fallback."""
    ordered = tuple(sorted(candidates, key=lambda item: (item.start, item.end, item.index)))
    verified = 0
    for candidate in ordered:
        verified += 1
        try:
            transcription = transcribe_window(candidate)
            relative = find_dialogue_candidates(
                query,
                transcription.words,
                fuzzy_threshold,
            )[0]
        except V0Error:
            continue
        match = offset_match(
            relative,
            candidate.start + audio_start_offset,
        )
        return LocatorSearchResult(
            match=match,
            candidates=ordered,
            candidates_verified=verified,
            verified_candidate_index=candidate.index,
            fallback_invoked=False,
            fallback_reason=None,
        )

    reason = locator_failure_reason
    if reason is None:
        reason = "no locator candidate verified" if ordered else "locator found no candidates"
    try:
        full_transcription = transcribe_full_audio()
        relative = find_dialogue_candidates(
            query,
            full_transcription.words,
            fuzzy_threshold,
        )[0]
        match = offset_match(relative, audio_start_offset)
    except V0Error:
        match = None
    return LocatorSearchResult(
        match=match,
        candidates=ordered,
        candidates_verified=verified,
        verified_candidate_index=None,
        fallback_invoked=True,
        fallback_reason=reason,
    )


def offset_match(match: DialogueMatch, offset: float) -> DialogueMatch:
    return DialogueMatch(
        matched_text=match.matched_text,
        start=match.start + offset,
        end=match.end + offset,
        match_type=match.match_type,
        score=match.score,
    )
