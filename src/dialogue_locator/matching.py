from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from .errors import V0Error
from .models import DialogueMatch, TranscriptWord


DEFAULT_FUZZY_THRESHOLD = 85.0


@dataclass(frozen=True)
class _NormalizedToken:
    text: str
    word_index: int


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _transcript_tokens(words: list[TranscriptWord]) -> list[_NormalizedToken]:
    tokens: list[_NormalizedToken] = []
    for word_index, word in enumerate(words):
        for token in normalize_text(word.text).split():
            tokens.append(_NormalizedToken(token, word_index))
    return tokens


def _match_from_tokens(
    words: list[TranscriptWord],
    tokens: list[_NormalizedToken],
    start: int,
    length: int,
    match_type: str,
    score: float,
) -> DialogueMatch:
    first_word = tokens[start].word_index
    last_word = tokens[start + length - 1].word_index
    matched_words = words[first_word : last_word + 1]
    matched_text = " ".join(word.text.strip() for word in matched_words).strip()
    matched_text = re.sub(r"\s+([.,!?;:])", r"\1", matched_text)
    return DialogueMatch(
        matched_text=matched_text,
        start=matched_words[0].start,
        end=matched_words[-1].end,
        match_type=match_type,
        score=score,
    )


def find_dialogue(
    query: str,
    words: list[TranscriptWord],
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> DialogueMatch:
    return find_dialogue_candidates(query, words, fuzzy_threshold)[0]


def find_dialogue_candidates(
    query: str,
    words: list[TranscriptWord],
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[DialogueMatch]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        raise V0Error("Target dialogue must contain at least one letter or number.")
    tokens = _transcript_tokens(words)
    if not tokens:
        raise V0Error("The transcription contains no matchable words.")

    query_tokens = normalized_query.split()
    query_length = len(query_tokens)
    transcript_texts = [token.text for token in tokens]

    exact_matches: list[DialogueMatch] = []
    for start in range(0, len(tokens) - query_length + 1):
        if transcript_texts[start : start + query_length] == query_tokens:
            exact_matches.append(
                _match_from_tokens(words, tokens, start, query_length, "exact", 100.0)
            )
    if exact_matches:
        return exact_matches

    length_delta = max(1, min(3, math.ceil(query_length * 0.25)))
    minimum_length = max(1, query_length - length_delta)
    maximum_length = query_length + length_delta
    fuzzy_matches: list[DialogueMatch] = []
    search_start = 0
    while search_start < len(tokens):
        first_candidate: tuple[int, int, float] | None = None
        for start in range(search_start, len(tokens)):
            best_length = 0
            best_score = -1.0
            for length in range(minimum_length, maximum_length + 1):
                if start + length > len(tokens):
                    break
                candidate = " ".join(transcript_texts[start : start + length])
                score = float(fuzz.ratio(normalized_query, candidate))
                if score > best_score:
                    best_score = score
                    best_length = length
            if best_score >= fuzzy_threshold:
                first_candidate = (start, best_length, best_score)
                break
        if first_candidate is None:
            break

        first_start, first_length, first_score = first_candidate
        best_start = first_start
        best_length = first_length
        best_score = first_score
        first_end = first_start + first_length

        # A variable-length window can cross the real phrase boundary. Refine
        # only within that first overlapping occurrence so a leading filler
        # word does not become the reported dialogue start, while preserving
        # the requirement that the earliest occurrence wins.
        for start in range(first_start + 1, min(first_end, len(tokens))):
            for length in range(minimum_length, maximum_length + 1):
                if start + length > len(tokens):
                    break
                candidate = " ".join(transcript_texts[start : start + length])
                score = float(fuzz.ratio(normalized_query, candidate))
                if score < fuzzy_threshold:
                    continue
                current_key = (best_score, -abs(best_length - query_length), -best_start)
                candidate_key = (score, -abs(length - query_length), -start)
                if candidate_key > current_key:
                    best_start = start
                    best_length = length
                    best_score = score

        fuzzy_matches.append(
            _match_from_tokens(words, tokens, best_start, best_length, "fuzzy", best_score)
        )

        last_word_index = tokens[best_start + best_length - 1].word_index
        search_start = best_start + best_length
        while search_start < len(tokens) and tokens[search_start].word_index <= last_word_index:
            search_start += 1

    if fuzzy_matches:
        return fuzzy_matches

    raise V0Error(
        f"Dialogue not found in the spoken-audio transcription "
        f"(fuzzy threshold {fuzzy_threshold:g})."
    )
