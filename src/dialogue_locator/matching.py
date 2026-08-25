from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from .errors import V0Error
from .models import DialogueMatch, TranscriptWord


DEFAULT_FUZZY_THRESHOLD = 85.0

_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORDS = set(_SMALL_NUMBERS) | set(_TENS)
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_DOLLAR_WORDS = {"dollar", "dollars", "usd"}
_DOLLAR_MARKER = "currencydollarsymbol"


@dataclass(frozen=True)
class _NormalizedToken:
    text: str
    word_index: int
    last_word_index: int


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def normalize_for_matching(text: str) -> str:
    """Normalize text plus conservative number/currency equivalence for matching only."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = normalized.replace("$", f" {_DOLLAR_MARKER} ")
    tokens = [
        _NormalizedToken(token, 0, 0)
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    ]
    return " ".join(token.text for token in _canonicalize_numeric_tokens(tokens))


def _transcript_tokens(words: list[TranscriptWord]) -> list[_NormalizedToken]:
    tokens: list[_NormalizedToken] = []
    for word_index, word in enumerate(words):
        normalized = unicodedata.normalize("NFKC", word.text).casefold()
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        normalized = normalized.replace("$", f" {_DOLLAR_MARKER} ")
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE):
            tokens.append(_NormalizedToken(token, word_index, word_index))
    return _canonicalize_numeric_tokens(tokens)


def _canonicalize_numeric_tokens(tokens: list[_NormalizedToken]) -> list[_NormalizedToken]:
    canonical: list[_NormalizedToken] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.text == _DOLLAR_MARKER:
            parsed = _consume_number(tokens, index + 1)
            if parsed is not None:
                value, end = parsed
                if end < len(tokens) and tokens[end].text in _DOLLAR_WORDS:
                    end += 1
                last_word = tokens[end - 1].last_word_index
                canonical.extend(
                    (
                        _NormalizedToken(str(value), token.word_index, last_word),
                        _NormalizedToken("usd", token.word_index, last_word),
                    )
                )
                index = end
                continue

        parsed = _consume_number(tokens, index)
        if parsed is not None:
            value, end = parsed
            last_word = tokens[end - 1].last_word_index
            canonical.append(_NormalizedToken(str(value), token.word_index, last_word))
            if end < len(tokens) and tokens[end].text in _DOLLAR_WORDS:
                last_word = tokens[end].last_word_index
                canonical[-1] = _NormalizedToken(str(value), token.word_index, last_word)
                canonical.append(_NormalizedToken("usd", token.word_index, last_word))
                end += 1
            index = end
            continue

        if token.text in _DOLLAR_WORDS:
            canonical.append(_NormalizedToken("usd", token.word_index, token.last_word_index))
        elif token.text != _DOLLAR_MARKER:
            canonical.append(token)
        index += 1
    return canonical


def _consume_number(
    tokens: list[_NormalizedToken],
    start: int,
) -> tuple[int, int] | None:
    if start >= len(tokens):
        return None
    first = tokens[start].text
    if first.isdecimal():
        value = int(first)
        end = start + 1
        if end < len(tokens) and tokens[end].text in _SCALES and value > 0:
            value *= _SCALES[tokens[end].text]
            end += 1
        return value, end
    if first not in _SMALL_NUMBERS and first not in _TENS:
        return None

    total = 0
    current = 0
    index = start
    last_kind: str | None = None
    last_scale: int | None = None
    while index < len(tokens):
        text = tokens[index].text
        if text in _SMALL_NUMBERS:
            value = _SMALL_NUMBERS[text]
            if last_kind == "tens" and value < 10:
                current += value
            elif last_kind in {"small", "tens"}:
                break
            else:
                current += value
            last_kind = "small"
        elif text in _TENS:
            if last_kind in {"small", "tens"}:
                break
            current += _TENS[text]
            last_kind = "tens"
        elif text == "hundred":
            if last_kind != "small" or not 1 <= current <= 9:
                break
            current *= 100
            last_kind = "hundred"
        elif text == "and":
            if last_kind not in {"hundred", "scale"} or index + 1 >= len(tokens):
                break
            if tokens[index + 1].text not in _NUMBER_WORDS:
                break
            last_kind = "and"
        elif text in _SCALES:
            scale = _SCALES[text]
            if current <= 0 or last_kind not in {"small", "tens", "hundred"}:
                break
            if last_scale is not None and scale >= last_scale:
                break
            total += current * scale
            current = 0
            last_scale = scale
            last_kind = "scale"
        else:
            break
        index += 1
    if index == start or last_kind == "and":
        return None
    return total + current, index


def _match_from_tokens(
    words: list[TranscriptWord],
    tokens: list[_NormalizedToken],
    start: int,
    length: int,
    match_type: str,
    score: float,
) -> DialogueMatch:
    first_word = tokens[start].word_index
    last_word = tokens[start + length - 1].last_word_index
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


def best_dialogue_match(
    query: str,
    words: list[TranscriptWord],
) -> DialogueMatch:
    """Return the highest-scoring diagnostic candidate without applying a threshold."""
    normalized_query = normalize_for_matching(query)
    if not normalized_query:
        raise V0Error("Target dialogue must contain at least one letter or number.")
    tokens = _transcript_tokens(words)
    if not tokens:
        raise V0Error("The transcription contains no matchable words.")

    query_tokens = normalized_query.split()
    query_length = len(query_tokens)
    transcript_texts = [token.text for token in tokens]
    length_delta = max(1, min(3, math.ceil(query_length * 0.25)))
    minimum_length = max(1, query_length - length_delta)
    maximum_length = min(len(tokens), query_length + length_delta)
    minimum_length = min(minimum_length, maximum_length)
    best: tuple[float, int, int] | None = None
    for start in range(len(tokens)):
        for length in range(minimum_length, maximum_length + 1):
            if start + length > len(tokens):
                break
            candidate = " ".join(transcript_texts[start : start + length])
            score = float(fuzz.ratio(normalized_query, candidate))
            candidate_key = (score, -abs(length - query_length), -start)
            if best is None:
                best = (score, start, length)
                continue
            best_score, best_start, best_length = best
            best_key = (best_score, -abs(best_length - query_length), -best_start)
            if candidate_key > best_key:
                best = (score, start, length)

    if best is None:
        raise V0Error("The transcription contains no matchable words.")
    score, start, length = best
    match_type = "exact" if score == 100.0 else "fuzzy"
    return _match_from_tokens(words, tokens, start, length, match_type, score)


def find_dialogue_candidates(
    query: str,
    words: list[TranscriptWord],
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[DialogueMatch]:
    normalized_query = normalize_for_matching(query)
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
