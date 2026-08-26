from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz

from .matching import normalize_text
from .models import CaptionCandidate, SubtitleEntry


_MATCH_PRIORITY = {"exact": 0, "substring": 1, "fuzzy": 2}


def find_caption_candidates(
    query: str,
    entries: list[SubtitleEntry],
    language: str,
    caption_source: str,
    max_window_entries: int = 4,
    fuzzy_threshold: float = 85.0,
) -> list[CaptionCandidate]:
    normalized_query = normalize_text(query)
    if not normalized_query or not entries:
        return []

    candidates: list[CaptionCandidate] = []
    for start_index, first_entry in enumerate(entries):
        window_text: list[str] = []
        for end_index in range(start_index, len(entries)):
            window_text.append(entries[end_index].text)
            combined_text = " ".join(window_text)
            normalized_caption = normalize_text(combined_text)
            if not normalized_caption:
                continue
            match_type: str | None
            if normalized_caption == normalized_query:
                match_type, score = "exact", 100.0
            elif normalized_query in normalized_caption:
                match_type, score = "substring", 100.0
            else:
                score = float(fuzz.ratio(normalized_query, normalized_caption))
                match_type = "fuzzy" if score >= fuzzy_threshold else None
            if match_type is not None:
                candidates.append(
                    CaptionCandidate(
                        text=combined_text,
                        start=first_entry.start,
                        end=entries[end_index].end,
                        match_type=match_type,
                        score=score,
                        language=language,
                        caption_source=caption_source,
                    )
                )

            # Keep the configured window for normal queries, but allow long
            # dialogue to span as many cues as needed to accumulate comparable
            # text. This avoids globally widening every caption search.
            window_entries = end_index - start_index + 1
            if (
                window_entries >= max_window_entries
                and len(normalized_caption) >= len(normalized_query)
            ):
                break

    return _deduplicate_candidates(candidates)


def merge_caption_candidates(groups: Iterable[list[CaptionCandidate]]) -> list[CaptionCandidate]:
    return _deduplicate_candidates(candidate for group in groups for candidate in group)


def _deduplicate_candidates(candidates: Iterable[CaptionCandidate]) -> list[CaptionCandidate]:
    best: dict[tuple[int, int], CaptionCandidate] = {}
    for candidate in candidates:
        key = (round(candidate.start * 1000), round(candidate.end * 1000))
        current = best.get(key)
        if current is None or (_MATCH_PRIORITY[candidate.match_type], -candidate.score) < (
            _MATCH_PRIORITY[current.match_type],
            -current.score,
        ):
            best[key] = candidate
    return sorted(
        best.values(),
        key=lambda candidate: (
            candidate.start,
            _MATCH_PRIORITY[candidate.match_type],
            candidate.end,
            -candidate.score,
        ),
    )
