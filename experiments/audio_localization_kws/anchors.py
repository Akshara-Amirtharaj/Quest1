from __future__ import annotations

import re

from dialogue_locator.matching import normalize_text


STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
        "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so",
        "that", "the", "their", "them", "there", "they", "this", "to", "us",
        "very", "was", "we", "were", "with", "you", "your",
    }
)
NUMBER_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand", "million", "billion", "trillion",
    }
)


def generate_phrase_anchors(target: str, max_anchors: int = 3) -> tuple[str, ...]:
    """Generate deterministic short phrase anchors without semantic models."""
    if max_anchors <= 0:
        raise ValueError("max_anchors must be greater than zero.")
    normalized = normalize_text(target)
    if not normalized:
        raise ValueError("target must contain at least one letter or number.")
    tokens = normalized.split()
    if len(tokens) <= 2:
        return (" ".join(tokens),)

    candidates: list[tuple[float, int, str]] = []
    original_tokens = re.findall(r"[^\W_]+", target, flags=re.UNICODE)
    for index in range(len(tokens) - 1):
        first, second = tokens[index : index + 2]
        if first in STOPWORDS and second in STOPWORDS:
            continue
        phrase = f"{first} {second}"
        score = len(first) + len(second)
        score += 5 * sum(
            token in NUMBER_WORDS or token.isdecimal() for token in (first, second)
        )
        if index < len(original_tokens) - 1:
            score += 2 * sum(
                token[:1].isupper() and not token.isupper()
                for token in original_tokens[index : index + 2]
            )
        candidates.append((float(score), index, phrase))

    if not candidates:
        return (" ".join(tokens),)
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:max_anchors]
    selected.sort(key=lambda item: item[1])
    return tuple(item[2] for item in selected)
