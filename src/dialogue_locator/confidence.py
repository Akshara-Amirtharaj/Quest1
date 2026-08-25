from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceAssessment:
    category: str
    reason: str


def assess_confidence(
    *,
    localization_source: str,
    verification_source: str,
    match_type: str,
    match_score: float,
    caption_match_type: str | None = None,
    caption_match_score: float | None = None,
    ocr_match_type: str | None = None,
    ocr_match_score: float | None = None,
    evidence_conflict: bool = False,
) -> ConfidenceAssessment:
    """Classify explicit evidence without combining it into a synthetic score."""
    if evidence_conflict:
        return ConfidenceAssessment(
            "LOW",
            "Accepted evidence sources disagree; inspect the raw caption, ASR, and OCR scores.",
        )

    if match_score < 85:
        return ConfidenceAssessment(
            "LOW",
            f"The accepted {match_type} match score is below the normal 85-point threshold.",
        )

    if caption_match_score is not None and caption_match_score < 85:
        return ConfidenceAssessment(
            "LOW",
            "The caption evidence is below the normal 85-point acceptance threshold.",
        )

    if ocr_match_score is not None and ocr_match_score < 85:
        return ConfidenceAssessment(
            "LOW",
            "The visible-text OCR evidence is below the normal 85-point acceptance threshold.",
        )

    if verification_source == "ocr":
        if match_type == "exact" and ocr_match_type == "exact":
            return ConfidenceAssessment(
                "HIGH",
                "Spoken localization and visible-text OCR independently produced exact matches.",
            )
        return ConfidenceAssessment(
            "MEDIUM",
            "Visible text corroborates the spoken localization, but at least one match is fuzzy.",
        )

    if localization_source == "caption" and verification_source == "asr":
        if match_type == "exact" and caption_match_type in {"exact", "substring"}:
            return ConfidenceAssessment(
                "HIGH",
                "Platform captions and short-window ASR independently produced exact matches.",
            )
        return ConfidenceAssessment(
            "MEDIUM",
            "Platform captions and ASR agree, but at least one accepted match is fuzzy.",
        )

    if match_type == "exact":
        return ConfidenceAssessment(
            "MEDIUM",
            "ASR produced an exact match, but no independent caption or OCR evidence corroborated it.",
        )
    return ConfidenceAssessment(
        "LOW",
        "Only a fuzzy ASR match supports this result; inspect the raw match score and frame.",
    )
