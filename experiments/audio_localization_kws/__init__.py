"""Open-vocabulary keyword spotting plus accurate-ASR verification experiment."""

from .anchors import generate_phrase_anchors
from .candidates import AnchorDetection, KWSCandidateRegion

__all__ = ["AnchorDetection", "KWSCandidateRegion", "generate_phrase_anchors"]
