"""Lightweight Whisper localization followed by accurate-ASR verification."""

from .localization import CandidateWindow, LocatorSearchResult

__all__ = ["CandidateWindow", "LocatorSearchResult"]
