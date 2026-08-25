from __future__ import annotations

from typing import Any, Mapping

from .models import CaptionInventory, CaptionTrack


def _normalize_tracks(raw: Any) -> list[CaptionTrack]:
    if not isinstance(raw, Mapping):
        return []

    tracks: list[CaptionTrack] = []
    for language, variants in raw.items():
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            tracks.append(
                CaptionTrack(
                    language=str(language),
                    name=_text_or_none(variant.get("name")),
                    extension=_text_or_none(variant.get("ext")),
                    url=_text_or_none(variant.get("url")),
                    protocol=_text_or_none(variant.get("protocol")),
                    impersonate=variant.get("impersonate"),
                    http_headers=_headers_or_none(variant.get("http_headers")),
                )
            )
    return tracks


def _text_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _headers_or_none(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): str(header) for key, header in value.items()}


def discover_captions(metadata: Mapping[str, Any]) -> CaptionInventory:
    return CaptionInventory(
        platform_subtitles=_normalize_tracks(metadata.get("subtitles")),
        automatic_captions=_normalize_tracks(metadata.get("automatic_captions")),
    )
