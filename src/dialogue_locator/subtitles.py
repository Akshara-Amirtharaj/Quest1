from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pysubs2
import requests
from yt_dlp import YoutubeDL
from yt_dlp.downloader import get_suitable_downloader

from .models import CaptionInventory, CaptionTrack, SubtitleEntry


SUPPORTED_SUBTITLE_EXTENSIONS = ("json3", "vtt", "srt", "ttml", "ass", "ssa")
SUBTITLE_DOWNLOAD_TIMEOUT = (15, 30)
MAX_SUBTITLE_BYTES = 10 * 1024 * 1024


class SubtitleRateLimitError(ValueError):
    """The caption provider asked the client to stop making requests."""


class _QuietYtDlpLogger:
    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def _language_matches(track_language: str, requested_language: str) -> bool:
    track = track_language.casefold().replace("_", "-")
    requested = requested_language.casefold().replace("_", "-")
    return track == requested or track.split("-", 1)[0] == requested.split("-", 1)[0]


def select_caption_tracks(
    inventory: CaptionInventory,
    requested_language: str | None = None,
) -> list[tuple[str, CaptionTrack]]:
    selected: list[tuple[str, CaptionTrack]] = []
    seen: set[tuple[str, str, str, str]] = set()
    sources = (
        ("manual", inventory.platform_subtitles),
        ("automatic", inventory.automatic_captions),
    )
    for source, tracks in sources:
        ordered = sorted(
            tracks,
            key=lambda track: (
                track.language.casefold(),
                SUPPORTED_SUBTITLE_EXTENSIONS.index(track.extension.casefold())
                if track.extension and track.extension.casefold() in SUPPORTED_SUBTITLE_EXTENSIONS
                else len(SUPPORTED_SUBTITLE_EXTENSIONS),
            ),
        )
        for track in ordered:
            extension = (track.extension or "").casefold()
            if extension not in SUPPORTED_SUBTITLE_EXTENSIONS or not track.url:
                continue
            if requested_language and not _language_matches(track.language, requested_language):
                continue
            key = (source, track.language.casefold(), extension, track.url)
            if key in seen:
                continue
            seen.add(key)
            selected.append((source, track))
    return selected


def download_subtitle(track: CaptionTrack, destination_dir: Path) -> Path:
    if not track.url or not track.extension:
        raise ValueError("Subtitle track is missing a URL or format")
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_language = "".join(char if char.isalnum() or char in "-_" else "_" for char in track.language)
    destination = destination_dir / f"subtitle-{safe_language}.{track.extension.casefold()}"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    orphan = _find_completed_orphan(destination)
    if orphan is not None:
        orphan.replace(destination)
        return destination

    temporary_dir = destination_dir / f".download-{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    temporary = temporary_dir / destination.name
    try:
        try:
            if track.impersonate is not None or (
                track.protocol and track.protocol not in {"http", "https"}
            ):
                _download_with_ytdlp(track, temporary)
            else:
                _download_with_requests(track, temporary)
        except Exception:
            # A concurrent V2 process may have completed the same stable cache
            # entry while this provider request was in flight.
            if destination.is_file() and destination.stat().st_size > 0:
                return destination
            raise
        actual_download = _find_actual_download(temporary, temporary_dir)
        if actual_download is None or actual_download.stat().st_size == 0:
            raise ValueError("Subtitle track is empty")
        with actual_download.open("rb") as subtitle_stream:
            prefix = subtitle_stream.read(512).lstrip().lower()
        json_caption = track.extension.casefold() == "json3"
        if prefix.startswith((b"<!doctype html", b"<html")) or (
            not json_caption and prefix.startswith((b"{", b"["))
        ):
            raise ValueError("Subtitle URL returned an HTML or error response")
        actual_download.replace(destination)
    finally:
        # External downloaders and antivirus/indexing processes can retain a
        # Windows handle briefly after a failed transfer. A disposable partial
        # subtitle must never abort the media pipeline or mask the real error.
        try:
            shutil.rmtree(temporary_dir)
        except OSError:
            pass
    return destination


def has_cached_subtitle(track: CaptionTrack, destination_dir: Path) -> bool:
    if not track.extension:
        return False
    safe_language = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in track.language
    )
    destination = destination_dir / f"subtitle-{safe_language}.{track.extension.casefold()}"
    return (
        destination.is_file() and destination.stat().st_size > 0
    ) or _find_completed_orphan(destination) is not None


def _find_completed_orphan(destination: Path) -> Path | None:
    candidates = []
    for path in destination.parent.glob(destination.name + ".*"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.name.endswith((".part", ".ytdl")):
            continue
        candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_size, default=None)


def _find_actual_download(expected: Path, temporary_dir: Path) -> Path | None:
    if expected.is_file():
        return expected
    candidates = [
        path
        for path in temporary_dir.iterdir()
        if path.is_file()
        and path.stat().st_size > 0
        and not path.name.endswith((".part", ".ytdl"))
    ]
    return max(candidates, key=lambda path: path.stat().st_size, default=None)


def _download_with_requests(track: CaptionTrack, destination: Path) -> None:
    try:
        with requests.get(
            track.url,
            headers=track.http_headers,
            stream=True,
            allow_redirects=True,
            timeout=SUBTITLE_DOWNLOAD_TIMEOUT,
            verify=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
            allowed_json = track.extension and track.extension.casefold() == "json3"
            if content_type in {"text/html", "application/xhtml+xml"} or (
                content_type == "application/json" and not allowed_json
            ):
                raise ValueError(f"Subtitle URL returned non-subtitle content ({content_type})")
            written = 0
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_SUBTITLE_BYTES:
                        raise ValueError("Subtitle track exceeds the 10 MiB safety limit")
                    output.write(chunk)
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 429 or "429" in str(exc):
            raise SubtitleRateLimitError(
                "Caption provider returned HTTP 429 Too Many Requests"
            ) from exc
        raise ValueError(f"Subtitle download failed: {exc}") from exc


def _download_with_ytdlp(track: CaptionTrack, destination: Path) -> None:
    info = {
        "url": track.url,
        "ext": track.extension,
        "protocol": track.protocol,
        "impersonate": track.impersonate,
        "http_headers": track.http_headers or {},
    }
    options = {
        "quiet": True,
        "no_warnings": True,
        "retries": 0,
        "fragment_retries": 0,
        "skip_unavailable_fragments": False,
        "nopart": True,
        "logger": _QuietYtDlpLogger(),
    }
    try:
        with YoutubeDL(options) as ydl:
            downloader_class = get_suitable_downloader(info, ydl.params)
            if downloader_class is None:
                raise ValueError(f"No yt-dlp downloader supports subtitle protocol {track.protocol!r}")
            success, _ = downloader_class(ydl, ydl.params).download(str(destination), info)
            if not success:
                raise ValueError("yt-dlp could not retrieve the subtitle track")
    except Exception as exc:
        if "429" in str(exc):
            raise SubtitleRateLimitError(
                "Caption provider returned HTTP 429 Too Many Requests"
            ) from exc
        raise ValueError(f"Subtitle download failed: {exc}") from exc


def parse_subtitle(path: Path) -> list[SubtitleEntry]:
    if path.suffix.casefold() == ".json3":
        return _parse_json3(path)
    try:
        subtitles = pysubs2.load(str(path), encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Could not parse subtitle track {path.name}: {exc}") from exc

    entries: list[SubtitleEntry] = []
    for event in subtitles:
        text = " ".join(event.plaintext.split())
        if not text or event.end <= event.start:
            continue
        entries.append(SubtitleEntry(text=text, start=event.start / 1000.0, end=event.end / 1000.0))
    if not entries:
        raise ValueError(f"Subtitle track {path.name} contains no usable entries")
    return sorted(entries, key=lambda entry: (entry.start, entry.end))


def _parse_json3(path: Path) -> list[SubtitleEntry]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse subtitle track {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError(f"Could not parse subtitle track {path.name}: invalid json3 payload")

    entries: list[SubtitleEntry] = []
    for event in payload["events"]:
        if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
            continue
        try:
            start_ms = float(event["tStartMs"])
            duration_ms = float(event["dDurationMs"])
        except (KeyError, TypeError, ValueError):
            continue
        text = "".join(
            str(segment.get("utf8", ""))
            for segment in event["segs"]
            if isinstance(segment, dict)
        )
        text = " ".join(text.split())
        if not text or duration_ms <= 0:
            continue
        entries.append(
            SubtitleEntry(
                text=text,
                start=start_ms / 1000.0,
                end=(start_ms + duration_ms) / 1000.0,
            )
        )
    if not entries:
        raise ValueError(f"Subtitle track {path.name} contains no usable entries")
    return sorted(entries, key=lambda entry: (entry.start, entry.end))
