from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .errors import V0Error
from .models import MediaInfo


def parse_rational(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_stream(streams: list[Mapping[str, Any]], codec_type: str) -> Mapping[str, Any] | None:
    return next((stream for stream in streams if stream.get("codec_type") == codec_type), None)


def parse_ffprobe(data: Mapping[str, Any]) -> MediaInfo:
    raw_streams = data.get("streams")
    streams = [stream for stream in raw_streams if isinstance(stream, Mapping)] if isinstance(raw_streams, list) else []
    video = _first_stream(streams, "video")
    audio = _first_stream(streams, "audio")
    subtitle_streams = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    format_data = data.get("format") if isinstance(data.get("format"), Mapping) else {}

    embedded = []
    for stream in subtitle_streams:
        tags = stream.get("tags") if isinstance(stream.get("tags"), Mapping) else {}
        embedded.append(
            {
                "index": _int_or_none(stream.get("index")),
                "codec": stream.get("codec_name"),
                "language": tags.get("language"),
                "title": tags.get("title"),
            }
        )

    return MediaInfo(
        duration=_float_or_none(format_data.get("duration")),
        has_video=video is not None,
        has_audio=audio is not None,
        embedded_subtitles=embedded,
        width=_int_or_none(video.get("width")) if video else None,
        height=_int_or_none(video.get("height")) if video else None,
        video_codec=str(video.get("codec_name")) if video and video.get("codec_name") else None,
        audio_codec=str(audio.get("codec_name")) if audio and audio.get("codec_name") else None,
        avg_frame_rate=str(video.get("avg_frame_rate")) if video and video.get("avg_frame_rate") else None,
        real_frame_rate=str(video.get("r_frame_rate")) if video and video.get("r_frame_rate") else None,
        video_time_base=str(video.get("time_base")) if video and video.get("time_base") else None,
        video_start_time=_float_or_none(video.get("start_time")) if video else None,
        audio_start_time=_float_or_none(audio.get("start_time")) if audio else None,
    )


def inspect_media(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    command = [
        ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise V0Error(f"Could not start ffprobe: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise V0Error(f"ffprobe failed for {path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise V0Error("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise V0Error("ffprobe returned invalid media metadata")
    return parse_ffprobe(payload)
