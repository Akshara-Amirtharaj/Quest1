from __future__ import annotations

import subprocess
from pathlib import Path

from .errors import V0Error


def extract_speech_audio(
    media_path: Path,
    audio_path: Path,
    ffmpeg: str = "ffmpeg",
    start_time: float | None = None,
    duration: float | None = None,
) -> Path:
    try:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise V0Error(f"Could not create temporary audio directory: {exc}") from exc

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(media_path),
    ]
    if start_time is not None:
        command.extend(["-ss", f"{start_time:.6f}"])
    if duration is not None:
        command.extend(["-t", f"{duration:.6f}"])
    command.extend(
        [
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise V0Error(f"Could not start FFmpeg for audio extraction: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown FFmpeg error"
        raise V0Error(f"Audio extraction failed: {detail}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise V0Error("FFmpeg completed but did not create usable audio.")
    return audio_path.resolve()
