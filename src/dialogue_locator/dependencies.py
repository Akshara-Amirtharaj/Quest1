from __future__ import annotations

import shutil
from dataclasses import dataclass

from .errors import V0Error


@dataclass(frozen=True)
class ExternalTools:
    ffmpeg: str
    ffprobe: str


def require_external_tools() -> ExternalTools:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        names = ", ".join(missing)
        raise V0Error(
            f"Missing required external tool(s): {names}. Install FFmpeg and ensure "
            "both ffmpeg and ffprobe are on PATH."
        )
    return ExternalTools(ffmpeg=shutil.which("ffmpeg") or "", ffprobe=shutil.which("ffprobe") or "")
