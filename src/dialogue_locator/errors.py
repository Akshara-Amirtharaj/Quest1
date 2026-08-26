from __future__ import annotations


class V0Error(RuntimeError):
    """Expected, user-facing operational failure with stable CLI provenance."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        stage: str | None = None,
    ) -> None:
        inferred_code, inferred_stage = _classify_legacy_message(message)
        self.code = code or inferred_code
        self.stage = stage or inferred_stage
        super().__init__(message)

    def to_dict(self) -> dict[str, str]:
        return {
            "error": str(self),
            "error_code": self.code,
            "error_stage": self.stage,
        }


def _classify_legacy_message(message: str) -> tuple[str, str]:
    """Classify existing domain messages without rewriting frozen algorithms."""
    normalized = message.casefold()
    if "video url must be" in normalized:
        return "invalid_url", "input"
    if "does not contain a video stream" in normalized or "no decodable video stream" in normalized:
        return "missing_video_stream", "media_inspection"
    if "does not contain an audio stream" in normalized:
        return "missing_audio_stream", "media_inspection"
    if "dialogue not found" in normalized or "transcription contains no matchable words" in normalized:
        return "dialogue_not_found", "matching"
    if "target dialogue" in normalized:
        return "invalid_dialogue", "input"
    if "ffprobe" in normalized or "invalid media" in normalized:
        return "invalid_media", "media_inspection"
    if "acquir" in normalized or "download" in normalized or "media cache directory" in normalized:
        return "acquisition_failed", "acquisition"
    if (
        "output directory" in normalized
        or "write final frame" in normalized
        or "frame artifact" in normalized
    ):
        return "frame_output_failed", "frame_output"
    if "pyav" in normalized or "decoded video frame" in normalized:
        return "video_decode_failed", "frame_resolution"
    if "audio extraction" in normalized or "temporary audio" in normalized:
        return "audio_processing_failed", "audio_extraction"
    if "speech recognition" in normalized or "speech model" in normalized:
        return "asr_failed", "speech_recognition"
    return "processing_failed", "pipeline"
