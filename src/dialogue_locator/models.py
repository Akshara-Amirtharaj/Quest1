from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def format_timestamp(seconds: float | None) -> str | None:
    """Format media seconds as a stable HH:MM:SS.mmm display value."""
    if seconds is None:
        return None
    total_milliseconds = max(0, round(seconds * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


@dataclass(frozen=True)
class CaptionTrack:
    language: str
    name: str | None = None
    extension: str | None = None
    url: str | None = None
    protocol: str | None = None
    impersonate: bool | str | None = None
    http_headers: dict[str, str] | None = None


@dataclass(frozen=True)
class SubtitleEntry:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class CaptionCandidate:
    text: str
    start: float
    end: float
    match_type: str
    score: float
    language: str
    caption_source: str


@dataclass(frozen=True)
class CaptionInventory:
    platform_subtitles: list[CaptionTrack] = field(default_factory=list)
    automatic_captions: list[CaptionTrack] = field(default_factory=list)

    @property
    def available_languages(self) -> list[str]:
        return sorted(
            {track.language for track in self.platform_subtitles + self.automatic_captions}
        )


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None
    has_video: bool
    has_audio: bool
    embedded_subtitles: list[dict[str, Any]]
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    avg_frame_rate: str | None
    real_frame_rate: str | None
    video_time_base: str | None
    video_start_time: float | None
    audio_start_time: float | None


@dataclass(frozen=True)
class SampleFrame:
    index: int
    pts: int | None
    time_base: str | None
    timestamp: float | None
    path: Path


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True)
class Transcription:
    text: str
    words: list[TranscriptWord]
    language: str | None
    language_probability: float | None
    alignment_source: str = "faster-whisper"
    precision_fallback_reason: str | None = None


@dataclass(frozen=True)
class DialogueMatch:
    matched_text: str
    start: float
    end: float
    match_type: str
    score: float


@dataclass(frozen=True)
class ResolvedFrame:
    index: int
    pts: int
    time_base: str
    timestamp: float
    path: Path


@dataclass(frozen=True)
class CandidateVideoFrame:
    index: int
    pts: int
    time_base: str
    timestamp: float
    image: Any


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class OCRTextMatch:
    matched_text: str
    match_type: str
    score: float


@dataclass(frozen=True)
class V1Result:
    source_url: str
    media_path: Path
    query: str
    match: DialogueMatch
    frame: ResolvedFrame
    model: str
    localization_source: str = "asr"
    verification_source: str = "asr"
    audio_processed_seconds: float | None = None
    caption_matched_text: str | None = None
    caption_match_type: str | None = None
    caption_match_score: float | None = None
    evidence_conflict: bool = False
    occurrences: tuple[DialogueMatch, ...] = ()
    transcription_language: str | None = None
    alignment_source: str = "faster-whisper"
    precision_mode: str = "default"
    precision_fallback_reason: str | None = None
    transcript_cache_hit: bool = False
    media_metadata_cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        from .confidence import assess_confidence

        confidence = assess_confidence(
            localization_source=self.localization_source,
            verification_source=self.verification_source,
            match_type=self.match.match_type,
            match_score=self.match.score,
            caption_match_type=self.caption_match_type,
            caption_match_score=self.caption_match_score,
            ocr_match_type=getattr(self, "ocr_match_type", None),
            ocr_match_score=getattr(self, "ocr_match_score", None),
            evidence_conflict=self.evidence_conflict,
        )
        return {
            "source_url": self.source_url,
            "media_path": str(self.media_path),
            "query": self.query,
            "matched_text": self.match.matched_text,
            "dialogue_start": self.match.start,
            "dialogue_start_hms": format_timestamp(self.match.start),
            "dialogue_end": self.match.end,
            "dialogue_end_hms": format_timestamp(self.match.end),
            "frame_index": self.frame.index,
            "frame_pts": self.frame.pts,
            "frame_time_base": self.frame.time_base,
            "frame_timestamp": self.frame.timestamp,
            "frame_timestamp_hms": format_timestamp(self.frame.timestamp),
            "frame_path": str(self.frame.path),
            "match_type": self.match.match_type,
            "match_score": self.match.score,
            "asr_model": self.model,
            "localization_source": self.localization_source,
            "verification_source": self.verification_source,
            "caption_matched_text": self.caption_matched_text,
            "caption_match_type": self.caption_match_type,
            "caption_match_score": self.caption_match_score,
            "evidence_conflict": self.evidence_conflict,
            "confidence": confidence.category,
            "confidence_reason": confidence.reason,
            "occurrence_count": len(self.occurrences) if self.occurrences else 1,
            "transcription_language": self.transcription_language,
            "alignment_source": self.alignment_source,
            "precision_mode": self.precision_mode,
            "precision_fallback_reason": self.precision_fallback_reason,
            "transcript_cache_hit": self.transcript_cache_hit,
            "media_metadata_cache_hit": self.media_metadata_cache_hit,
            "audio_processed_seconds": self.audio_processed_seconds,
            "audio_processed_hms": format_timestamp(self.audio_processed_seconds),
        }


@dataclass(frozen=True)
class V3Result(V1Result):
    frame_match_type: str = "spoken_dialogue"
    ocr_processed_frames: int = 0
    ocr_model: str | None = None
    ocr_matched_text: str | None = None
    ocr_match_type: str | None = None
    ocr_match_score: float | None = None
    ocr_cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "frame_match_type": self.frame_match_type,
                "ocr_processed_frames": self.ocr_processed_frames,
                "ocr_model": self.ocr_model,
                "ocr_matched_text": self.ocr_matched_text,
                "ocr_match_type": self.ocr_match_type,
                "ocr_match_score": self.ocr_match_score,
                "ocr_cache_hits": self.ocr_cache_hits,
            }
        )
        return result


@dataclass(frozen=True)
class V0Result:
    source_url: str
    media_path: Path
    media: MediaInfo
    captions: CaptionInventory
    sample_frame: SampleFrame
    metadata_cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "media_path": str(self.media_path),
            "duration": self.media.duration,
            "duration_hms": format_timestamp(self.media.duration),
            "has_video": self.media.has_video,
            "has_audio": self.media.has_audio,
            "embedded_subtitles": self.media.embedded_subtitles,
            "platform_subtitles": [asdict(track) for track in self.captions.platform_subtitles],
            "automatic_captions": [asdict(track) for track in self.captions.automatic_captions],
            "available_caption_languages": self.captions.available_languages,
            "width": self.media.width,
            "height": self.media.height,
            "video_codec": self.media.video_codec,
            "audio_codec": self.media.audio_codec,
            "avg_frame_rate": self.media.avg_frame_rate,
            "real_frame_rate": self.media.real_frame_rate,
            "video_time_base": self.media.video_time_base,
            "video_start_time": self.media.video_start_time,
            "audio_start_time": self.media.audio_start_time,
            "sample_frame_index": self.sample_frame.index,
            "sample_frame_pts": self.sample_frame.pts,
            "sample_frame_time_base": self.sample_frame.time_base,
            "sample_frame_timestamp": self.sample_frame.timestamp,
            "sample_frame_timestamp_hms": format_timestamp(self.sample_frame.timestamp),
            "sample_frame_path": str(self.sample_frame.path),
            "metadata_cache_hit": self.metadata_cache_hit,
        }
