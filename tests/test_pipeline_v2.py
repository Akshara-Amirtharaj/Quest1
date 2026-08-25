from pathlib import Path
from unittest.mock import patch

from dialogue_locator.dependencies import ExternalTools
from dialogue_locator.models import (
    CaptionCandidate,
    CaptionTrack,
    DialogueMatch,
    MediaInfo,
    ResolvedFrame,
    V1Result,
)
from dialogue_locator.caption_verification import CaptionVerification
from dialogue_locator.pipeline import run_v2
from dialogue_locator.subtitles import SubtitleRateLimitError


def test_rate_limited_caption_stops_requests_and_falls_back_to_v1(tmp_path: Path) -> None:
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"media")
    media = MediaInfo(
        duration=30.0,
        has_video=True,
        has_audio=True,
        embedded_subtitles=[],
        width=1280,
        height=720,
        video_codec="h264",
        audio_codec="aac",
        avg_frame_rate="25/1",
        real_frame_rate="25/1",
        video_time_base="1/90000",
        video_start_time=0.0,
        audio_start_time=0.0,
    )
    fallback = V1Result(
        "https://example.test/watch",
        media_path,
        "target words",
        DialogueMatch("target words", 3.0, 4.0, "exact", 100.0),
        ResolvedFrame(1, 270000, "1/90000", 3.0, tmp_path / "frame.png"),
        "base.en",
        audio_processed_seconds=30.0,
    )
    metadata = {
        "subtitles": {
            "en": [
                {"ext": "vtt", "url": "https://example.test/limited.vtt"},
                {"ext": "srt", "url": "https://example.test/limited.srt"},
            ]
        }
    }

    with (
        patch("dialogue_locator.pipeline.require_external_tools", return_value=ExternalTools("ffmpeg", "ffprobe")),
        patch("dialogue_locator.pipeline.acquire_media", return_value=(media_path, metadata)) as acquire,
        patch("dialogue_locator.pipeline.inspect_media", return_value=media) as inspect,
        patch(
            "dialogue_locator.pipeline.download_subtitle",
            side_effect=SubtitleRateLimitError("HTTP 429"),
        ) as download,
        patch("dialogue_locator.pipeline._localize_full_audio", return_value=fallback) as full_audio,
    ):
        result = run_v2(
            "https://example.test/watch",
            "target words",
            tmp_path / "work",
            tmp_path / "output",
            tmp_path / "models",
            language="en",
        )

    assert result.localization_source == "asr"
    assert result.verification_source == "asr"
    assert result.audio_processed_seconds == 30.0
    acquire.assert_called_once()
    inspect.assert_called_once()
    download.assert_called_once()
    full_audio.assert_called_once()


def test_failed_manual_vtt_retries_manual_srt_before_automatic(tmp_path: Path) -> None:
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"media")
    media = MediaInfo(
        duration=19.0,
        has_video=True,
        has_audio=True,
        embedded_subtitles=[],
        width=320,
        height=240,
        video_codec="h264",
        audio_codec="aac",
        avg_frame_rate="15/1",
        real_frame_rate="15/1",
        video_time_base="1/1000",
        video_start_time=0.0,
        audio_start_time=0.0,
    )
    metadata = {
        "subtitles": {
            "en": [
                {"ext": "vtt", "url": "https://example.test/manual.vtt"},
                {"ext": "srt", "url": "https://example.test/manual.srt"},
            ]
        },
        "automatic_captions": {
            "en": [{"ext": "vtt", "url": "https://example.test/automatic.vtt"}]
        },
    }
    subtitle_path = tmp_path / "manual.srt"
    subtitle_path.write_text("valid", encoding="utf-8")
    candidate = CaptionCandidate("target words", 1.0, 2.0, "exact", 100.0, "en", "manual")
    match = DialogueMatch("target words", 1.1, 1.8, "exact", 100.0)
    verification = CaptionVerification(match, 5.0, candidate)
    frame = ResolvedFrame(2, 1133, "1/1000", 1.133, tmp_path / "frame.png")
    downloaded_urls: list[str] = []

    def download(track: CaptionTrack, _: Path) -> Path:
        url = track.url
        downloaded_urls.append(url)
        if url.endswith(".vtt"):
            raise ValueError("429 Too Many Requests")
        return subtitle_path

    with (
        patch("dialogue_locator.pipeline.require_external_tools", return_value=ExternalTools("ffmpeg", "ffprobe")),
        patch("dialogue_locator.pipeline.acquire_media", return_value=(media_path, metadata)),
        patch("dialogue_locator.pipeline.inspect_media", return_value=media),
        patch("dialogue_locator.pipeline.download_subtitle", side_effect=download),
        patch("dialogue_locator.pipeline.parse_subtitle", return_value=[]) as parse,
        patch("dialogue_locator.pipeline.find_caption_candidates", return_value=[candidate]),
        patch("dialogue_locator.pipeline.verify_caption_candidates", return_value=(verification, 5.0)),
        patch("dialogue_locator.pipeline.resolve_frame_at_timestamp", return_value=frame),
        patch("dialogue_locator.pipeline._localize_full_audio") as full_audio,
    ):
        result = run_v2(
            "https://example.test/watch",
            "target words",
            tmp_path / "work",
            tmp_path / "output",
            tmp_path / "models",
            language="en",
        )

    assert downloaded_urls == [
        "https://example.test/manual.vtt",
        "https://example.test/manual.srt",
    ]
    assert parse.call_count == 1
    assert result.localization_source == "caption"
    assert result.verification_source == "asr"
    full_audio.assert_not_called()
