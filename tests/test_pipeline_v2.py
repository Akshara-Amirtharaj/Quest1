from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dialogue_locator.dependencies import ExternalTools
from dialogue_locator.acquisition import SourceInspection
from dialogue_locator.config import V2Config
from dialogue_locator.errors import V0Error
from dialogue_locator.matching import find_dialogue
from dialogue_locator.models import (
    CaptionCandidate,
    CaptionInventory,
    CaptionTrack,
    DialogueMatch,
    MediaInfo,
    ResolvedFrame,
    TranscriptWord,
    Transcription,
    V1Result,
)
from dialogue_locator.caption_verification import CaptionVerification
from dialogue_locator.pipeline import _select_relevant_caption_tracks, run_v2
from dialogue_locator.subtitles import SubtitleRateLimitError


def test_implicit_caption_language_prefers_original_source_track() -> None:
    inventory = CaptionInventory(
        automatic_captions=[
            CaptionTrack("aa", "Afar", "json3", "https://example.test/aa.json3"),
            CaptionTrack("en", "English", "json3", "https://example.test/en.json3"),
            CaptionTrack(
                "en-orig",
                "English (Original)",
                "json3",
                "https://example.test/en-orig.json3",
            ),
            CaptionTrack("fr", "French", "json3", "https://example.test/fr.json3"),
        ]
    )

    selected = _select_relevant_caption_tracks(
        inventory,
        requested_language=None,
        metadata={"language": "en"},
    )

    assert [(source, track.language) for source, track in selected] == [
        ("automatic", "en-orig")
    ]


def test_explicit_caption_language_keeps_existing_language_selection() -> None:
    inventory = CaptionInventory(
        automatic_captions=[
            CaptionTrack("en", "English", "json3", "https://example.test/en.json3"),
            CaptionTrack("fr", "French", "json3", "https://example.test/fr.json3"),
        ]
    )

    selected = _select_relevant_caption_tracks(
        inventory,
        requested_language="fr",
        metadata={"language": "en"},
    )

    assert [(source, track.language) for source, track in selected] == [
        ("automatic", "fr")
    ]


def test_explicit_base_language_prefers_original_variant() -> None:
    inventory = CaptionInventory(
        automatic_captions=[
            CaptionTrack("en", "English", "vtt", "https://example.test/en.vtt"),
            CaptionTrack(
                "en-orig",
                "English (Original)",
                "json3",
                "https://example.test/en-orig.json3",
            ),
        ]
    )

    selected = _select_relevant_caption_tracks(
        inventory,
        requested_language="en",
        metadata={"language": "en"},
    )

    assert [(source, track.language, track.extension) for source, track in selected] == [
        ("automatic", "en-orig", "json3")
    ]


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


def test_no_captions_selects_audio_only_path_before_full_media(tmp_path: Path) -> None:
    marker = V1Result(
        "https://example.test/watch",
        tmp_path / "audio.m4a",
        "target words",
        DialogueMatch("target words", 3.0, 4.0, "exact", 100.0),
        ResolvedFrame(1, 3000, "1/1000", 3.0, tmp_path / "frame.png"),
        "base.en",
    )
    source = SourceInspection(
        {"id": "example", "subtitles": {}, "automatic_captions": {}},
        "https://cdn.example/video.mp4",
        {},
    )
    with (
        patch(
            "dialogue_locator.pipeline.require_external_tools",
            return_value=ExternalTools("ffmpeg", "ffprobe"),
        ),
        patch("dialogue_locator.pipeline.inspect_source", return_value=source),
        patch(
            "dialogue_locator.pipeline._run_audio_only_localization",
            return_value=marker,
        ) as audio_only,
        patch("dialogue_locator.pipeline.acquire_media") as full_media,
    ):
        result = run_v2(
            "https://example.test/watch",
            "target words",
            tmp_path / "work",
            tmp_path / "output",
            tmp_path / "models",
            language="en",
        )

    assert result is marker
    audio_only.assert_called_once()
    assert audio_only.call_args.kwargs["source"].metadata is source.metadata
    full_media.assert_not_called()


def test_caption_inventory_keeps_existing_full_media_flow(tmp_path: Path) -> None:
    source = SourceInspection(
        {
            "subtitles": {
                "en": [{"ext": "vtt", "url": "https://example.test/captions.vtt"}]
            }
        },
        "https://cdn.example/video.mp4",
        {},
    )
    with (
        patch(
            "dialogue_locator.pipeline.require_external_tools",
            return_value=ExternalTools("ffmpeg", "ffprobe"),
        ),
        patch("dialogue_locator.pipeline.inspect_source", return_value=source),
        patch(
            "dialogue_locator.pipeline._caption_candidates_for_source",
            return_value=[Mock()],
        ),
        patch(
            "dialogue_locator.pipeline.acquire_media",
            side_effect=V0Error("expected full-media sentinel"),
        ) as full_media,
        patch("dialogue_locator.pipeline._run_audio_only_localization") as audio_only,
    ):
        with pytest.raises(V0Error, match="full-media sentinel"):
            run_v2(
                "https://example.test/watch",
                "target words",
                tmp_path / "work",
                tmp_path / "output",
                tmp_path / "models",
                language="en",
            )

    full_media.assert_called_once()
    audio_only.assert_not_called()


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


def test_caption_path_returns_precision_fallback_provenance(tmp_path: Path) -> None:
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"media")
    media = MediaInfo(
        duration=30.0,
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
            "en": [{"ext": "vtt", "url": "https://example.test/manual.vtt"}]
        }
    }
    candidate = CaptionCandidate(
        "How's it looking, Barley?",
        10.0,
        12.0,
        "exact",
        100.0,
        "en",
        "manual",
    )

    class Transcriber:
        def __init__(self, transcription: Transcription) -> None:
            self.transcription = transcription
            self.calls = 0
            self.last_cache_hit = False
            self.last_transcription: Transcription | None = None

        def __call__(self, _: Path) -> Transcription:
            self.calls += 1
            self.last_transcription = self.transcription
            return self.transcription

    base = Transcriber(
        Transcription(
            "As it looking, Bali?",
            [
                TranscriptWord("As", 0.0, 0.5),
                TranscriptWord("it", 0.5, 0.7),
                TranscriptWord("looking,", 0.7, 1.0),
                TranscriptWord("Bali?", 1.0, 1.3),
            ],
            "en",
            0.9,
        )
    )
    precision = Transcriber(
        Transcription(
            "How's it looking, Barley?",
            [
                TranscriptWord("How's", 0.0, 0.4),
                TranscriptWord("it", 0.4, 0.6),
                TranscriptWord("looking,", 0.6, 0.9),
                TranscriptWord("Barley?", 0.9, 1.2),
            ],
            "en",
            0.99,
        )
    )

    def verify(*args, **_kwargs):
        transcriber = args[7]
        transcription = transcriber(tmp_path / "same-window.wav")
        relative = find_dialogue("How's it looking, Barley?", transcription.words, 85.0)
        return CaptionVerification(relative, 12.0, candidate), 12.0

    frame = ResolvedFrame(2, 10000, "1/1000", 10.0, tmp_path / "frame.png")
    with (
        patch(
            "dialogue_locator.pipeline.require_external_tools",
            return_value=ExternalTools("ffmpeg", "ffprobe"),
        ),
        patch("dialogue_locator.pipeline.acquire_media", return_value=(media_path, metadata)),
        patch("dialogue_locator.pipeline.inspect_media", return_value=media),
        patch(
            "dialogue_locator.pipeline._caption_candidates_for_source",
            side_effect=lambda source, *_args: [candidate] if source == "manual" else [],
        ),
        patch("dialogue_locator.pipeline._create_transcriber", side_effect=[base, precision]),
        patch("dialogue_locator.pipeline.verify_caption_candidates", side_effect=verify),
        patch("dialogue_locator.pipeline.resolve_frame_at_timestamp", return_value=frame),
    ):
        result = run_v2(
            "https://example.test/watch",
            "How's it looking, Barley?",
            tmp_path / "work",
            tmp_path / "output",
            tmp_path / "models",
            language="en",
            config=V2Config(),
        )

    assert base.calls == 1
    assert precision.calls == 1
    assert result.match.matched_text == "How's it looking, Barley?"
    assert result.asr_model_used == "distil-large-v3"
    assert result.precision_fallback_used is True
    assert result.precision_scope == "candidate_window"
    assert result.base_match_score is not None and result.base_match_score < 85.0
    assert result.precision_match_score == 100.0
    assert result.precision_trigger_threshold == 45.0
    assert result.precision_fallback_eligible is True
    assert result.precision_fallback_skip_reason is None


def test_all_low_score_caption_candidates_continue_to_full_asr_fallback(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"media")
    media = MediaInfo(
        duration=30.0,
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
            "en": [{"ext": "vtt", "url": "https://example.test/manual.vtt"}]
        }
    }
    candidate = CaptionCandidate(
        "wrong caption",
        10.0,
        12.0,
        "fuzzy",
        90.0,
        "en",
        "manual",
    )

    class BaseTranscriber:
        last_cache_hit = False

        def __init__(self) -> None:
            self.calls = 0
            self.last_transcription: Transcription | None = None

        def __call__(self, _: Path) -> Transcription:
            self.calls += 1
            self.last_transcription = Transcription(
                "Completely unrelated window content.",
                [
                    TranscriptWord("Completely", 0.0, 0.2),
                    TranscriptWord("unrelated", 0.2, 0.4),
                    TranscriptWord("window", 0.4, 0.6),
                    TranscriptWord("content.", 0.6, 0.8),
                ],
                "en",
                0.99,
            )
            return self.last_transcription

    base = BaseTranscriber()

    def verify(*args, **_kwargs):
        transcriber = args[7]
        transcription = transcriber(tmp_path / "same-window.wav")
        assert transcriber.precision_fallback_used is False
        assert transcriber.precision_fallback_skip_reason == (
            "base_match_score_below_precision_trigger"
        )
        with pytest.raises(V0Error, match="Dialogue not found"):
            find_dialogue(
                "How's it looking, Barley?",
                transcription.words,
                85.0,
            )
        return None, 6.0

    fallback = V1Result(
        "https://example.test/watch",
        media_path,
        "How's it looking, Barley?",
        DialogueMatch("How's it looking, Barley?", 20.0, 21.0, "exact", 100.0),
        ResolvedFrame(1, 20000, "1/1000", 20.0, tmp_path / "frame.png"),
        "base.en",
        audio_processed_seconds=30.0,
    )
    with (
        patch(
            "dialogue_locator.pipeline.require_external_tools",
            return_value=ExternalTools("ffmpeg", "ffprobe"),
        ),
        patch("dialogue_locator.pipeline.acquire_media", return_value=(media_path, metadata)),
        patch("dialogue_locator.pipeline.inspect_media", return_value=media),
        patch(
            "dialogue_locator.pipeline._caption_candidates_for_source",
            side_effect=lambda source, *_args: [candidate] if source == "manual" else [],
        ),
        patch("dialogue_locator.pipeline._create_transcriber", side_effect=[base]),
        patch("dialogue_locator.pipeline.verify_caption_candidates", side_effect=verify),
        patch(
            "dialogue_locator.pipeline._localize_full_audio",
            return_value=fallback,
        ) as full_audio,
    ):
        result = run_v2(
            "https://example.test/watch",
            "How's it looking, Barley?",
            tmp_path / "work",
            tmp_path / "output",
            tmp_path / "models",
            language="en",
            config=V2Config(),
        )

    assert base.calls == 1
    full_audio.assert_called_once()
    assert result.localization_source == "asr"
    assert result.audio_processed_seconds == 36.0
