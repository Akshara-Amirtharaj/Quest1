from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dialogue_locator.models import CaptionInventory, CaptionTrack
from dialogue_locator.subtitles import (
    SubtitleRateLimitError,
    download_subtitle,
    has_cached_subtitle,
    parse_subtitle,
    select_caption_tracks,
)


def test_keeps_supported_format_fallbacks_per_language_and_source() -> None:
    inventory = CaptionInventory(
        platform_subtitles=[
            CaptionTrack("en-US", extension="json3", url="https://example.test/en.json3"),
            CaptionTrack("en-US", extension="srt", url="https://example.test/en.srt"),
            CaptionTrack("en-US", extension="ttml", url="https://example.test/en.ttml"),
            CaptionTrack("fr", extension="vtt", url="https://example.test/fr.vtt"),
        ],
        automatic_captions=[
            CaptionTrack("en", extension="vtt", url="https://example.test/auto-en.vtt"),
            CaptionTrack("en", extension="srt", url="https://example.test/auto-en.srt"),
        ],
    )

    selected = select_caption_tracks(inventory, "en")

    assert [(source, track.language, track.extension) for source, track in selected] == [
        ("manual", "en-US", "json3"),
        ("manual", "en-US", "srt"),
        ("manual", "en-US", "ttml"),
        ("automatic", "en", "vtt"),
        ("automatic", "en", "srt"),
    ]


def test_no_language_hint_conservatively_keeps_available_languages() -> None:
    inventory = CaptionInventory(
        platform_subtitles=[
            CaptionTrack("en", extension="vtt", url="https://example.test/en.vtt"),
            CaptionTrack("fr", extension="vtt", url="https://example.test/fr.vtt"),
        ]
    )
    assert [track.language for _, track in select_caption_tracks(inventory)] == ["en", "fr"]


def test_parse_srt_normalizes_entries(tmp_path: Path) -> None:
    subtitle = tmp_path / "sample.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nHello,   world!\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nSecond\nline\n",
        encoding="utf-8",
    )

    entries = parse_subtitle(subtitle)

    assert [(entry.text, entry.start, entry.end) for entry in entries] == [
        ("Hello, world!", 1.0, 2.5),
        ("Second line", 2.5, 4.0),
    ]


def test_parse_youtube_json3_entries(tmp_path: Path) -> None:
    subtitle = tmp_path / "sample.json3"
    subtitle.write_text(
        '{"events":['
        '{"tStartMs":1000,"dDurationMs":1500,"segs":['
        '{"utf8":"Hello, "},{"utf8":"world!"}]},'
        '{"tStartMs":2500,"dDurationMs":500,"segs":[{"utf8":"Second line"}]}'
        ']}' ,
        encoding="utf-8",
    )

    entries = parse_subtitle(subtitle)

    assert [(entry.text, entry.start, entry.end) for entry in entries] == [
        ("Hello, world!", 1.0, 2.5),
        ("Second line", 2.5, 3.0),
    ]


def _response(content_type: str, chunks: list[bytes]) -> Mock:
    response = Mock()
    response.headers = {"Content-Type": content_type}
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    return response


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("text/html; charset=utf-8", b"<html>rate limited</html>"),
        ("application/octet-stream", b"<!doctype html><title>expired</title>"),
        ("application/json", b'{"error":"expired"}'),
    ],
)
def test_download_rejects_html_and_error_responses(
    tmp_path: Path, content_type: str, body: bytes
) -> None:
    track = CaptionTrack("en", extension="vtt", url="https://example.test/captions")
    with patch("dialogue_locator.subtitles.requests.get", return_value=_response(content_type, [body])):
        with pytest.raises(ValueError, match="non-subtitle|HTML or error"):
            download_subtitle(track, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_http_error_is_reported_without_leaving_partial_file(tmp_path: Path) -> None:
    response = _response("text/vtt", [])
    response.raise_for_status.side_effect = __import__("requests").HTTPError("429 Too Many Requests")
    track = CaptionTrack("en", extension="vtt", url="https://example.test/captions")

    with patch("dialogue_locator.subtitles.requests.get", return_value=response):
        with pytest.raises(SubtitleRateLimitError, match="HTTP 429"):
            download_subtitle(track, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_yt_dlp_metadata_routes_through_its_downloader(tmp_path: Path) -> None:
    track = CaptionTrack(
        "en",
        extension="vtt",
        url="https://example.test/captions",
        protocol="https",
        impersonate=True,
    )

    def write_valid_subtitle(_: CaptionTrack, destination: Path) -> None:
        destination.write_bytes(b"WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n")

    with (
        patch("dialogue_locator.subtitles._download_with_ytdlp", side_effect=write_valid_subtitle) as ytdlp,
        patch("dialogue_locator.subtitles.requests.get") as requests_get,
    ):
        result = download_subtitle(track, tmp_path)

    assert result.read_bytes().startswith(b"WEBVTT")
    ytdlp.assert_called_once()
    requests_get.assert_not_called()


def test_locked_failed_partial_does_not_mask_download_error(tmp_path: Path) -> None:
    track = CaptionTrack(
        "en",
        extension="vtt",
        url="https://example.test/captions",
        protocol="m3u8_native",
    )

    with (
        patch(
            "dialogue_locator.subtitles._download_with_ytdlp",
            side_effect=ValueError("transfer failed"),
        ),
        patch("pathlib.Path.unlink", side_effect=PermissionError("file is locked")),
    ):
        with pytest.raises(ValueError, match="transfer failed"):
            download_subtitle(track, tmp_path)


def test_concurrent_completed_cache_wins_over_failed_request(tmp_path: Path) -> None:
    track = CaptionTrack(
        "en",
        extension="json3",
        url="https://example.test/captions",
        protocol="m3u8_native",
    )
    cached = tmp_path / "subtitle-en.json3"

    def concurrent_completion(_: CaptionTrack, __: Path) -> None:
        cached.write_text('{"events": []}', encoding="utf-8")
        raise ValueError("provider request failed")

    with patch(
        "dialogue_locator.subtitles._download_with_ytdlp",
        side_effect=concurrent_completion,
    ):
        result = download_subtitle(track, tmp_path)

    assert result == cached


def test_adopts_completed_file_left_by_ytdlp_filename_rewrite(tmp_path: Path) -> None:
    track = CaptionTrack("en", extension="json3", url="https://example.test/captions")
    rewritten = tmp_path / "subtitle-en.json3.download-id"
    rewritten.write_text('{"events": []}', encoding="utf-8")

    assert has_cached_subtitle(track, tmp_path)
    result = download_subtitle(track, tmp_path)

    assert result == tmp_path / "subtitle-en.json3"
    assert result.read_text(encoding="utf-8") == '{"events": []}'
    assert not rewritten.exists()
