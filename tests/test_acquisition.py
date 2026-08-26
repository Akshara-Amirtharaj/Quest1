from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from dialogue_locator.acquisition import (
    AUDIO_FORMAT_SELECTOR,
    DOWNLOAD_RETRIES,
    FRAGMENT_RETRIES,
    SOCKET_TIMEOUT_SECONDS,
    DOWNLOAD_CHUNK_SIZE,
    FORMAT_SELECTOR,
    MEDIA_URL_CACHE_NAMESPACE,
    acquire_audio_only,
    acquire_media,
    download_direct_media,
    inspect_source,
    normalize_public_url,
    validate_public_url,
)
from dialogue_locator.cache import JsonFileCache
from yt_dlp.utils import DownloadError
from dialogue_locator.errors import V0Error
from dialogue_locator.pipeline import run_v0


def test_format_selector_prefers_bounded_height_then_falls_back() -> None:
    assert "height<=720" in FORMAT_SELECTOR
    assert FORMAT_SELECTOR.endswith("bv*+ba/b")


def test_metadata_inspection_never_downloads_media() -> None:
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)
    downloader.extract_info.return_value = {
        "id": "example",
        "requested_formats": [
            {
                "url": "https://cdn.example/video.mp4",
                "vcodec": "h264",
                "http_headers": {"Referer": "https://provider.example/"},
            },
            {"url": "https://cdn.example/audio.m4a", "vcodec": "none"},
        ],
    }
    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader) as ydl:
        inspected = inspect_source("https://provider.example/watch/1")

    assert inspected.video_url == "https://cdn.example/video.mp4"
    assert inspected.video_headers == {"Referer": "https://provider.example/"}
    assert ydl.call_args.args[0]["skip_download"] is True
    downloader.extract_info.assert_called_once_with(
        "https://provider.example/watch/1", download=False
    )


def test_audio_only_acquisition_selects_audio_and_rejects_partial(tmp_path: Path) -> None:
    final = tmp_path / "audio-only" / "Generic-example.m4a"
    partial = final.with_suffix(".m4a.part")
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)

    def download(*_args, **_kwargs):
        final.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial")
        final.write_bytes(b"audio")
        return {"id": "example", "filepath": str(final)}

    downloader.extract_info.side_effect = download
    downloader.prepare_filename.return_value = str(final)
    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader) as ydl:
        path, _ = acquire_audio_only("https://example.test/media", tmp_path)

    assert path == final.resolve()
    assert ydl.call_args.args[0]["format"] == AUDIO_FORMAT_SELECTOR
    assert partial.is_file()


def test_audio_only_reuses_inspected_stream_without_second_provider_request(
    tmp_path: Path,
) -> None:
    final = tmp_path / "audio-only" / "Odnoklassniki-example.m4a"
    inspected = {
        "id": "example",
        "extractor": "Odnoklassniki",
        "title": "Example",
        "formats": [
            {
                "format_id": "video",
                "url": "https://cdn.example/video.mp4",
                "vcodec": "h264",
                "acodec": "none",
                "tbr": 900,
            },
            {
                "format_id": "audio",
                "url": "https://cdn.example/audio.m4a",
                "vcodec": "none",
                "acodec": "aac",
                "abr": 128,
                "ext": "m4a",
            },
        ],
    }
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)

    def process(info: dict, *, download: bool) -> dict:
        assert download is True
        assert info["url"] == "https://cdn.example/audio.m4a"
        assert info["vcodec"] == "none"
        assert "formats" not in info
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"audio")
        return {**info, "filepath": str(final)}

    downloader.process_ie_result.side_effect = process
    downloader.prepare_filename.return_value = str(final)
    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader) as ydl:
        path, _ = acquire_audio_only(
            "https://ok.ru/video/example",
            tmp_path,
            source_metadata=inspected,
        )

    assert path == final.resolve()
    downloader.extract_info.assert_not_called()
    options = ydl.call_args.args[0]
    assert options["retries"] == DOWNLOAD_RETRIES
    assert options["fragment_retries"] == FRAGMENT_RETRIES
    assert options["socket_timeout"] == SOCKET_TIMEOUT_SECONDS
    assert options["concurrent_fragment_downloads"] == 1


def test_audio_only_refreshes_metadata_once_when_inspected_stream_expires(
    tmp_path: Path,
) -> None:
    final = tmp_path / "audio-only" / "Generic-example.m4a"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"audio")
    inspected = {
        "id": "example",
        "extractor": "Generic",
        "url": "https://cdn.example/expired.m4a",
        "vcodec": "none",
        "acodec": "aac",
    }
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)
    downloader.process_ie_result.side_effect = DownloadError("expired stream")
    downloader.extract_info.return_value = {"id": "example", "filepath": str(final)}
    downloader.prepare_filename.return_value = str(final)

    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader):
        path, _ = acquire_audio_only(
            "https://example.test/video/example",
            tmp_path,
            source_metadata=inspected,
        )

    assert path == final.resolve()
    downloader.extract_info.assert_called_once_with(
        "https://example.test/video/example", download=True
    )


@pytest.mark.parametrize("url", ["", "not a url", "file:///tmp/video.mp4", "ftp://example.test/a"])
def test_rejects_non_public_url_shapes(url: str) -> None:
    with pytest.raises(V0Error):
        validate_public_url(url)


def test_accepts_http_url_shape() -> None:
    assert validate_public_url("https://example.test/video") == "https://example.test/video"


def test_normalizes_markdown_link_and_escaped_url_characters() -> None:
    pasted = "[https://example.test/a\\_b](https://example.test/a\\_b?x=1\\&y=2)"

    assert normalize_public_url(pasted) == "https://example.test/a_b?x=1&y=2"
    assert validate_public_url(pasted) == "https://example.test/a_b?x=1&y=2"


def _response(content_type: str, chunks: list[bytes]) -> Mock:
    response = Mock()
    response.headers = {"Content-Type": content_type}
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    return response


def test_direct_fallback_streams_media_without_requiring_mp4_suffix(tmp_path: Path) -> None:
    response = _response("video/webm; charset=binary", [b"media-", b"bytes"])
    cache_dir = tmp_path / "nested" / "cache"
    with patch("dialogue_locator.acquisition.requests.get", return_value=response) as get:
        path, metadata = download_direct_media("https://example.test/watch?id=7", cache_dir)

    assert path.read_bytes() == b"media-bytes"
    assert path.suffix == ".media"
    assert metadata["extractor"] == "direct-http"
    get.assert_called_once_with(
        "https://example.test/watch?id=7",
        stream=True,
        allow_redirects=True,
        timeout=(15, 60),
        verify=True,
    )
    response.iter_content.assert_called_once_with(chunk_size=DOWNLOAD_CHUNK_SIZE)


def test_direct_fallback_rejects_html_provider_page(tmp_path: Path) -> None:
    response = _response("text/html; charset=utf-8", [b"<html>not media</html>"])
    with patch("dialogue_locator.acquisition.requests.get", return_value=response):
        with pytest.raises(V0Error, match="non-media response"):
            download_direct_media("https://example.test/watch/7", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_acquisition_uses_direct_fallback_after_ytdlp_failure(tmp_path: Path) -> None:
    response = _response("application/octet-stream", [b"direct-media"])
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)
    downloader.extract_info.side_effect = DownloadError("extractor failed")

    with (
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader),
        patch("dialogue_locator.acquisition.requests.get", return_value=response),
    ):
        path, metadata = acquire_media("https://example.test/media?id=7", tmp_path)

    assert path.read_bytes() == b"direct-media"
    assert metadata["extractor"] == "direct-http"


def test_acquisition_reuses_url_index_without_contacting_provider(tmp_path: Path) -> None:
    media_path = tmp_path / "Example-abc123.mp4"
    metadata = {
        "id": "abc123",
        "extractor": "Example",
        "webpage_url": "https://example.test/video/abc123",
        "filepath": str(media_path),
    }
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)

    def download(_: str, *, download: bool) -> dict:
        assert download is True
        media_path.write_bytes(b"cached media")
        return metadata

    downloader.extract_info.side_effect = download
    downloader.prepare_filename.return_value = str(media_path)

    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader):
        first_path, _ = acquire_media("https://example.test/video/abc123", tmp_path)
        second_path, second_metadata = acquire_media("https://example.test/video/abc123", tmp_path)

    assert first_path == media_path.resolve()
    assert second_path == media_path.resolve()
    assert second_metadata["media_cache_hit"] is True
    downloader.extract_info.assert_called_once()


def test_acquisition_adopts_legacy_provider_file_before_network(tmp_path: Path) -> None:
    media_path = tmp_path / "BitChute-XUGoMI86XNJq.mp4"
    media_path.write_bytes(b"previously downloaded media")

    with patch("dialogue_locator.acquisition.YoutubeDL") as youtube_dl:
        path, metadata = acquire_media(
            "https://www.bitchute.com/video/XUGoMI86XNJq",
            tmp_path,
        )

    assert path == media_path.resolve()
    assert metadata["media_cache_hit"] is True
    youtube_dl.assert_not_called()


def _failed_downloader(message: str = "provider failed") -> Mock:
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)
    downloader.extract_info.side_effect = DownloadError(message)
    return downloader


def test_stale_ytdl_without_final_media_is_never_adopted(tmp_path: Path) -> None:
    stale = tmp_path / "Odnoklassniki-248244667877.mp4.ytdl"
    stale.write_bytes(b"resume metadata")
    downloader = _failed_downloader()

    with (
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader),
        patch(
            "dialogue_locator.acquisition.download_direct_media",
            side_effect=V0Error("direct fallback failed"),
        ),
    ):
        with pytest.raises(V0Error) as raised:
            acquire_media("https://ok.example/video/248244667877", tmp_path)

    assert raised.value.code == "acquisition_failed"
    downloader.extract_info.assert_called_once()
    assert stale.read_bytes() == b"resume metadata"


def test_stale_part_without_final_media_is_never_adopted(tmp_path: Path) -> None:
    stale = tmp_path / "Odnoklassniki-248244667877.mp4.part"
    stale.write_bytes(b"partial media")
    downloader = _failed_downloader()

    with (
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader),
        patch(
            "dialogue_locator.acquisition.download_direct_media",
            side_effect=V0Error("direct fallback failed"),
        ),
    ):
        with pytest.raises(V0Error):
            acquire_media("https://ok.example/video/248244667877", tmp_path)

    downloader.extract_info.assert_called_once()
    assert stale.read_bytes() == b"partial media"


def test_zero_byte_final_media_is_not_adopted(tmp_path: Path) -> None:
    empty = tmp_path / "Odnoklassniki-248244667877.mp4"
    empty.touch()
    downloader = _failed_downloader()

    with (
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader),
        patch(
            "dialogue_locator.acquisition.download_direct_media",
            side_effect=V0Error("direct fallback failed"),
        ),
    ):
        with pytest.raises(V0Error):
            acquire_media("https://ok.example/video/248244667877", tmp_path)

    downloader.extract_info.assert_called_once()


def test_completed_media_wins_over_newer_stale_ytdl(tmp_path: Path) -> None:
    completed = tmp_path / "Odnoklassniki-248244667877.mp4"
    completed.write_bytes(b"complete media")
    stale = tmp_path / "Odnoklassniki-248244667877.mp4.ytdl"
    stale.write_bytes(b"newer resume metadata")
    stale.touch()

    with patch("dialogue_locator.acquisition.YoutubeDL") as youtube_dl:
        path, metadata = acquire_media(
            "https://ok.example/video/248244667877",
            tmp_path,
        )

    assert path == completed.resolve()
    assert metadata["media_cache_hit"] is True
    youtube_dl.assert_not_called()


def test_poisoned_url_index_cannot_promote_ytdl_to_completed_media(tmp_path: Path) -> None:
    url = "https://ok.example/video/248244667877"
    stale = tmp_path / "Odnoklassniki-248244667877.mp4.ytdl"
    stale.write_bytes(b"resume metadata")
    from dialogue_locator.acquisition import _media_url_cache_key

    JsonFileCache(tmp_path / ".quest1-cache").put(
        MEDIA_URL_CACHE_NAMESPACE,
        _media_url_cache_key(url),
        {"url": url, "filename": stale.name, "metadata": {}},
    )
    downloader = _failed_downloader()

    with (
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader),
        patch(
            "dialogue_locator.acquisition.download_direct_media",
            side_effect=V0Error("direct fallback failed"),
        ),
    ):
        with pytest.raises(V0Error):
            acquire_media(url, tmp_path)

    downloader.extract_info.assert_called_once()


def test_ytdlp_reported_temporary_path_is_not_trusted(tmp_path: Path) -> None:
    stale = tmp_path / "Odnoklassniki-248244667877.mp4.ytdl"
    stale.write_bytes(b"resume metadata")
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=False)
    downloader.extract_info.return_value = {
        "id": "248244667877",
        "extractor": "Odnoklassniki",
        "requested_downloads": [{"filepath": str(stale)}],
    }
    downloader.prepare_filename.return_value = str(
        tmp_path / "Odnoklassniki-248244667877.mp4"
    )

    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader):
        with pytest.raises(V0Error) as raised:
            acquire_media("https://ok.example/video/248244667877", tmp_path)

    assert raised.value.code == "acquisition_failed"
    assert raised.value.stage == "acquisition"
    assert ".ytdl" not in str(raised.value)


def test_failed_download_partial_does_not_poison_retry(tmp_path: Path) -> None:
    url = "https://ok.example/video/248244667877"
    stale = tmp_path / "Odnoklassniki-248244667877.mp4.ytdl"
    final = tmp_path / "Odnoklassniki-248244667877.mp4"
    first = _failed_downloader()

    def fail_after_partial(*_args, **_kwargs):
        stale.write_bytes(b"resume metadata")
        raise DownloadError("interrupted")

    first.extract_info.side_effect = fail_after_partial
    with (
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=first),
        patch(
            "dialogue_locator.acquisition.download_direct_media",
            side_effect=V0Error("direct fallback failed"),
        ),
    ):
        with pytest.raises(V0Error):
            acquire_media(url, tmp_path)

    second = Mock()
    second.__enter__ = Mock(return_value=second)
    second.__exit__ = Mock(return_value=False)

    def complete(*_args, **_kwargs):
        final.write_bytes(b"complete media")
        return {
            "id": "248244667877",
            "extractor": "Odnoklassniki",
            "filepath": str(final),
        }

    second.extract_info.side_effect = complete
    second.prepare_filename.return_value = str(final)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep")

    with patch("dialogue_locator.acquisition.YoutubeDL", return_value=second):
        path, _ = acquire_media(url, tmp_path)

    assert path == final.resolve()
    assert stale.is_file()
    assert unrelated.read_text() == "keep"


def test_partial_artifact_failure_stops_before_media_inspection(tmp_path: Path) -> None:
    work_dir = tmp_path / "media"
    work_dir.mkdir()
    (work_dir / "Odnoklassniki-248244667877.mp4.ytdl").write_bytes(b"resume metadata")
    downloader = _failed_downloader()

    with (
        patch(
            "dialogue_locator.pipeline.require_external_tools",
            return_value=SimpleNamespace(ffmpeg="ffmpeg", ffprobe="ffprobe"),
        ),
        patch("dialogue_locator.acquisition.YoutubeDL", return_value=downloader),
        patch(
            "dialogue_locator.acquisition.download_direct_media",
            side_effect=V0Error("direct fallback failed"),
        ),
        patch("dialogue_locator.pipeline._inspect_media_cached") as inspect,
    ):
        with pytest.raises(V0Error) as raised:
            run_v0(
                "https://ok.example/video/248244667877",
                work_dir,
                tmp_path / "output",
            )

    assert raised.value.code == "acquisition_failed"
    assert raised.value.stage == "acquisition"
    inspect.assert_not_called()
