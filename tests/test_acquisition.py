from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dialogue_locator.acquisition import (
    DOWNLOAD_CHUNK_SIZE,
    FORMAT_SELECTOR,
    acquire_media,
    download_direct_media,
    normalize_public_url,
    validate_public_url,
)
from yt_dlp.utils import DownloadError
from dialogue_locator.errors import V0Error


def test_format_selector_prefers_bounded_height_then_falls_back() -> None:
    assert "height<=720" in FORMAT_SELECTOR
    assert FORMAT_SELECTOR.endswith("bv*+ba/b")


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
