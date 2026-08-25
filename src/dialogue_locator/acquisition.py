from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .cache import JsonFileCache
from .errors import V0Error


FORMAT_SELECTOR = (
    "bv*[height<=720]+ba/b[height<=720]/"
    "bv*+ba/b"
)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DIRECT_DOWNLOAD_TIMEOUT = (15, 60)
MEDIA_URL_CACHE_NAMESPACE = "media-urls"
MEDIA_URL_CACHE_DIRECTORY = ".quest1-cache"
MEDIA_METADATA_CACHE_KEYS = (
    "id",
    "extractor",
    "extractor_key",
    "webpage_url",
    "original_url",
    "duration",
    "subtitles",
    "automatic_captions",
)


def normalize_public_url(url: str) -> str:
    candidate = url.strip()
    markdown = re.fullmatch(r"\[[^\]]*\]\((https?://[^)]+)\)", candidate, flags=re.IGNORECASE)
    if markdown:
        candidate = markdown.group(1)
    if candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1].strip()
    candidate = re.sub(r"\\([_&?=#%])", r"\1", candidate)
    return candidate


def validate_public_url(url: str) -> str:
    normalized = normalize_public_url(url)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise V0Error("Video URL must be a publicly accessible http:// or https:// URL.")
    return normalized


def _is_probably_media_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type.startswith(("video/", "audio/")):
        return True
    return content_type in {"", "application/octet-stream", "binary/octet-stream"}


def _direct_download_path(url: str, cache_dir: Path) -> Path:
    suffix = Path(urlparse(url).path).suffix
    if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
        suffix = ".media"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"direct-{digest}{suffix.lower()}"


def _media_url_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _cached_media_for_url(url: str, cache_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    cache = JsonFileCache(cache_dir / MEDIA_URL_CACHE_DIRECTORY)
    cached = cache.get(MEDIA_URL_CACHE_NAMESPACE, _media_url_cache_key(url))
    if cached is None or cached.get("url") != url:
        return None

    filename = cached.get("filename")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        return None
    media_path = (cache_dir / filename).resolve()
    try:
        media_path.relative_to(cache_dir.resolve())
    except ValueError:
        return None
    if not media_path.is_file() or media_path.stat().st_size == 0:
        return None

    metadata = cached.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return media_path, {**metadata, "webpage_url": url, "media_cache_hit": True}


def _store_media_for_url(
    url: str,
    cache_dir: Path,
    media_path: Path,
    metadata: dict[str, Any],
) -> None:
    resolved_cache = cache_dir.resolve()
    resolved_media = media_path.resolve()
    try:
        relative = resolved_media.relative_to(resolved_cache)
    except ValueError:
        return
    if len(relative.parts) != 1:
        return

    cache_metadata = {
        key: value for key, value in YoutubeDL.sanitize_info(metadata).items()
        if key in MEDIA_METADATA_CACHE_KEYS
    }
    JsonFileCache(cache_dir / MEDIA_URL_CACHE_DIRECTORY).put(
        MEDIA_URL_CACHE_NAMESPACE,
        _media_url_cache_key(url),
        {
            "url": url,
            "filename": relative.name,
            "metadata": cache_metadata,
        },
    )


def _legacy_cached_media(url: str, cache_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Adopt a media file downloaded before the URL index was introduced."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    identifiers = [query[key][0] for key in ("v", "id", "video_id") if query.get(key)]
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        identifiers.append(path_parts[-1])

    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    provider = hostname.split(".", 1)[0]

    for identifier in dict.fromkeys(identifiers):
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,128}", identifier):
            continue
        matches = [
            path for path in cache_dir.glob(f"*-{identifier}.*")
            if path.is_file() and path.stat().st_size > 0 and path.suffix.lower() != ".part"
        ]
        provider_matches = [
            path for path in matches if path.name.lower().startswith(f"{provider}-")
        ]
        usable = provider_matches or (matches if len(matches) == 1 else [])
        if not usable:
            continue
        media_path = max(usable, key=lambda path: path.stat().st_mtime).resolve()
        metadata = {
            "id": identifier,
            "extractor": media_path.name.split("-", 1)[0],
            "webpage_url": url,
            "media_cache_hit": True,
        }
        _store_media_for_url(url, cache_dir, media_path, metadata)
        return media_path, metadata
    return None


def download_direct_media(url: str, cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise V0Error(f"Could not create media cache directory {cache_dir}: {exc}") from exc
    destination = _direct_download_path(url, cache_dir)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.resolve(), {"extractor": "direct-http", "webpage_url": url}

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=DIRECT_DOWNLOAD_TIMEOUT,
            verify=True,
        ) as response:
            response.raise_for_status()
            if not _is_probably_media_response(response):
                content_type = response.headers.get("Content-Type", "unknown")
                raise V0Error(
                    "The URL returned a page or non-media response "
                    f"({content_type}); direct HTTP fallback is only safe for media URLs."
                )
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        output.write(chunk)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise V0Error("Direct HTTP fallback returned an empty media file.")
        temporary.replace(destination)
    except V0Error:
        temporary.unlink(missing_ok=True)
        raise
    except (requests.RequestException, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise V0Error(f"Direct HTTP media download failed: {exc}") from exc

    return destination.resolve(), {
        "extractor": "direct-http",
        "webpage_url": url,
        "http_headers": {"content_type": response.headers.get("Content-Type")},
    }


def acquire_media(
    url: str,
    cache_dir: Path,
    *,
    cookies_from_browser: str | None = None,
    cookie_file: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    url = validate_public_url(url)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise V0Error(f"Could not create media cache directory {cache_dir}: {exc}") from exc

    cached = _cached_media_for_url(url, cache_dir)
    if cached is not None:
        return cached
    direct_path = _direct_download_path(url, cache_dir)
    if direct_path.is_file() and direct_path.stat().st_size > 0:
        metadata = {"extractor": "direct-http", "webpage_url": url, "media_cache_hit": True}
        _store_media_for_url(url, cache_dir, direct_path, metadata)
        return direct_path.resolve(), metadata
    legacy = _legacy_cached_media(url, cache_dir)
    if legacy is not None:
        return legacy

    options: dict[str, Any] = {
        "format": FORMAT_SELECTOR,
        "outtmpl": str(cache_dir / "%(extractor)s-%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "continuedl": True,
        "overwrites": False,
        "merge_output_format": "mkv",
    }
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookie_file:
        options["cookiefile"] = str(cookie_file)
    try:
        with YoutubeDL(options) as downloader:
            metadata = downloader.extract_info(url, download=True)
            if not isinstance(metadata, dict):
                raise V0Error("yt-dlp did not return media metadata.")
            requested = metadata.get("requested_downloads")
            candidates = []
            if isinstance(requested, list):
                candidates.extend(item.get("filepath") for item in requested if isinstance(item, dict))
            candidates.extend([metadata.get("filepath"), downloader.prepare_filename(metadata)])
    except (DownloadError, TypeError, ValueError) as exc:
        try:
            media_path, metadata = download_direct_media(url, cache_dir)
            _store_media_for_url(url, cache_dir, media_path, metadata)
            return media_path, metadata
        except V0Error as fallback_exc:
            raise V0Error(
                f"yt-dlp could not acquire the media: {exc}. "
                f"Direct HTTP fallback also failed: {fallback_exc} "
                "If the provider requires sign-in, retry with "
                "--cookies-from-browser BROWSER or --cookies-file PATH."
            ) from exc
    except OSError as exc:
        raise V0Error(f"Media acquisition failed: {exc}") from exc

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            media_path = Path(candidate).resolve()
            _store_media_for_url(url, cache_dir, media_path, metadata)
            return media_path, metadata

    # A merge can change the extension from yt-dlp's prepared filename.
    identifier = str(metadata.get("id", ""))
    matches = sorted(cache_dir.glob(f"*-{identifier}.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if matches:
        media_path = matches[0].resolve()
        _store_media_for_url(url, cache_dir, media_path, metadata)
        return media_path, metadata
    raise V0Error("yt-dlp completed but the downloaded media file could not be located.")
