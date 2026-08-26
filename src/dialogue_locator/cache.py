from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import MediaInfo, OCRLine, Transcription, TranscriptWord


CACHE_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    value = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonFileCache:
    """Best-effort, schema-versioned JSON cache with atomic publication."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self.root / namespace / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        value = payload.get("value")
        return value if isinstance(value, dict) else None

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> bool:
        directory = self.root / namespace
        destination = directory / f"{key}.json"
        temporary = directory / f".{destination.name}.{uuid.uuid4().hex}.part"
        payload = {"schema_version": CACHE_SCHEMA_VERSION, "value": value}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(destination)
            return True
        except OSError as exc:
            LOGGER.warning("Optional %s cache write skipped: %s", namespace, exc)
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class CachedTranscriber:
    def __init__(
        self,
        transcriber: Callable[[Path], Transcription],
        cache: JsonFileCache,
        identity: str,
    ) -> None:
        self.transcriber = transcriber
        self.cache = cache
        self.identity = identity
        self.last_cache_hit = False
        self.last_transcription: Transcription | None = None

    def __call__(self, audio_path: Path) -> Transcription:
        # Temporary WAV paths change between runs, so transcript reuse is keyed
        # by audio bytes plus the complete model/configuration identity.
        key_source = f"{self.identity}|{content_digest(audio_path)}"
        key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
        cached = self.cache.get("transcripts", key)
        if cached is not None:
            try:
                transcription = _transcription_from_dict(cached)
            except (KeyError, TypeError, ValueError):
                transcription = None
            if transcription is not None:
                self.last_cache_hit = True
                self.last_transcription = transcription
                return transcription

        transcription = self.transcriber(audio_path)
        self.cache.put("transcripts", key, asdict(transcription))
        self.last_cache_hit = False
        self.last_transcription = transcription
        return transcription


class CachedOCRReader:
    def __init__(self, reader: Callable[[object], list[OCRLine]], cache: JsonFileCache) -> None:
        self.reader = reader
        self.cache = cache
        self.model_description = getattr(reader, "model_description", "ocr")
        self.cache_hits = 0

    def __call__(self, image: object) -> list[OCRLine]:
        key = _image_cache_key(image, self.model_description)
        cached = self.cache.get("ocr", key)
        if cached is not None and isinstance(cached.get("lines"), list):
            try:
                lines = [OCRLine(str(item["text"]), item.get("confidence")) for item in cached["lines"]]
            except (AttributeError, KeyError, TypeError):
                lines = []
            else:
                self.cache_hits += 1
                return lines
        lines = self.reader(image)
        self.cache.put("ocr", key, {"lines": [asdict(line) for line in lines]})
        return lines


def load_media_info(
    media_path: Path,
    cache: JsonFileCache,
    loader: Callable[[Path], MediaInfo],
) -> tuple[MediaInfo, bool]:
    key = file_fingerprint(media_path)
    cached = cache.get("media-info", key)
    if cached is not None:
        try:
            return MediaInfo(**cached), True
        except TypeError:
            pass
    info = loader(media_path)
    cache.put("media-info", key, asdict(info))
    return info, False


def _transcription_from_dict(payload: dict[str, Any]) -> Transcription:
    words = [TranscriptWord(**word) for word in payload["words"]]
    return Transcription(
        text=str(payload["text"]),
        words=words,
        language=payload.get("language"),
        language_probability=payload.get("language_probability"),
        alignment_source=str(payload.get("alignment_source", "faster-whisper")),
        precision_fallback_reason=payload.get("precision_fallback_reason"),
    )


def _image_cache_key(image: object, model_description: str) -> str:
    digest = hashlib.sha256(model_description.encode("utf-8"))
    if hasattr(image, "mode"):
        digest.update(str(getattr(image, "mode")).encode("utf-8"))
    if hasattr(image, "size"):
        digest.update(repr(getattr(image, "size")).encode("utf-8"))
    if hasattr(image, "tobytes"):
        digest.update(image.tobytes())
    else:
        digest.update(repr(image).encode("utf-8"))
    return digest.hexdigest()
