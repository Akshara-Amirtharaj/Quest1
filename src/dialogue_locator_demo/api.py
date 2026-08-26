from __future__ import annotations

import os
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dialogue_locator.errors import V0Error
from dialogue_locator.pipeline import run_v2


LOGGER = logging.getLogger("uvicorn.error")
HEARTBEAT_INTERVAL_SECONDS = 15.0


class FindRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    video_url: str = Field(min_length=1, max_length=4096)
    dialogue: str = Field(min_length=1, max_length=2000)

    @field_validator("video_url")
    @classmethod
    def validate_video_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("invalid public video URL")
        return value


class ApiError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ApiError


class FrameResult(BaseModel):
    url: str
    index: int
    pts: int
    time_base: str
    timestamp: float
    timestamp_hms: str


class MatchResult(BaseModel):
    query: str
    text: str
    start: float
    start_hms: str
    end: float
    end_hms: str
    type: str
    score: float
    confidence: str
    confidence_reason: str
    occurrence_count: int


class EvidenceResult(BaseModel):
    localization_source: str
    verification_source: str
    frame_match_type: str
    caption_text: str | None
    caption_match_type: str | None
    caption_match_score: float | None
    ocr_text: str | None
    ocr_match_type: str | None
    ocr_match_score: float | None
    evidence_conflict: bool


class ProcessingResult(BaseModel):
    elapsed_seconds: float
    audio_processed_seconds: float | None
    audio_processed_hms: str | None


class FindResponse(BaseModel):
    frame: FrameResult
    match: MatchResult
    evidence: EvidenceResult
    processing: ProcessingResult
    details: dict[str, Any]


class FrameRegistry:
    """Process-local opaque IDs for frame paths produced by this API process."""

    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()

    def add(self, path: Path) -> str:
        safe_id = uuid.uuid4().hex
        with self._lock:
            self._paths[safe_id] = path.resolve(strict=True)
        return safe_id

    def get(self, safe_id: str) -> Path | None:
        if len(safe_id) != 32 or any(char not in "0123456789abcdef" for char in safe_id):
            return None
        with self._lock:
            return self._paths.get(safe_id)


PUBLIC_ERRORS: dict[str, tuple[int, str]] = {
    "VIDEO_URL_REQUIRED": (422, "Enter a video URL to continue."),
    "INVALID_VIDEO_URL": (422, "Enter a valid public video URL."),
    "DIALOGUE_REQUIRED": (422, "Enter the dialogue you want to find."),
    "VIDEO_UNAVAILABLE": (
        422,
        "We couldn't access this video. Check the URL and make sure the video is publicly available.",
    ),
    "DIALOGUE_NOT_FOUND": (
        404,
        "We couldn't find this dialogue in the video. Try a slightly different phrase.",
    ),
    "PROCESSING_FAILED": (
        500,
        "Something went wrong while processing the video. Please try again.",
    ),
}


def _public_error(code: str) -> tuple[int, dict[str, str]]:
    status, message = PUBLIC_ERRORS[code]
    return status, {"code": code, "message": message}


def _pipeline_error_code(message: str) -> str:
    normalized = message.casefold()
    if "dialogue not found" in normalized or "transcription contains no matchable words" in normalized:
        return "DIALOGUE_NOT_FOUND"
    if (
        "target dialogue cannot be empty" in normalized
        or "target dialogue must contain at least one letter or number" in normalized
    ):
        return "DIALOGUE_REQUIRED"
    if any(
        phrase in normalized
        for phrase in (
            "video url must be",
            "invalid video url",
            "video url cannot be empty",
            "video url is empty",
        )
    ):
        return "INVALID_VIDEO_URL"
    if any(
        term in normalized
        for term in (
            "acquir",
            "download",
            "yt-dlp",
            "direct http",
            "http error",
            "httperror",
            "extractor",
            "cookie",
            "ffmpeg",
            "ffprobe",
            "does not contain an audio stream",
            "does not contain a video stream",
            "no decodable video stream",
        )
    ):
        return "VIDEO_UNAVAILABLE"
    return "PROCESSING_FAILED"


def _adapt_result(result: Any, frame_id: str, elapsed_seconds: float) -> FindResponse:
    payload = result.to_dict()
    return FindResponse(
        frame=FrameResult(
            url=f"/api/frames/{frame_id}",
            index=payload["frame_index"],
            pts=payload["frame_pts"],
            time_base=payload["frame_time_base"],
            timestamp=payload["frame_timestamp"],
            timestamp_hms=payload["frame_timestamp_hms"],
        ),
        match=MatchResult(
            query=payload["query"],
            text=payload["matched_text"],
            start=payload["dialogue_start"],
            start_hms=payload["dialogue_start_hms"],
            end=payload["dialogue_end"],
            end_hms=payload["dialogue_end_hms"],
            type=payload["match_type"],
            score=payload["match_score"],
            confidence=payload["confidence"],
            confidence_reason=payload["confidence_reason"],
            occurrence_count=payload["occurrence_count"],
        ),
        evidence=EvidenceResult(
            localization_source=payload["localization_source"],
            verification_source=payload["verification_source"],
            frame_match_type=payload.get("frame_match_type", "spoken_dialogue"),
            caption_text=payload["caption_matched_text"],
            caption_match_type=payload["caption_match_type"],
            caption_match_score=payload["caption_match_score"],
            ocr_text=payload.get("ocr_matched_text"),
            ocr_match_type=payload.get("ocr_match_type"),
            ocr_match_score=payload.get("ocr_match_score"),
            evidence_conflict=payload["evidence_conflict"],
        ),
        processing=ProcessingResult(
            elapsed_seconds=round(elapsed_seconds, 3),
            audio_processed_seconds=payload["audio_processed_seconds"],
            audio_processed_hms=payload["audio_processed_hms"],
        ),
        details={
            "transcription_language": payload["transcription_language"],
            "alignment_source": payload["alignment_source"],
            "asr_model": payload["asr_model"],
            "asr_model_used": payload.get("asr_model_used", payload["asr_model"]),
            "precision_mode": payload["precision_mode"],
            "precision_fallback_reason": payload["precision_fallback_reason"],
            "precision_fallback_used": payload.get("precision_fallback_used", False),
            "precision_scope": payload.get("precision_scope"),
            "base_match_score": payload.get("base_match_score", payload["match_score"]),
            "precision_match_score": payload.get("precision_match_score"),
            "ocr_model": payload.get("ocr_model"),
            "ocr_processed_frames": payload.get("ocr_processed_frames", 0),
            "transcript_cache_hit": payload["transcript_cache_hit"],
            "media_metadata_cache_hit": payload["media_metadata_cache_hit"],
            "ocr_cache_hits": payload.get("ocr_cache_hits", 0),
        },
    )


def create_app(data_root: Path | None = None) -> FastAPI:
    root = (data_root or Path(os.environ.get("QUEST1_DEMO_DATA_DIR", ".cache/demo"))).resolve()
    static_root = Path(__file__).with_name("static")
    registry = FrameRegistry()
    app = FastAPI(title="Quest1 Demo API", version="1.0.0")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = first.get("loc", (None,))[-1]
        value = first.get("input")
        if field == "video_url":
            code = (
                "VIDEO_URL_REQUIRED"
                if first.get("type") == "missing" or value is None or not str(value).strip()
                else "INVALID_VIDEO_URL"
            )
        elif field == "dialogue":
            code = "DIALOGUE_REQUIRED"
        else:
            code = "PROCESSING_FAILED"
        status, detail = _public_error(code)
        return JSONResponse(
            status_code=status,
            content={"detail": detail},
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/find",
        response_model=FindResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def find_frame(request: FindRequest) -> FindResponse:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        stop_heartbeat = threading.Event()

        def report_heartbeat() -> None:
            while not stop_heartbeat.wait(HEARTBEAT_INTERVAL_SECONDS):
                LOGGER.info(
                    "Quest1 request %s is still processing (%.0f seconds elapsed). "
                    "Caption unavailability can require full-audio ASR.",
                    request_id[:8],
                    time.perf_counter() - started,
                )

        LOGGER.info(
            "Quest1 request %s started: acquiring media and checking captions/audio/video.",
            request_id[:8],
        )
        heartbeat = threading.Thread(
            target=report_heartbeat,
            name=f"quest1-progress-{request_id[:8]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            # This UI locates spoken dialogue. Visible-text verification remains
            # available through the production V3/V4 commands, but must not load
            # PaddleOCR for an already verified spoken-dialogue result.
            result = run_v2(
                url=request.video_url,
                query=request.dialogue,
                work_dir=root / "media",
                output_dir=root / "runs" / request_id,
                model_cache=root / "models",
                language=None,
            )
            frame_id = registry.add(result.frame.path)
        except V0Error as exc:
            code = _pipeline_error_code(str(exc))
            status, detail = _public_error(code)
            LOGGER.warning(
                "Quest1 request %s failed after %.1f seconds: %s",
                request_id[:8],
                time.perf_counter() - started,
                exc,
            )
            raise HTTPException(status_code=status, detail=detail) from exc
        except OSError as exc:
            LOGGER.exception(
                "Quest1 request %s could not store its result after %.1f seconds.",
                request_id[:8],
                time.perf_counter() - started,
            )
            status, detail = _public_error("PROCESSING_FAILED")
            raise HTTPException(status_code=status, detail=detail) from exc
        except Exception as exc:
            LOGGER.exception(
                "Quest1 request %s failed internally after %.1f seconds.",
                request_id[:8],
                time.perf_counter() - started,
            )
            status, detail = _public_error("PROCESSING_FAILED")
            raise HTTPException(status_code=status, detail=detail) from exc
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=0.1)
        elapsed = time.perf_counter() - started
        LOGGER.info("Quest1 request %s completed in %.1f seconds.", request_id[:8], elapsed)
        return _adapt_result(result, frame_id, elapsed)

    @app.get("/api/frames/{safe_id}", response_class=FileResponse)
    def get_frame(safe_id: str) -> FileResponse:
        frame_path = registry.get(safe_id)
        if frame_path is None or not frame_path.is_file():
            raise HTTPException(
                status_code=404,
                detail={"code": "FRAME_NOT_FOUND", "message": "The requested frame is unavailable."},
            )
        return FileResponse(frame_path, media_type="image/png", filename="dialogue-frame.png")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(static_root / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=static_root), name="static")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("dialogue_locator_demo.api:app", host="127.0.0.1", port=8000, reload=False)
