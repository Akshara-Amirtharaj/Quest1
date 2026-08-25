from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

from rapidfuzz import fuzz

from .acquisition import acquire_media, validate_public_url
from .audio import extract_speech_audio
from .caption_matching import find_caption_candidates, merge_caption_candidates
from .caption_verification import verify_caption_candidates
from .captions import discover_captions
from .cache import CachedOCRReader, CachedTranscriber, JsonFileCache, load_media_info
from .config import V2Config, V3Config
from .dependencies import require_external_tools
from .errors import V0Error
from .frames import decode_sample_frame, iter_frames_in_interval, resolve_frame_at_timestamp
from .inspection import inspect_media
from .matching import DEFAULT_FUZZY_THRESHOLD, find_dialogue_candidates, normalize_text
from .models import CaptionCandidate, CaptionTrack, DialogueMatch, MediaInfo, ResolvedFrame, Transcription, V0Result, V1Result, V3Result
from .ocr import OCRReader, PADDLE_MODEL_DESCRIPTION, PaddleOCRReader, find_first_visible_frame
from .subtitles import (
    SubtitleRateLimitError,
    download_subtitle,
    has_cached_subtitle,
    parse_subtitle,
    select_caption_tracks,
)
from .transcription import (
    DEFAULT_MODEL,
    FasterWhisperTranscriber,
    OptionalWhisperXTranscriber,
    WhisperXAligner,
    resolve_model_name,
)


LOGGER = logging.getLogger(__name__)
Transcriber = Callable[[Path], Transcription]
OCRReaderFactory = Callable[[Path], OCRReader]


def run_v0(
    url: str,
    work_dir: Path,
    output_dir: Path,
    cookies_from_browser: str | None = None,
    cookie_file: Path | None = None,
) -> V0Result:
    url = validate_public_url(url)
    tools = require_external_tools()
    media_path, metadata = acquire_media(
        url,
        work_dir,
        cookies_from_browser=cookies_from_browser,
        cookie_file=cookie_file,
    )
    media, metadata_cache_hit = _inspect_media_cached(
        media_path, tools.ffprobe, work_dir.parent / "pipeline-cache"
    )
    if not media.has_video:
        raise V0Error("The acquired media does not contain a video stream.")
    captions = discover_captions(metadata)
    sample_frame = decode_sample_frame(media_path, output_dir)
    return V0Result(
        url,
        media_path,
        media,
        captions,
        sample_frame,
        metadata_cache_hit=metadata_cache_hit,
    )


def run_v1(
    url: str,
    query: str,
    work_dir: Path,
    output_dir: Path,
    model_cache: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    language: str | None = None,
    cookies_from_browser: str | None = None,
    cookie_file: Path | None = None,
    precision_mode: str = "default",
) -> V1Result:
    url = validate_public_url(url)
    if not query.strip():
        raise V0Error("Target dialogue cannot be empty.")
    tools = require_external_tools()
    media_path, _ = acquire_media(
        url,
        work_dir,
        cookies_from_browser=cookies_from_browser,
        cookie_file=cookie_file,
    )
    cache_root = model_cache.parent / "pipeline-cache"
    media, metadata_cache_hit = _inspect_media_cached(media_path, tools.ffprobe, cache_root)
    _require_audio_video(media)
    resolved_model = resolve_model_name(model_name, language)
    transcriber = _create_transcriber(
        resolved_model,
        model_cache,
        device,
        compute_type,
        language,
        precision_mode,
        cache_root,
    )
    return _localize_full_audio(
        url,
        query,
        media_path,
        media,
        work_dir,
        output_dir,
        tools.ffmpeg,
        resolved_model,
        fuzzy_threshold,
        transcriber,
        precision_mode=precision_mode,
        metadata_cache_hit=metadata_cache_hit,
    )


def run_v2(
    url: str,
    query: str,
    work_dir: Path,
    output_dir: Path,
    model_cache: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    config: V2Config | None = None,
    cookies_from_browser: str | None = None,
    cookie_file: Path | None = None,
    precision_mode: str = "default",
) -> V1Result:
    url = validate_public_url(url)
    if not query.strip():
        raise V0Error("Target dialogue cannot be empty.")
    config = config or V2Config()
    tools = require_external_tools()
    media_path, metadata = acquire_media(
        url,
        work_dir,
        cookies_from_browser=cookies_from_browser,
        cookie_file=cookie_file,
    )
    cache_root = model_cache.parent / "pipeline-cache"
    media, metadata_cache_hit = _inspect_media_cached(media_path, tools.ffprobe, cache_root)
    _require_audio_video(media)
    work_dir.mkdir(parents=True, exist_ok=True)
    subtitle_cache_dir = work_dir / "captions" / media_path.stem
    resolved_model = resolve_model_name(model_name, language)

    transcriber: Transcriber | None = None
    attempted_audio_seconds = 0.0
    with tempfile.TemporaryDirectory(
        prefix="v2-", dir=work_dir, ignore_cleanup_errors=True
    ) as temporary:
        temporary_dir = Path(temporary)
        inventory = discover_captions(metadata)
        selected_tracks = select_caption_tracks(inventory, language)
        for source in ("manual", "automatic"):
            source_tracks = [track for track_source, track in selected_tracks if track_source == source]
            try:
                candidates = _caption_candidates_for_source(
                    source,
                    source_tracks,
                    query,
                    subtitle_cache_dir,
                    config,
                )
            except SubtitleRateLimitError:
                LOGGER.warning(
                    "Caption provider returned HTTP 429; stopping caption requests "
                    "and using full-audio ASR fallback."
                )
                break
            if not candidates:
                continue
            if transcriber is None:
                transcriber = _create_transcriber(
                    resolved_model,
                    model_cache,
                    device,
                    compute_type,
                    language,
                    precision_mode,
                    cache_root,
                )
            verification, source_audio_seconds = verify_caption_candidates(
                candidates,
                query,
                media_path,
                media.duration,
                media.audio_start_time or 0.0,
                temporary_dir,
                tools.ffmpeg,
                transcriber,
                config,
            )
            attempted_audio_seconds += source_audio_seconds
            if verification is not None:
                frame = resolve_frame_at_timestamp(media_path, verification.match.start, output_dir)
                provenance = _transcriber_provenance(transcriber)
                return V1Result(
                    url,
                    media_path,
                    query,
                    verification.match,
                    frame,
                    resolved_model,
                    localization_source="caption",
                    verification_source="asr",
                    audio_processed_seconds=attempted_audio_seconds,
                    caption_matched_text=verification.candidate.text,
                    caption_match_type=verification.candidate.match_type,
                    caption_match_score=verification.candidate.score,
                    evidence_conflict=_evidence_conflicts(
                        verification.candidate.text,
                        verification.match.matched_text,
                        config.verification_fuzzy_threshold,
                    ),
                    occurrences=(verification.match,),
                    transcription_language=provenance[0],
                    alignment_source=provenance[1],
                    precision_mode=precision_mode,
                    precision_fallback_reason=provenance[2],
                    transcript_cache_hit=provenance[3],
                    media_metadata_cache_hit=metadata_cache_hit,
                )

    if transcriber is None:
        transcriber = _create_transcriber(
            resolved_model,
            model_cache,
            device,
            compute_type,
            language,
            precision_mode,
            cache_root,
        )
    fallback = _localize_full_audio(
        url,
        query,
        media_path,
        media,
        work_dir,
        output_dir,
        tools.ffmpeg,
        resolved_model,
        config.verification_fuzzy_threshold,
        transcriber,
        precision_mode=precision_mode,
        metadata_cache_hit=metadata_cache_hit,
    )
    fallback_seconds = fallback.audio_processed_seconds or 0.0
    return replace(
        fallback,
        localization_source="asr",
        verification_source="asr",
        audio_processed_seconds=attempted_audio_seconds + fallback_seconds,
        media_metadata_cache_hit=metadata_cache_hit,
    )


def run_v3(
    url: str,
    query: str,
    work_dir: Path,
    output_dir: Path,
    model_cache: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    v2_config: V2Config | None = None,
    v3_config: V3Config | None = None,
    ocr_reader_factory: OCRReaderFactory = PaddleOCRReader,
    cookies_from_browser: str | None = None,
    cookie_file: Path | None = None,
    precision_mode: str = "default",
) -> V3Result:
    config = v3_config or V3Config()
    localized = run_v2(
        url,
        query,
        work_dir,
        output_dir,
        model_cache,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        config=v2_config,
        cookies_from_browser=cookies_from_browser,
        cookie_file=cookie_file,
        precision_mode=precision_mode,
    )
    interval_start = max(0.0, localized.match.start - config.search_margin)
    interval_end = localized.match.end + config.search_margin
    processed_frames = 0
    ocr_cache_hits = 0
    try:
        uncached_reader = ocr_reader_factory(model_cache / "paddleocr")
        reader = CachedOCRReader(
            uncached_reader,
            JsonFileCache(model_cache.parent / "pipeline-cache"),
        )
        frame, ocr_match, processed_frames = find_first_visible_frame(
            query,
            iter_frames_in_interval(localized.media_path, interval_start, interval_end),
            reader,
            config.fuzzy_threshold,
        )
        ocr_cache_hits = reader.cache_hits
    except Exception as exc:
        LOGGER.warning("Visible-text verification unavailable; using spoken result: %s", exc)
        frame = None
        ocr_match = None

    if frame is None or ocr_match is None:
        return V3Result(
            **localized.__dict__,
            frame_match_type="spoken_dialogue",
            ocr_processed_frames=processed_frames,
            ocr_model=PADDLE_MODEL_DESCRIPTION,
            ocr_cache_hits=ocr_cache_hits,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = (output_dir / "visible_dialogue_frame.png").resolve()
    frame.image.save(frame_path, format="PNG")
    resolved = ResolvedFrame(
        index=frame.index,
        pts=frame.pts,
        time_base=frame.time_base,
        timestamp=frame.timestamp,
        path=frame_path,
    )
    values = localized.__dict__ | {
        "frame": resolved,
        "verification_source": "ocr",
        "frame_match_type": "visible_text",
        "ocr_processed_frames": processed_frames,
        "ocr_model": getattr(reader, "model_description", PADDLE_MODEL_DESCRIPTION),
        "ocr_matched_text": ocr_match.matched_text,
        "ocr_match_type": ocr_match.match_type,
        "ocr_match_score": ocr_match.score,
        "ocr_cache_hits": ocr_cache_hits,
        "evidence_conflict": localized.evidence_conflict
        or _evidence_conflicts(
            localized.match.matched_text,
            ocr_match.matched_text,
            config.fuzzy_threshold,
        ),
    }
    return V3Result(**values)


def run_v4(*args, **kwargs) -> V3Result:
    """Final hardened pipeline; V0-V3 algorithms remain the underlying stages."""
    return run_v3(*args, **kwargs)


def _caption_candidates_for_source(
    source: str,
    tracks: list[CaptionTrack],
    query: str,
    subtitle_cache_dir: Path,
    config: V2Config,
) -> list[CaptionCandidate]:
    tracks_by_language: dict[str, list[CaptionTrack]] = {}
    for track in tracks:
        tracks_by_language.setdefault(track.language.casefold(), []).append(track)

    candidate_groups: list[list[CaptionCandidate]] = []
    for language_tracks in tracks_by_language.values():
        language_tracks.sort(
            key=lambda track: not has_cached_subtitle(
                track, subtitle_cache_dir / source
            )
        )
        for track in language_tracks:
            subtitle_path: Path | None = None
            try:
                subtitle_path = download_subtitle(track, subtitle_cache_dir / source)
                entries = parse_subtitle(subtitle_path)
            except SubtitleRateLimitError:
                raise
            except Exception as exc:
                if subtitle_path is not None:
                    try:
                        subtitle_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                LOGGER.warning(
                    "Ignoring unusable %s subtitle track %s (%s): %s",
                    source,
                    track.language,
                    track.extension,
                    exc,
                )
                continue
            candidate_groups.append(
                find_caption_candidates(
                    query,
                    entries,
                    language=track.language,
                    caption_source=source,
                    max_window_entries=config.subtitle_window_size,
                    fuzzy_threshold=config.caption_fuzzy_threshold,
                )
            )
            # Other variants normally contain the same cues; use them only as acquisition fallbacks.
            break
    return merge_caption_candidates(candidate_groups)


def _require_audio_video(media: MediaInfo) -> None:
    if not media.has_video:
        raise V0Error("The acquired media does not contain a video stream.")
    if not media.has_audio:
        raise V0Error("The acquired media does not contain an audio stream.")


def _localize_full_audio(
    url: str,
    query: str,
    media_path: Path,
    media: MediaInfo,
    work_dir: Path,
    output_dir: Path,
    ffmpeg: str,
    model_name: str,
    fuzzy_threshold: float,
    transcriber: Transcriber,
    precision_mode: str = "default",
    metadata_cache_hit: bool = False,
) -> V1Result:
    with tempfile.TemporaryDirectory(
        prefix="v1-audio-", dir=work_dir, ignore_cleanup_errors=True
    ) as temporary:
        audio_path = extract_speech_audio(media_path, Path(temporary) / "speech.wav", ffmpeg)
        transcription = transcriber(audio_path)
    relative_matches = find_dialogue_candidates(query, transcription.words, fuzzy_threshold)
    audio_offset = media.audio_start_time or 0.0
    occurrences = tuple(
        DialogueMatch(
            matched_text=relative_match.matched_text,
            start=relative_match.start + audio_offset,
            end=relative_match.end + audio_offset,
            match_type=relative_match.match_type,
            score=relative_match.score,
        )
        for relative_match in relative_matches
    )
    match = occurrences[0]
    frame = resolve_frame_at_timestamp(media_path, match.start, output_dir)
    processed_seconds = media.duration
    if processed_seconds is None and transcription.words:
        processed_seconds = transcription.words[-1].end
    return V1Result(
        url,
        media_path,
        query,
        match,
        frame,
        model_name,
        audio_processed_seconds=processed_seconds,
        occurrences=occurrences,
        transcription_language=transcription.language,
        alignment_source=transcription.alignment_source,
        precision_mode=precision_mode,
        precision_fallback_reason=transcription.precision_fallback_reason,
        transcript_cache_hit=bool(getattr(transcriber, "last_cache_hit", False)),
        media_metadata_cache_hit=metadata_cache_hit,
    )


def _create_transcriber(
    model_name: str,
    model_cache: Path,
    device: str,
    compute_type: str,
    language: str | None,
    precision_mode: str,
    cache_root: Path,
) -> CachedTranscriber:
    if precision_mode not in {"default", "whisperx"}:
        raise ValueError("precision_mode must be 'default' or 'whisperx'")
    base = FasterWhisperTranscriber(
        model_name=model_name,
        model_cache=model_cache,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    transcriber: Transcriber = base
    whisperx_available = find_spec("whisperx") is not None
    if precision_mode == "whisperx":
        transcriber = OptionalWhisperXTranscriber(
            base,
            WhisperXAligner(model_cache / "whisperx", device, language),
        )
    identity = "|".join(
        (
            model_name,
            _package_version("faster-whisper"),
            compute_type,
            language or "auto",
            precision_mode,
            f"whisperx-installed={whisperx_available}",
            _package_version("whisperx") if whisperx_available else "whisperx-absent",
        )
    )
    return CachedTranscriber(transcriber, JsonFileCache(cache_root), identity)


def _inspect_media_cached(
    media_path: Path,
    ffprobe: str,
    cache_root: Path,
) -> tuple[MediaInfo, bool]:
    return load_media_info(
        media_path,
        JsonFileCache(cache_root),
        lambda path: inspect_media(path, ffprobe),
    )


def _transcriber_provenance(
    transcriber: Transcriber,
) -> tuple[str | None, str, str | None, bool]:
    transcription = getattr(transcriber, "last_transcription", None)
    if transcription is None:
        return None, "faster-whisper", None, False
    return (
        transcription.language,
        transcription.alignment_source,
        transcription.precision_fallback_reason,
        bool(getattr(transcriber, "last_cache_hit", False)),
    )


def _evidence_conflicts(first: str, second: str, threshold: float) -> bool:
    normalized_first = normalize_text(first)
    normalized_second = normalize_text(second)
    if not normalized_first or not normalized_second:
        return True
    return float(fuzz.partial_ratio(normalized_first, normalized_second)) < threshold


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"
