from pathlib import Path

import pytest

from dialogue_locator.cache import CachedOCRReader, CachedTranscriber, JsonFileCache, load_media_info
from dialogue_locator.confidence import assess_confidence
from dialogue_locator.errors import V0Error
from dialogue_locator.matching import find_dialogue_candidates
from dialogue_locator.models import (
    MediaInfo,
    OCRLine,
    TranscriptWord,
    Transcription,
)
from dialogue_locator.transcription import (
    OptionalWhisperXTranscriber,
    resolve_model_name,
)


def _transcription() -> Transcription:
    return Transcription(
        "target words",
        [TranscriptWord("target", 1.0, 1.2), TranscriptWord("words", 1.2, 1.5)],
        "en",
        0.99,
    )


@pytest.mark.parametrize(
    ("options", "category"),
    [
        (
            dict(
                localization_source="caption",
                verification_source="asr",
                match_type="exact",
                match_score=100.0,
                caption_match_type="exact",
                caption_match_score=100.0,
            ),
            "HIGH",
        ),
        (
            dict(
                localization_source="asr",
                verification_source="ocr",
                match_type="exact",
                match_score=100.0,
                ocr_match_type="exact",
                ocr_match_score=100.0,
            ),
            "HIGH",
        ),
        (
            dict(
                localization_source="asr",
                verification_source="asr",
                match_type="exact",
                match_score=100.0,
            ),
            "MEDIUM",
        ),
        (
            dict(
                localization_source="asr",
                verification_source="asr",
                match_type="fuzzy",
                match_score=91.0,
            ),
            "LOW",
        ),
    ],
)
def test_confidence_categories_are_evidence_rules(options: dict, category: str) -> None:
    assessment = assess_confidence(**options)
    assert assessment.category == category
    assert assessment.reason


def test_conflicting_evidence_is_low_confidence() -> None:
    assessment = assess_confidence(
        localization_source="caption",
        verification_source="ocr",
        match_type="exact",
        match_score=100.0,
        caption_match_type="exact",
        caption_match_score=100.0,
        ocr_match_type="exact",
        ocr_match_score=100.0,
        evidence_conflict=True,
    )
    assert assessment.category == "LOW"
    assert "disagree" in assessment.reason


def test_weak_ocr_evidence_is_low_confidence() -> None:
    assessment = assess_confidence(
        localization_source="asr",
        verification_source="ocr",
        match_type="exact",
        match_score=100.0,
        ocr_match_type="fuzzy",
        ocr_match_score=72.0,
    )
    assert assessment.category == "LOW"
    assert "OCR evidence" in assessment.reason


def test_multiple_occurrences_are_detected_and_first_remains_default() -> None:
    words = [
        TranscriptWord("target", 1.0, 1.2),
        TranscriptWord("words", 1.2, 1.5),
        TranscriptWord("filler", 2.0, 2.5),
        TranscriptWord("target", 5.0, 5.2),
        TranscriptWord("words", 5.2, 5.5),
    ]
    matches = find_dialogue_candidates("target words", words)
    assert [match.start for match in matches] == [1.0, 5.0]
    assert all(match.matched_text == "target words" for match in matches)


def test_multilingual_model_selection_keeps_english_optimized() -> None:
    assert resolve_model_name("base.en", None) == "base.en"
    assert resolve_model_name("base.en", "en-US") == "base.en"
    assert resolve_model_name("base.en", "es") == "base"
    assert resolve_model_name("small", "ta-IN") == "small"
    with pytest.raises(V0Error, match="English-only"):
        resolve_model_name("tiny.en", "fr")


def test_transcript_cache_reuses_content_across_temporary_paths(tmp_path: Path) -> None:
    calls = 0

    def transcribe(_: Path) -> Transcription:
        nonlocal calls
        calls += 1
        return _transcription()

    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"same audio bytes")
    second_audio.write_bytes(b"same audio bytes")
    cached = CachedTranscriber(transcribe, JsonFileCache(tmp_path / "cache"), "base.en|en")

    assert cached(first_audio).text == "target words"
    assert cached.last_cache_hit is False
    assert cached(second_audio).text == "target words"
    assert cached.last_cache_hit is True
    assert calls == 1


def test_ocr_cache_reuses_identical_frame_without_calling_model(tmp_path: Path) -> None:
    calls = 0

    class Image:
        mode = "RGB"
        size = (2, 1)

        def tobytes(self) -> bytes:
            return b"abcdef"

    class Reader:
        model_description = "mock-ocr"

        def __call__(self, _: object) -> list[OCRLine]:
            nonlocal calls
            calls += 1
            return [OCRLine("visible words", 0.95)]

    reader = CachedOCRReader(Reader(), JsonFileCache(tmp_path / "cache"))
    assert reader(Image())[0].text == "visible words"
    assert reader(Image())[0].text == "visible words"
    assert reader.cache_hits == 1
    assert calls == 1


def test_media_metadata_cache_reuses_safe_ffprobe_result(tmp_path: Path) -> None:
    media_path = tmp_path / "video.mkv"
    media_path.write_bytes(b"media")
    calls = 0
    info = MediaInfo(1.0, True, True, [], 10, 10, "h264", "aac", "30/1", "30/1", "1/1000", 0.0, 0.0)

    def load(_: Path) -> MediaInfo:
        nonlocal calls
        calls += 1
        return info

    cache = JsonFileCache(tmp_path / "cache")
    assert load_media_info(media_path, cache, load) == (info, False)
    assert load_media_info(media_path, cache, load) == (info, True)
    assert calls == 1


def test_whisperx_unavailable_falls_back_to_faster_whisper() -> None:
    def unavailable(_: Path, __: Transcription) -> Transcription:
        raise RuntimeError("WhisperX is not installed")

    transcriber = OptionalWhisperXTranscriber(lambda _: _transcription(), unavailable)
    result = transcriber(Path("audio.wav"))

    assert result.alignment_source == "faster-whisper"
    assert result.precision_fallback_reason == "WhisperX is not installed"
