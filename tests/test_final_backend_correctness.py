from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dialogue_locator.errors import V0Error
from dialogue_locator.frames import DIALOGUE_FRAME_FILENAME
from dialogue_locator.models import MediaInfo, ResolvedFrame, Transcription, TranscriptWord
from dialogue_locator.pipeline import run_v1


class _StaticTranscriber:
    def __init__(self, transcription: Transcription) -> None:
        self.transcription = transcription
        self.calls = 0

    def __call__(self, _: Path) -> Transcription:
        self.calls += 1
        return self.transcription


def _media() -> MediaInfo:
    return MediaInfo(
        duration=20.0,
        has_video=True,
        has_audio=True,
        embedded_subtitles=[],
        width=1280,
        height=720,
        video_codec="h264",
        audio_codec="aac",
        avg_frame_rate="30/1",
        real_frame_rate="30/1",
        video_time_base="1/1000",
        video_start_time=0.0,
        audio_start_time=0.0,
    )


def _run_mocked_full_asr(
    tmp_path: Path,
    query: str,
    words: list[TranscriptWord],
):
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    output_dir = tmp_path / "output"
    media_path = work_dir / "media.mkv"
    media_path.write_bytes(b"media")
    audio_path = work_dir / "speech.wav"
    audio_path.write_bytes(b"audio")
    transcriber = _StaticTranscriber(
        Transcription(
            text="".join(word.text for word in words).strip(),
            words=words,
            language="en",
            language_probability=1.0,
        )
    )

    def resolve_frame(_: Path, timestamp: float, destination: Path) -> ResolvedFrame:
        destination.mkdir(parents=True, exist_ok=True)
        frame_path = destination / DIALOGUE_FRAME_FILENAME
        frame_path.write_bytes(b"png")
        return ResolvedFrame(42, round(timestamp * 1000), "1/1000", timestamp, frame_path)

    with (
        patch(
            "dialogue_locator.pipeline.require_external_tools",
            return_value=SimpleNamespace(ffprobe="ffprobe", ffmpeg="ffmpeg"),
        ),
        patch("dialogue_locator.pipeline.acquire_media", return_value=(media_path, {})),
        patch("dialogue_locator.pipeline._inspect_media_cached", return_value=(_media(), False)),
        patch("dialogue_locator.pipeline._create_transcriber", return_value=transcriber),
        patch("dialogue_locator.pipeline.extract_speech_audio", return_value=audio_path),
        patch("dialogue_locator.pipeline.resolve_frame_at_timestamp", side_effect=resolve_frame),
    ):
        result = run_v1(
            "https://example.test/video",
            query,
            work_dir,
            output_dir,
            tmp_path / "models",
        )
    return result, transcriber


def test_full_asr_repeated_dialogue_returns_first_occurrence_and_consistent_result(
    tmp_path: Path,
) -> None:
    words = [
        TranscriptWord(" Come", 2.0, 2.3),
        TranscriptWord(" here.", 2.3, 2.8),
        TranscriptWord(" filler", 4.0, 4.5),
        TranscriptWord(" Come", 9.0, 9.3),
        TranscriptWord(" here.", 9.3, 9.8),
    ]

    result, transcriber = _run_mocked_full_asr(tmp_path, "come here", words)
    payload = result.to_dict()

    assert transcriber.calls == 1
    assert result.match.start == 2.0
    assert result.match.end == 2.8
    assert [occurrence.start for occurrence in result.occurrences] == [2.0, 9.0]
    assert payload["occurrence_count"] == 2
    assert payload["matched_text"] == "Come here."
    assert payload["dialogue_start"] == payload["frame_timestamp"] == 2.0
    assert payload["frame_pts"] == 2000
    assert payload["frame_time_base"] == "1/1000"
    assert Path(payload["frame_path"]).is_file()
    assert payload["localization_source"] == "asr"
    assert payload["verification_source"] == "asr"
    assert payload["asr_model_used"] == "base.en"
    assert payload["precision_fallback_used"] is False
    assert payload["confidence"] in {"HIGH", "MEDIUM", "LOW"}
    assert payload["confidence_reason"]


def test_full_asr_numeric_equivalence_preserves_original_transcript_text(
    tmp_path: Path,
) -> None:
    words = [
        TranscriptWord(" The", 5.0, 5.1),
        TranscriptWord(" company", 5.1, 5.3),
        TranscriptWord(" reported", 5.3, 5.6),
        TranscriptWord(" revenue", 5.6, 5.9),
        TranscriptWord(" of", 5.9, 6.0),
        TranscriptWord(" $20", 6.0, 6.2),
        TranscriptWord(" million.", 6.2, 6.5),
    ]

    result, _ = _run_mocked_full_asr(
        tmp_path,
        "The company reported revenue of twenty million dollars.",
        words,
    )

    assert result.match.match_type == "exact"
    assert result.match.score == 100.0
    assert result.match.matched_text == "The company reported revenue of $20 million."
    assert result.match.start == 5.0
    assert result.match.end == 6.5


def test_full_asr_absent_target_is_structured_and_removes_stale_frame(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stale_frame = output_dir / DIALOGUE_FRAME_FILENAME
    stale_frame.write_bytes(b"stale")
    words = [TranscriptWord(" unrelated", 1.0, 1.5), TranscriptWord(" speech", 1.5, 2.0)]

    with pytest.raises(V0Error) as raised:
        _run_mocked_full_asr(tmp_path, "target dialogue", words)

    assert raised.value.code == "dialogue_not_found"
    assert raised.value.stage == "matching"
    assert not stale_frame.exists()


def test_success_failure_success_does_not_poison_output_or_cache_state(
    tmp_path: Path,
) -> None:
    matching_words = [
        TranscriptWord(" target", 2.0, 2.3),
        TranscriptWord(" dialogue", 2.3, 2.8),
    ]

    first, _ = _run_mocked_full_asr(tmp_path, "target dialogue", matching_words)
    frame_path = first.frame.path
    assert frame_path.read_bytes() == b"png"

    with pytest.raises(V0Error) as raised:
        _run_mocked_full_asr(
            tmp_path,
            "absent dialogue",
            [TranscriptWord(" unrelated", 1.0, 1.5)],
        )
    assert raised.value.code == "dialogue_not_found"
    assert not frame_path.exists()

    final, _ = _run_mocked_full_asr(tmp_path, "target dialogue", matching_words)
    assert final.frame.path == frame_path
    assert frame_path.read_bytes() == b"png"
    assert [path.name for path in frame_path.parent.iterdir()] == [DIALOGUE_FRAME_FILENAME]
