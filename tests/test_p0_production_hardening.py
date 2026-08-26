from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests

from dialogue_locator import cli
from dialogue_locator.acquisition import download_direct_media
from dialogue_locator.cache import CachedTranscriber, JsonFileCache
from dialogue_locator.errors import V0Error
from dialogue_locator.frames import (
    DIALOGUE_FRAME_FILENAME,
    LEGACY_VISIBLE_FRAME_FILENAME,
    prepare_dialogue_frame_output,
    resolve_frame_at_timestamp,
    save_frame_image,
)
from dialogue_locator.inspection import inspect_media
from dialogue_locator.models import (
    CandidateVideoFrame,
    DialogueMatch,
    MediaInfo,
    ResolvedFrame,
    Transcription,
    TranscriptWord,
    V1Result,
)
from dialogue_locator.pipeline import run_v1, run_v3


class _Image:
    def __init__(self, content: bytes = b"png") -> None:
        self.content = content

    def save(self, path: Path, format: str) -> None:
        assert format == "PNG"
        Path(path).write_bytes(self.content)


class _PartialFailureImage:
    def save(self, path: Path, format: str) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")


def _media(*, video: bool = True, audio: bool = True) -> MediaInfo:
    return MediaInfo(
        duration=10.0,
        has_video=video,
        has_audio=audio,
        embedded_subtitles=[],
        width=16 if video else None,
        height=9 if video else None,
        video_codec="h264" if video else None,
        audio_codec="aac" if audio else None,
        avg_frame_rate="30/1" if video else None,
        real_frame_rate="30/1" if video else None,
        video_time_base="1/1000" if video else None,
        video_start_time=0.0 if video else None,
        audio_start_time=0.0 if audio else None,
    )


def _localized(tmp_path: Path) -> V1Result:
    media_path = tmp_path / "video.mkv"
    media_path.write_bytes(b"media")
    frame_path = tmp_path / "spoken-frame.png"
    frame_path.write_bytes(b"spoken")
    return V1Result(
        "https://example.test/video",
        media_path,
        "target dialogue",
        DialogueMatch("target dialogue", 5.0, 6.0, "exact", 100.0),
        ResolvedFrame(7, 5000, "1/1000", 5.0, frame_path),
        "base.en",
        localization_source="caption",
        verification_source="asr",
        asr_model_used="base.en",
    )


def test_prepare_output_removes_only_owned_final_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / DIALOGUE_FRAME_FILENAME).write_bytes(b"stale")
    (output / LEGACY_VISIBLE_FRAME_FILENAME).write_bytes(b"legacy")
    unrelated = output / "notes.txt"
    unrelated.write_text("keep")

    prepare_dialogue_frame_output(output)

    assert not (output / DIALOGUE_FRAME_FILENAME).exists()
    assert not (output / LEGACY_VISIBLE_FRAME_FILENAME).exists()
    assert unrelated.read_text() == "keep"


def test_partial_frame_save_is_structured_and_cleans_temporary_file(tmp_path: Path) -> None:
    with pytest.raises(V0Error) as raised:
        save_frame_image(_PartialFailureImage(), tmp_path, DIALOGUE_FRAME_FILENAME)

    assert raised.value.code == "frame_output_failed"
    assert raised.value.stage == "frame_output"
    assert not (tmp_path / DIALOGUE_FRAME_FILENAME).exists()
    assert not list(tmp_path.glob(".*.part"))


def test_output_directory_creation_failure_is_structured(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_bytes(b"do not overwrite")

    with pytest.raises(V0Error) as raised:
        save_frame_image(_Image(), output_file, DIALOGUE_FRAME_FILENAME)

    assert raised.value.code == "frame_output_failed"
    assert raised.value.stage == "frame_output"
    assert output_file.read_bytes() == b"do not overwrite"


def test_atomic_replace_failure_preserves_previous_authoritative_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / DIALOGUE_FRAME_FILENAME
    destination.write_bytes(b"spoken")
    original_replace = Path.replace

    def fail_replace(path: Path, target: Path) -> Path:
        if path.name.startswith(f".{DIALOGUE_FRAME_FILENAME}."):
            raise OSError("replace denied")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(V0Error, match="replace denied"):
        save_frame_image(_Image(b"ocr"), tmp_path, DIALOGUE_FRAME_FILENAME)

    assert destination.read_bytes() == b"spoken"
    assert not list(tmp_path.glob(".*.part"))


def test_failed_new_run_invalidates_stale_frame_but_keeps_unrelated_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / DIALOGUE_FRAME_FILENAME).write_bytes(b"old")
    (output / LEGACY_VISIBLE_FRAME_FILENAME).write_bytes(b"old-ocr")
    unrelated = output / "keep.bin"
    unrelated.write_bytes(b"keep")

    with patch("dialogue_locator.pipeline.run_v2", side_effect=V0Error("acquisition failed")):
        with pytest.raises(V0Error):
            run_v3(
                "https://example.test/video",
                "target",
                tmp_path / "work",
                output,
                tmp_path / "models",
            )

    assert not (output / DIALOGUE_FRAME_FILENAME).exists()
    assert not (output / LEGACY_VISIBLE_FRAME_FILENAME).exists()
    assert unrelated.read_bytes() == b"keep"


def test_v3_ocr_success_publishes_only_authoritative_frame(tmp_path: Path) -> None:
    localized = _localized(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / LEGACY_VISIBLE_FRAME_FILENAME).write_bytes(b"stale")
    candidate = CandidateVideoFrame(9, 5100, "1/1000", 5.1, _Image(b"ocr"))

    class Reader:
        model_description = "mock-ocr"

        def __init__(self, _: Path) -> None:
            pass

        def __call__(self, _: object):
            from dialogue_locator.models import OCRLine

            return [OCRLine("target dialogue")]

    with (
        patch("dialogue_locator.pipeline.run_v2", return_value=localized),
        patch("dialogue_locator.pipeline.iter_frames_in_interval", return_value=iter((candidate,))),
    ):
        result = run_v3(
            localized.source_url,
            localized.query,
            tmp_path / "work",
            output,
            tmp_path / "models",
            ocr_reader_factory=Reader,
        )

    assert result.verification_source == "ocr"
    assert result.frame.path == (output / DIALOGUE_FRAME_FILENAME).resolve()
    assert result.frame.path.read_bytes() == b"ocr"
    assert not (output / LEGACY_VISIBLE_FRAME_FILENAME).exists()
    assert result.to_dict()["frame_timestamp"] == 5.1


def test_v3_ocr_publish_failure_returns_existing_spoken_result(tmp_path: Path) -> None:
    localized = _localized(tmp_path)
    candidate = CandidateVideoFrame(9, 5100, "1/1000", 5.1, _Image(b"ocr"))

    with (
        patch("dialogue_locator.pipeline.run_v2", return_value=localized),
        patch("dialogue_locator.pipeline.iter_frames_in_interval", return_value=iter((candidate,))),
        patch(
            "dialogue_locator.pipeline.find_first_visible_frame",
            return_value=(candidate, SimpleNamespace(matched_text="target dialogue", match_type="exact", score=100.0), 1),
        ),
        patch(
            "dialogue_locator.pipeline.save_frame_image",
            side_effect=V0Error(
                "Could not write final frame artifact",
                code="frame_output_failed",
                stage="frame_output",
            ),
        ),
    ):
        result = run_v3(
            localized.source_url,
            localized.query,
            tmp_path / "work",
            tmp_path / "output",
            tmp_path / "models",
        )

    assert result.verification_source == "asr"
    assert result.frame_match_type == "spoken_dialogue"
    assert result.frame == localized.frame
    assert result.frame.path.is_file()


def test_optional_cache_write_failure_does_not_abort_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    original_write_text = Path.write_text

    def fail_cache_write(path: Path, *args, **kwargs):
        if path.name.endswith(".part"):
            raise OSError("disk full")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_cache_write)
    transcription = Transcription(
        "target",
        [TranscriptWord("target", 1.0, 1.2)],
        "en",
        1.0,
    )
    cached = CachedTranscriber(lambda _: transcription, JsonFileCache(tmp_path / "cache"), "base")

    assert cached(audio) == transcription
    assert not list((tmp_path / "cache").rglob("*.part"))


@pytest.mark.parametrize(
    ("video", "audio", "code"),
    [
        (False, True, "missing_video_stream"),
        (True, False, "missing_audio_stream"),
        (False, False, "missing_video_stream"),
    ],
)
def test_missing_streams_fail_before_asr_loading(
    tmp_path: Path, video: bool, audio: bool, code: str
) -> None:
    media_path = tmp_path / "media.bin"
    media_path.write_bytes(b"media")
    with (
        patch("dialogue_locator.pipeline.require_external_tools", return_value=SimpleNamespace(ffprobe="ffprobe", ffmpeg="ffmpeg")),
        patch("dialogue_locator.pipeline.acquire_media", return_value=(media_path, {})),
        patch("dialogue_locator.pipeline._inspect_media_cached", return_value=(_media(video=video, audio=audio), False)),
        patch("dialogue_locator.pipeline._create_transcriber") as create_transcriber,
    ):
        with pytest.raises(V0Error) as raised:
            run_v1(
                "https://example.test/video",
                "target",
                tmp_path / "work",
                tmp_path / "output",
                tmp_path / "models",
            )

    assert raised.value.code == code
    create_transcriber.assert_not_called()


def test_ffprobe_non_object_payload_is_structured_invalid_media(tmp_path: Path) -> None:
    completed = Mock(returncode=0, stdout="[]", stderr="")
    with patch("dialogue_locator.inspection.subprocess.run", return_value=completed):
        with pytest.raises(V0Error) as raised:
            inspect_media(tmp_path / "invalid.media")

    assert raised.value.code == "invalid_media"
    assert raised.value.stage == "media_inspection"


@pytest.mark.parametrize(
    "failure",
    [
        requests.exceptions.Timeout("timeout"),
        requests.exceptions.SSLError("bad certificate"),
    ],
)
def test_direct_network_failures_are_structured_and_leave_no_partial(
    tmp_path: Path, failure: requests.RequestException
) -> None:
    with patch("dialogue_locator.acquisition.requests.get", side_effect=failure):
        with pytest.raises(V0Error) as raised:
            download_direct_media("https://example.test/video", tmp_path)

    assert raised.value.code == "acquisition_failed"
    assert raised.value.stage == "acquisition"
    assert not list(tmp_path.glob("*.part"))


def test_locked_partial_cleanup_does_not_mask_network_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_unlink = Path.unlink

    def locked_partial(path: Path, *args, **kwargs):
        if path.name.endswith(".part"):
            raise PermissionError("locked by scanner")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_partial)
    with patch(
        "dialogue_locator.acquisition.requests.get",
        side_effect=requests.exceptions.Timeout("provider timeout"),
    ):
        with pytest.raises(V0Error, match="provider timeout") as raised:
            download_direct_media("https://example.test/video", tmp_path)

    assert raised.value.code == "acquisition_failed"


def test_cli_operational_error_is_structured_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    failure = V0Error("The acquired media does not contain an audio stream.")
    with patch("dialogue_locator.cli.run_v1", side_effect=failure):
        status = cli.v1_main(["https://example.test/video", "target"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert status == 1
    assert payload == {
        "error": str(failure),
        "error_code": "missing_audio_stream",
        "error_stage": "media_inspection",
    }
    assert "Traceback" not in captured.err


def test_cli_configuration_failure_invalidates_stale_authoritative_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    stale = output / DIALOGUE_FRAME_FILENAME
    stale.write_bytes(b"old result")

    with patch("dialogue_locator.cli.run_v2") as runner:
        status = cli.v2_main(
            [
                "https://example.test/video",
                "target",
                "--output-dir",
                str(output),
                "--fuzzy-threshold",
                "40",
                "--precision-trigger-threshold",
                "45",
            ]
        )

    payload = json.loads(capsys.readouterr().err)
    assert status == 1
    assert payload["error_code"] == "invalid_configuration"
    assert not stale.exists()
    runner.assert_not_called()


class _FakeFrame:
    def __init__(self, pts: int | None, time_base: Fraction) -> None:
        self.pts = pts
        self.time_base = time_base

    def to_image(self) -> _Image:
        return _Image(str(self.pts).encode())


class _FakeContainer:
    def __init__(self, frames: list[_FakeFrame], time_base: Fraction) -> None:
        self._frames = frames
        self.stream = SimpleNamespace(time_base=time_base)
        self.streams = SimpleNamespace(video=[self.stream])
        self.seek_calls: list[tuple[int, object, bool, bool]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def seek(self, offset: int, *, stream: object, backward: bool, any_frame: bool) -> None:
        self.seek_calls.append((offset, stream, backward, any_frame))

    def decode(self, stream: object):
        assert stream is self.stream
        return iter(self._frames)


@pytest.mark.parametrize(
    ("pts_values", "target", "expected_pts"),
    [
        ([1000, 1350, 2100], 1.4, 2100),
        ([1000, 1400, 2100], 1.4, 1400),
        ([None, 10000, 10500, 11100], 10.5, 10500),
    ],
)
def test_frame_resolution_uses_actual_pts_for_vfr_and_nonzero_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pts_values: list[int | None],
    target: float,
    expected_pts: int,
) -> None:
    time_base = Fraction(1, 1000)
    container = _FakeContainer([_FakeFrame(pts, time_base) for pts in pts_values], time_base)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: container))

    result = resolve_frame_at_timestamp(tmp_path / "video.mkv", target, tmp_path / "output")

    assert result.pts == expected_pts
    assert result.timestamp == float(expected_pts * time_base)
    assert result.timestamp >= target
    assert result.path.read_bytes() == str(expected_pts).encode()
    assert container.seek_calls


def test_success_result_frame_fields_and_provenance_are_consistent(tmp_path: Path) -> None:
    frame_path = tmp_path / DIALOGUE_FRAME_FILENAME
    frame_path.write_bytes(b"png")
    result = V1Result(
        "https://example.test/video",
        tmp_path / "video.mkv",
        "target",
        DialogueMatch("detected target", 2.0, 2.5, "fuzzy", 90.0),
        ResolvedFrame(3, 2100, "1/1000", 2.1, frame_path),
        "base.en",
        localization_source="caption",
        verification_source="asr",
        caption_matched_text="target",
        caption_match_type="exact",
        caption_match_score=100.0,
        asr_model_used="base.en",
    )

    payload = result.to_dict()

    assert payload["matched_text"] == "detected target"
    assert payload["frame_path"] == str(frame_path)
    assert frame_path.is_file()
    assert float(payload["frame_pts"] * Fraction(payload["frame_time_base"])) == payload["frame_timestamp"]
    assert payload["localization_source"] == "caption"
    assert payload["verification_source"] == "asr"
    assert payload["confidence"] in {"HIGH", "MEDIUM", "LOW"}
