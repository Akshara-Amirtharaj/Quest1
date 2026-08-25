from pathlib import Path
from unittest.mock import patch

from dialogue_locator.config import V3Config
from dialogue_locator.models import (
    CandidateVideoFrame,
    DialogueMatch,
    OCRLine,
    ResolvedFrame,
    V1Result,
)
from dialogue_locator.pipeline import run_v3


class _Image:
    def __init__(self, text: str) -> None:
        self.text = text

    def save(self, path: Path, format: str) -> None:
        Path(path).write_bytes(b"png")


class _Reader:
    model_description = "mock-paddle"

    def __init__(self, _: Path) -> None:
        pass

    def __call__(self, image: _Image) -> list[OCRLine]:
        return [OCRLine(image.text)]


def _localized(tmp_path: Path) -> V1Result:
    media_path = tmp_path / "video.mkv"
    media_path.write_bytes(b"media")
    frame_path = tmp_path / "dialogue_frame.png"
    frame_path.write_bytes(b"png")
    return V1Result(
        "https://example.test/video",
        media_path,
        "target dialogue",
        DialogueMatch("target dialogue", 5.0, 6.0, "exact", 100.0),
        ResolvedFrame(0, 5000, "1/1000", 5.0, frame_path),
        "base.en",
        localization_source="caption",
        verification_source="asr",
        audio_processed_seconds=4.0,
    )


def test_v3_selects_first_visible_frame_and_preserves_localization_source(tmp_path: Path) -> None:
    localized = _localized(tmp_path)
    frames = [
        CandidateVideoFrame(10, 4900, "1/1000", 4.9, _Image("target")),
        CandidateVideoFrame(11, 5000, "1/1000", 5.0, _Image("target dialogue")),
        CandidateVideoFrame(12, 5100, "1/1000", 5.1, _Image("target dialogue")),
    ]

    with (
        patch("dialogue_locator.pipeline.run_v2", return_value=localized),
        patch("dialogue_locator.pipeline.iter_frames_in_interval", return_value=iter(frames)) as decode,
    ):
        result = run_v3(
            localized.source_url,
            localized.query,
            tmp_path,
            tmp_path / "output",
            tmp_path / "models",
            v3_config=V3Config(search_margin=0.5),
            ocr_reader_factory=_Reader,
        )

    decode.assert_called_once_with(localized.media_path, 4.5, 6.5)
    assert result.localization_source == "caption"
    assert result.verification_source == "ocr"
    assert result.frame_match_type == "visible_text"
    assert result.frame.timestamp == 5.0
    assert result.ocr_processed_frames == 2
    assert result.frame.path.is_file()


def test_v3_no_visible_text_returns_unchanged_spoken_frame(tmp_path: Path) -> None:
    localized = _localized(tmp_path)
    frames = [CandidateVideoFrame(10, 5000, "1/1000", 5.0, _Image("other text"))]

    with (
        patch("dialogue_locator.pipeline.run_v2", return_value=localized),
        patch("dialogue_locator.pipeline.iter_frames_in_interval", return_value=iter(frames)),
    ):
        result = run_v3(
            localized.source_url,
            localized.query,
            tmp_path,
            tmp_path / "output",
            tmp_path / "models",
            ocr_reader_factory=_Reader,
        )

    assert result.verification_source == "asr"
    assert result.frame_match_type == "spoken_dialogue"
    assert result.frame == localized.frame
    assert result.ocr_processed_frames == 1


def test_v3_forwards_authentication_options_to_v2(tmp_path: Path) -> None:
    localized = _localized(tmp_path)
    cookie_file = tmp_path / "cookies.txt"

    with (
        patch("dialogue_locator.pipeline.run_v2", return_value=localized) as run_v2,
        patch("dialogue_locator.pipeline.iter_frames_in_interval", return_value=iter(())),
    ):
        run_v3(
            localized.source_url,
            localized.query,
            tmp_path,
            tmp_path / "output",
            tmp_path / "models",
            ocr_reader_factory=_Reader,
            cookies_from_browser="edge",
            cookie_file=cookie_file,
        )

    assert run_v2.call_args.kwargs["cookies_from_browser"] == "edge"
    assert run_v2.call_args.kwargs["cookie_file"] == cookie_file
