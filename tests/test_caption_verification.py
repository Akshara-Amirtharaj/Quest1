from pathlib import Path

from dialogue_locator.caption_verification import verify_caption_candidates
from dialogue_locator.config import V2Config
from dialogue_locator.models import CaptionCandidate, TranscriptWord, Transcription


def _candidate(start: float = 10.0, end: float = 12.0) -> CaptionCandidate:
    return CaptionCandidate("target words", start, end, "exact", 100.0, "en", "manual")


def _transcription(text: str) -> Transcription:
    words = [
        TranscriptWord(word, index * 0.4, (index + 1) * 0.4)
        for index, word in enumerate(text.split())
    ]
    return Transcription(text, words, "en", 1.0)


def test_progressive_widening_stops_when_larger_window_verifies(tmp_path: Path) -> None:
    extracted: list[tuple[float, float]] = []
    transcriptions = iter([_transcription("unrelated speech"), _transcription("target words")])

    def extractor(_: Path, output: Path, __: str, **options: float) -> Path:
        extracted.append((options["start_time"], options["duration"]))
        output.write_bytes(b"wav")
        return output

    verification, processed = verify_caption_candidates(
        [_candidate()],
        "target words",
        tmp_path / "media.mp4",
        30.0,
        0.0,
        tmp_path,
        "ffmpeg",
        lambda _: next(transcriptions),
        V2Config(verification_margins=(1.0, 3.0)),
        audio_extractor=extractor,
    )

    assert verification is not None
    assert extracted == [(9.0, 4.0), (7.0, 8.0)]
    assert processed == 12.0
    assert verification.match.start == 7.0


def test_success_does_not_invoke_asr_for_later_candidates(tmp_path: Path) -> None:
    calls = 0

    def extractor(_: Path, output: Path, __: str, **___: float) -> Path:
        output.write_bytes(b"wav")
        return output

    def transcriber(_: Path) -> Transcription:
        nonlocal calls
        calls += 1
        return _transcription("target words")

    verification, _ = verify_caption_candidates(
        [_candidate(5.0, 6.0), _candidate(20.0, 21.0)],
        "target words",
        tmp_path / "media.mp4",
        30.0,
        0.0,
        tmp_path,
        "ffmpeg",
        transcriber,
        V2Config(verification_margins=(1.0, 3.0)),
        audio_extractor=extractor,
    )

    assert verification is not None
    assert calls == 1
