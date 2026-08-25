from pathlib import Path
from unittest.mock import Mock, patch

from dialogue_locator.audio import extract_speech_audio


def test_audio_extraction_uses_mono_16khz_pcm(tmp_path: Path) -> None:
    media_path = tmp_path / "input.mkv"
    audio_path = tmp_path / "temporary" / "speech.wav"
    media_path.write_bytes(b"media")

    def complete(command: list[str], **_: object) -> Mock:
        Path(command[-1]).write_bytes(b"wav")
        return Mock(returncode=0, stderr="")

    with patch("dialogue_locator.audio.subprocess.run", side_effect=complete) as run:
        result = extract_speech_audio(media_path, audio_path, "ffmpeg-test")

    command = run.call_args.args[0]
    assert command[0] == "ffmpeg-test"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert result == audio_path.resolve()


def test_audio_extraction_accepts_a_bounded_verification_window(tmp_path: Path) -> None:
    media_path = tmp_path / "input.mkv"
    audio_path = tmp_path / "speech.wav"
    media_path.write_bytes(b"media")

    def complete(command: list[str], **_: object) -> Mock:
        Path(command[-1]).write_bytes(b"wav")
        return Mock(returncode=0, stderr="")

    with patch("dialogue_locator.audio.subprocess.run", side_effect=complete) as run:
        extract_speech_audio(media_path, audio_path, start_time=8.25, duration=6.5)

    command = run.call_args.args[0]
    assert command[command.index("-ss") + 1] == "8.250000"
    assert command[command.index("-t") + 1] == "6.500000"
