from pathlib import Path

from dialogue_locator.models import (
    DialogueMatch,
    ResolvedFrame,
    V1Result,
    format_timestamp,
)


def test_format_timestamp_uses_hours_minutes_seconds_and_milliseconds() -> None:
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(210.895) == "00:03:30.895"
    assert format_timestamp(2122.267) == "00:35:22.267"
    assert format_timestamp(3661.005) == "01:01:01.005"
    assert format_timestamp(None) is None


def test_v1_result_keeps_numeric_and_human_readable_timestamps() -> None:
    result = V1Result(
        "https://example.test/video",
        Path("video.mkv"),
        "target",
        DialogueMatch("target", 2122.267, 2124.787, "exact", 100.0),
        ResolvedFrame(1, 2122287, "1/1000", 2122.287, Path("frame.png")),
        "base.en",
        audio_processed_seconds=7.17,
    ).to_dict()

    assert result["dialogue_start"] == 2122.267
    assert result["dialogue_start_hms"] == "00:35:22.267"
    assert result["dialogue_end_hms"] == "00:35:24.787"
    assert result["frame_timestamp_hms"] == "00:35:22.287"
    assert result["audio_processed_hms"] == "00:00:07.170"
