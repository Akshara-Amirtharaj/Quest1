from dialogue_locator.inspection import parse_ffprobe, parse_rational


def test_parse_rational() -> None:
    assert parse_rational("30000/1001") == 30000 / 1001
    assert parse_rational("25/1") == 25.0
    assert parse_rational("0/0") is None
    assert parse_rational(None) is None
    assert parse_rational("not-a-rate") is None


def test_parse_complete_ffprobe_payload() -> None:
    info = parse_ffprobe(
        {
            "format": {"duration": "12.5"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "30000/1001",
                    "r_frame_rate": "30/1",
                    "time_base": "1/90000",
                    "start_time": "0.033",
                },
                {"index": 1, "codec_type": "audio", "codec_name": "aac", "start_time": "0.021"},
                {
                    "index": 2,
                    "codec_type": "subtitle",
                    "codec_name": "webvtt",
                    "tags": {"language": "eng", "title": "English"},
                },
            ],
        }
    )

    assert info.duration == 12.5
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (1280, 720)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.video_time_base == "1/90000"
    assert info.video_start_time == 0.033
    assert info.audio_start_time == 0.021
    assert info.embedded_subtitles[0]["language"] == "eng"


def test_missing_fields_are_safe() -> None:
    info = parse_ffprobe({"streams": [{"codec_type": "video"}], "format": {}})
    assert info.has_video is True
    assert info.has_audio is False
    assert info.duration is None
    assert info.width is None
    assert info.audio_codec is None
    assert info.audio_start_time is None
    assert info.embedded_subtitles == []


def test_missing_streams_are_safe() -> None:
    info = parse_ffprobe({})
    assert not info.has_video
    assert not info.has_audio
    assert info.avg_frame_rate is None
