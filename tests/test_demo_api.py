from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dialogue_locator.errors import V0Error
from dialogue_locator.models import DialogueMatch, ResolvedFrame, V1Result
from dialogue_locator_demo.api import create_app


PUBLIC_MESSAGES = {
    "VIDEO_URL_REQUIRED": "Enter a video URL to continue.",
    "INVALID_VIDEO_URL": "Enter a valid public video URL.",
    "DIALOGUE_REQUIRED": "Enter the dialogue you want to find.",
    "VIDEO_UNAVAILABLE": (
        "We couldn't access this video. Check the URL and make sure the video is publicly available."
    ),
    "DIALOGUE_NOT_FOUND": (
        "We couldn't find this dialogue in the video. Try a slightly different phrase."
    ),
    "PROCESSING_FAILED": "Something went wrong while processing the video. Please try again.",
}


def _result(frame_path: Path) -> V1Result:
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"png")
    return V1Result(
        source_url="https://example.test/video",
        media_path=frame_path.parent / "video.mkv",
        query="Elementary, my dear Watson.",
        match=DialogueMatch("Elementary my dear Watson", 12.25, 13.8, "exact", 100.0),
        frame=ResolvedFrame(42, 1102500, "1/90000", 12.25, frame_path),
        model="base.en",
        localization_source="caption",
        verification_source="asr",
        audio_processed_seconds=7.5,
        caption_matched_text="Elementary, my dear Watson.",
        caption_match_type="exact",
        caption_match_score=100.0,
        occurrences=(DialogueMatch("Elementary my dear Watson", 12.25, 13.8, "exact", 100.0),),
        transcription_language="en",
    )


def test_find_adapts_v4_result_and_serves_opaque_frame_url(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    result = _result(tmp_path / "generated" / "frame.png")

    with patch("dialogue_locator_demo.api.run_v2", return_value=result) as run:
        response = client.post(
            "/api/find",
            json={
                "video_url": "https://example.test/video",
                "dialogue": "Elementary, my dear Watson.",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["frame"]["url"].startswith("/api/frames/")
    assert "frame_path" not in response.text
    assert "media_path" not in response.text
    assert payload["frame"]["index"] == 42
    assert payload["match"]["confidence"] == "HIGH"
    assert payload["evidence"]["localization_source"] == "caption"
    assert payload["evidence"]["verification_source"] == "asr"
    assert payload["evidence"]["frame_match_type"] == "spoken_dialogue"
    assert payload["details"]["ocr_model"] is None
    assert payload["details"]["ocr_processed_frames"] == 0
    assert payload["processing"]["elapsed_seconds"] >= 0
    assert client.get(payload["frame"]["url"]).content == b"png"
    _, kwargs = run.call_args
    assert kwargs["language"] is None
    assert kwargs["output_dir"].parent == (tmp_path / "data" / "runs").resolve()


def test_internal_and_storage_errors_are_sanitized(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    request = {"video_url": "https://example.test/video", "dialogue": "target"}

    with patch(
        "dialogue_locator_demo.api.run_v2",
        side_effect=OSError(r"C:\\Users\\secret\\frame.png is locked"),
    ):
        storage = client.post("/api/find", json=request)
    assert storage.status_code == 500
    assert storage.json()["detail"] == {
        "code": "PROCESSING_FAILED",
        "message": PUBLIC_MESSAGES["PROCESSING_FAILED"],
    }
    assert "C:\\Users" not in storage.text

    with patch("dialogue_locator_demo.api.run_v2", side_effect=RuntimeError("secret detail")):
        internal = client.post("/api/find", json=request)
    assert internal.status_code == 500
    assert internal.json()["detail"] == {
        "code": "PROCESSING_FAILED",
        "message": PUBLIC_MESSAGES["PROCESSING_FAILED"],
    }
    assert "secret detail" not in internal.text


def test_known_pipeline_errors_are_structured(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    cases = [
        ("Dialogue not found in the spoken-audio transcription.", "DIALOGUE_NOT_FOUND", 404),
        ("The acquired media does not contain an audio stream.", "VIDEO_UNAVAILABLE", 422),
        ("The acquired media does not contain a video stream.", "VIDEO_UNAVAILABLE", 422),
        ("yt-dlp could not acquire the media: expired", "VIDEO_UNAVAILABLE", 422),
    ]
    for message, code, status in cases:
        with patch("dialogue_locator_demo.api.run_v2", side_effect=V0Error(message)):
            response = client.post(
                "/api/find",
                json={"video_url": "https://example.test/video", "dialogue": "target"},
            )
        assert response.status_code == status
        assert response.json()["detail"] == {"code": code, "message": PUBLIC_MESSAGES[code]}


def test_request_validation_has_stable_public_errors(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    cases = [
        ({"dialogue": "target"}, "VIDEO_URL_REQUIRED"),
        ({"video_url": "   ", "dialogue": "target"}, "VIDEO_URL_REQUIRED"),
        ({"video_url": "not a URL", "dialogue": "target"}, "INVALID_VIDEO_URL"),
        ({"video_url": "ftp://example.test/video", "dialogue": "target"}, "INVALID_VIDEO_URL"),
        ({"video_url": "https://example.test/video"}, "DIALOGUE_REQUIRED"),
        ({"video_url": "https://example.test/video", "dialogue": " \t "}, "DIALOGUE_REQUIRED"),
    ]
    with patch("dialogue_locator_demo.api.run_v2") as run:
        for request, code in cases:
            response = client.post("/api/find", json=request)
            assert response.status_code == 422
            assert response.json()["detail"] == {"code": code, "message": PUBLIC_MESSAGES[code]}
    run.assert_not_called()


def test_technical_pipeline_details_and_ansi_never_cross_api_boundary(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    technical = (
        "\x1b[31myt-dlp HTTPError 404 for URL https://example.test/video "
        "at C:\\Users\\secret\\video.part; "
        "try --cookies-from-browser edge\x1b[0m"
    )
    with patch("dialogue_locator_demo.api.run_v2", side_effect=V0Error(technical)):
        response = client.post(
            "/api/find",
            json={"video_url": "https://example.test/video", "dialogue": "target"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "VIDEO_UNAVAILABLE",
        "message": PUBLIC_MESSAGES["VIDEO_UNAVAILABLE"],
    }
    for forbidden in ("yt-dlp", "HTTPError", "cookies-from-browser", "C:\\Users", "\x1b"):
        assert forbidden not in response.text


def test_frontend_uses_stable_error_codes_and_preserves_success_metadata(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    script = client.get("/static/app.js").text
    page = client.get("/").text

    for code, message in PUBLIC_MESSAGES.items():
        assert code in script
        assert message in script
    assert "payload.detail?.message" not in script
    assert "clearError()" in script
    assert "videoUrl.addEventListener('input'" in script
    assert "dialogue.addEventListener('input'" in script
    assert "setText('#result-localized', titleCase(result.evidence.localization_source))" in script
    assert "setText('#result-verified', titleCase(result.evidence.verification_source))" in script
    assert "V4 pipeline" not in page
    assert "Caption localization · ASR verification · PTS frame resolution" not in page


def test_health_and_invalid_frame_id(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json() == {"status": "ok"}
    frontend = client.get("/")
    assert frontend.status_code == 200
    assert "Dialogue frame locator" in frontend.text
    assert 'id="language"' not in frontend.text
    assert 'id="result-start"' in frontend.text
    assert 'id="result-end"' in frontend.text
    assert 'id="confidence-summary"' in frontend.text
    assert client.get("/static/app.js").status_code == 200
    request_fields = client.get("/openapi.json").json()["components"]["schemas"]["FindRequest"]["properties"]
    assert set(request_fields) == {"video_url", "dialogue"}
    response = client.get("/api/frames/not-safe")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "FRAME_NOT_FOUND"
