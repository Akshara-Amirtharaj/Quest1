from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dialogue_locator.errors import V0Error
from dialogue_locator.models import DialogueMatch, ResolvedFrame, V1Result
from dialogue_locator_demo.api import create_app


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
    assert storage.json()["detail"]["code"] == "STORAGE_ERROR"
    assert "C:\\Users" not in storage.text

    with patch("dialogue_locator_demo.api.run_v2", side_effect=RuntimeError("secret detail")):
        internal = client.post("/api/find", json=request)
    assert internal.status_code == 500
    assert internal.json()["detail"] == {
        "code": "INTERNAL_ERROR",
        "message": "The backend encountered an internal error. Please retry.",
    }
    assert "secret detail" not in internal.text


def test_known_pipeline_errors_are_structured(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    cases = [
        ("Dialogue not found in the spoken-audio transcription.", "NO_MATCH", 404),
        ("The acquired media does not contain an audio stream.", "NO_AUDIO", 422),
        ("The acquired media does not contain a video stream.", "NO_VIDEO", 422),
        ("yt-dlp could not acquire the media: expired", "MEDIA_UNAVAILABLE", 422),
    ]
    for message, code, status in cases:
        with patch("dialogue_locator_demo.api.run_v2", side_effect=V0Error(message)):
            response = client.post(
                "/api/find",
                json={"video_url": "https://example.test/video", "dialogue": "target"},
            )
        assert response.status_code == status
        assert response.json()["detail"] == {"code": code, "message": message}


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
