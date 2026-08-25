from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import ManifestError, load_manifest


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_parses_defaults_overrides_and_production_reference(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "benchmark.json",
        {
            "version": 1,
            "defaults": {
                "work_dir": "cache/media",
                "output_dir": "results/frames",
                "model_cache": "cache/models",
                "model": "base.en",
                "language": "en",
                "reuse_transcript_cache": False,
            },
            "cases": [
                {
                    "id": "case-one",
                    "url": "https://example.test/video",
                    "source_page_url": "https://publisher.example.test/watch/one",
                    "target": "Target words",
                    "model": "small",
                    "production_baseline": {
                        "dialogue_start_seconds": 4.25,
                        "timestamp_tolerance_seconds": 0.2,
                        "matched_text": "target words",
                    },
                }
            ],
        },
    )

    manifest = load_manifest(path)

    assert manifest.defaults.work_dir == Path("cache/media")
    assert manifest.defaults.reuse_transcript_cache is False
    assert len(manifest.cases) == 1
    case = manifest.cases[0]
    assert case.case_id == "case-one"
    assert case.source_page_url == "https://publisher.example.test/watch/one"
    assert case.model == "small"
    assert case.language == "en"
    assert case.production_baseline is not None
    assert case.production_baseline.dialogue_start_seconds == 4.25


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(version=2), "version must be 1"),
        (lambda payload: payload.update(cases=[]), "non-empty JSON array"),
        (
            lambda payload: payload["cases"].append(dict(payload["cases"][0])),
            "Duplicate case id",
        ),
        (
            lambda payload: payload["cases"][0].update(fuzzy_threshold=101),
            "at most 100",
        ),
    ],
)
def test_manifest_rejects_invalid_configuration(tmp_path: Path, mutation, message: str) -> None:
    payload = {
        "version": 1,
        "cases": [
            {
                "id": "valid-case",
                "url": "https://example.test/video",
                "target": "target words",
            }
        ],
    }
    mutation(payload)
    path = _write_manifest(tmp_path / "invalid.json", payload)

    with pytest.raises(ManifestError, match=message):
        load_manifest(path)
