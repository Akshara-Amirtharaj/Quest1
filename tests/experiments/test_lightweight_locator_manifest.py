from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_lightweight_locator.manifest import load_locator_manifest


def _manifest(path: Path, locator: object) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "lightweight_locator": locator,
                "cases": [
                    {
                        "id": "one",
                        "url": "https://example.test/video",
                        "target": "target phrase",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_locator_model_and_margins_are_configurable(tmp_path: Path) -> None:
    loaded = load_locator_manifest(
        _manifest(
            tmp_path / "manifest.json",
            {
                "model": "tiny",
                "candidate_margin_before_seconds": 2.5,
                "candidate_margin_after_seconds": 3,
            },
        )
    )

    assert loaded.locator.model == "tiny"
    assert loaded.locator.candidate_margin_before_seconds == 2.5
    assert loaded.locator.candidate_margin_after_seconds == 3


@pytest.mark.parametrize(
    "locator",
    [
        [],
        {"model": ""},
        {"candidate_margin_before_seconds": -1},
        {"candidate_margin_after_seconds": "nan"},
    ],
)
def test_invalid_locator_configuration_is_rejected(tmp_path: Path, locator: object) -> None:
    with pytest.raises(ManifestError):
        load_locator_manifest(_manifest(tmp_path / "invalid.json", locator))
