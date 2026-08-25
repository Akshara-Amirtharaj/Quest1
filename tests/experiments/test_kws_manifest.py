from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_kws.manifest import load_kws_manifest


def _manifest(path: Path, kws: object) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kws": kws,
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


def test_kws_parameters_are_configurable(tmp_path: Path) -> None:
    loaded = load_kws_manifest(
        _manifest(
            tmp_path / "kws.json",
            {
                "model_dir": ".cache/custom-kws",
                "max_anchors": 4,
                "grouping_gap_seconds": 1.5,
                "candidate_margin_before_seconds": 3,
                "candidate_margin_after_seconds": 2.5,
                "num_threads": 2,
                "keywords_score": 2.5,
                "keywords_threshold": 0.2,
                "num_trailing_blanks": 2,
            },
        )
    )

    assert loaded.kws.model_dir == Path(".cache/custom-kws")
    assert loaded.kws.max_anchors == 4
    assert loaded.kws.keywords_threshold == 0.2


@pytest.mark.parametrize(
    "kws",
    [
        [],
        {"max_anchors": 0},
        {"grouping_gap_seconds": -1},
        {"keywords_threshold": 1.1},
        {"num_threads": 0},
        {"num_trailing_blanks": -1},
    ],
)
def test_invalid_kws_configuration_is_rejected(tmp_path: Path, kws: object) -> None:
    with pytest.raises(ManifestError):
        load_kws_manifest(_manifest(tmp_path / "invalid.json", kws))
