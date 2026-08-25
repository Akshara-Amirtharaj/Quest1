from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_chunked.manifest import (
    load_chunked_manifest,
    override_chunk_config,
)
from experiments.audio_localization_chunked.metrics import (
    percentage_asr_audio_avoided,
    speedup_ratio,
    timestamp_delta,
)


def _write_manifest(path: Path, chunked: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "chunked_asr": chunked,
                "cases": [
                    {
                        "id": "case-one",
                        "url": "https://example.test/video",
                        "target": "target words",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_chunk_configuration_parses_and_cli_values_override_it(tmp_path: Path) -> None:
    loaded = load_chunked_manifest(
        _write_manifest(
            tmp_path / "manifest.json",
            {"chunk_duration_seconds": 12, "overlap_seconds": 3},
        )
    )

    assert loaded.chunked_asr.chunk_duration_seconds == 12
    assert loaded.chunked_asr.overlap_seconds == 3
    overridden = override_chunk_config(
        loaded.chunked_asr,
        chunk_duration_seconds=8,
        overlap_seconds=2,
    )
    assert overridden.chunk_duration_seconds == 8
    assert overridden.overlap_seconds == 2


@pytest.mark.parametrize(
    "config",
    [
        {"chunk_duration_seconds": 0, "overlap_seconds": 0},
        {"chunk_duration_seconds": 10, "overlap_seconds": -1},
        {"chunk_duration_seconds": 10, "overlap_seconds": 10},
    ],
)
def test_invalid_chunk_configuration_is_rejected(tmp_path: Path, config: dict) -> None:
    with pytest.raises(ManifestError):
        load_chunked_manifest(_write_manifest(tmp_path / "invalid.json", config))


def test_chunked_comparison_metrics_are_deterministic() -> None:
    assert percentage_asr_audio_avoided(8.0, 25.0) == 68.0
    assert speedup_ratio(16.0, 8.0) == 2.0
    assert timestamp_delta(2.18, 2.16) == pytest.approx(0.02)
    assert timestamp_delta(None, 2.16) is None
