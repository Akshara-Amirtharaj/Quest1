from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import ManifestError
from experiments.audio_localization_chunked_vad.manifest import load_vad_manifest


def _manifest(path: Path, vad: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "chunked_asr": {"chunk_duration_seconds": 8, "overlap_seconds": 2},
                "vad": vad,
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


def test_conservative_vad_manifest_parameters_are_configurable(tmp_path: Path) -> None:
    loaded = load_vad_manifest(
        _manifest(
            tmp_path / "vad.json",
            {
                "threshold": 0.3,
                "neg_threshold": 0.15,
                "min_speech_duration_ms": 10,
                "min_silence_duration_ms": 1200,
                "speech_pad_ms": 600,
                "clear_silence_rms_threshold": 0.001,
                "max_removed_fraction_before_fallback": 0.95,
                "fallback_on_no_match": True,
                "fallback_on_baseline_mismatch": True,
            },
        )
    )

    assert loaded.vad.threshold == 0.3
    assert loaded.vad.min_silence_duration_ms == 1200
    assert loaded.vad.speech_pad_ms == 600
    assert loaded.chunked.chunked_asr.chunk_duration_seconds == 8


@pytest.mark.parametrize(
    "vad",
    [
        {"threshold": 1.0},
        {"threshold": 0.3, "neg_threshold": 0.4},
        {"speech_pad_ms": -1},
        {"max_removed_fraction_before_fallback": 1.1},
    ],
)
def test_invalid_vad_configuration_is_rejected(tmp_path: Path, vad: dict) -> None:
    with pytest.raises(ManifestError):
        load_vad_manifest(_manifest(tmp_path / "invalid.json", vad))
