from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.audio_localization_baseline.manifest import BenchmarkCase, ManifestError
from experiments.audio_localization_chunked.manifest import (
    load_chunked_manifest,
    override_chunk_config,
)
from experiments.audio_localization_chunked.metrics import (
    percentage_asr_audio_avoided,
    speedup_ratio,
    timestamp_delta,
)
from experiments.audio_localization_chunked.runner import (
    MATCHING_LOGIC_ID,
    _current_environment,
    benchmark_configuration_parity,
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
            {
                "chunk_duration_seconds": 12,
                "overlap_seconds": 3,
                "transcript_context_seconds": 7,
            },
        )
    )

    assert loaded.chunked_asr.chunk_duration_seconds == 12
    assert loaded.chunked_asr.overlap_seconds == 3
    assert loaded.chunked_asr.transcript_context_seconds == 7
    overridden = override_chunk_config(
        loaded.chunked_asr,
        chunk_duration_seconds=8,
        overlap_seconds=2,
        transcript_context_seconds=6,
    )
    assert overridden.chunk_duration_seconds == 8
    assert overridden.overlap_seconds == 2
    assert overridden.transcript_context_seconds == 6


@pytest.mark.parametrize(
    "config",
    [
        {"chunk_duration_seconds": 0, "overlap_seconds": 0},
        {"chunk_duration_seconds": 10, "overlap_seconds": -1},
        {"chunk_duration_seconds": 10, "overlap_seconds": 10},
        {
            "chunk_duration_seconds": 10,
            "overlap_seconds": 2,
            "transcript_context_seconds": 0,
        },
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


def test_benchmark_configuration_parity_verifies_controlled_inputs(tmp_path: Path) -> None:
    media = tmp_path / "media.mkv"
    media.touch()
    case = BenchmarkCase(
        case_id="controlled",
        url="https://example.test/video",
        source_page_url=None,
        target="target words",
        model="base.en",
        device="cpu",
        compute_type="int8",
        language="en",
        fuzzy_threshold=85.0,
    )
    baseline = {
        "url": case.url,
        "media_path": str(media.resolve()),
        "target": case.target,
        "model": case.model,
        "device": case.device,
        "compute_type": case.compute_type,
        "language": case.language,
        "fuzzy_threshold": case.fuzzy_threshold,
        "_matching_logic": MATCHING_LOGIC_ID,
        "_benchmark_environment": _current_environment(),
        "matched_text": "different ASR spelling is diagnostic only",
    }

    parity = benchmark_configuration_parity(case, baseline, media)

    assert parity["all_identical"] is True
    baseline["fuzzy_threshold"] = 80.0
    assert benchmark_configuration_parity(case, baseline, media)["all_identical"] is False
