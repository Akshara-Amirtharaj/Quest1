from __future__ import annotations

import numpy as np
import pytest

from dialogue_locator.models import TranscriptWord
from experiments.audio_localization_chunked_vad.fallbacks import global_fallback_reason
from experiments.audio_localization_chunked_vad.manifest import ConservativeVADConfig
from experiments.audio_localization_chunked_vad.transcriber import build_vad_options
from experiments.audio_localization_chunked_vad.vad_filter import (
    prepare_vad_audio,
    requires_unfiltered_chunk_fallback,
    restore_vad_word_timestamps,
)


SAMPLE_RATE = 16000


def test_silence_at_start_and_end_is_removed() -> None:
    audio = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    audio[2 * SAMPLE_RATE : 8 * SAMPLE_RATE] = 0.1

    prepared = prepare_vad_audio(
        audio,
        [{"start": 2 * SAMPLE_RATE, "end": 8 * SAMPLE_RATE}],
    )

    assert prepared.original_audio_seconds == 10.0
    assert prepared.speech_audio_seconds == 6.0
    assert prepared.removed_fraction == pytest.approx(0.4)
    assert len(prepared.filtered_audio) == 6 * SAMPLE_RATE


def test_speech_padding_is_configured_and_boundary_regions_are_not_clipped() -> None:
    config = ConservativeVADConfig(speech_pad_ms=500)
    options = build_vad_options(config)
    audio = np.ones(2 * SAMPLE_RATE, dtype=np.float32) * 0.05

    prepared = prepare_vad_audio(
        audio,
        [{"start": 0, "end": len(audio)}],
    )

    assert options.speech_pad_ms == 500
    assert prepared.speech_audio_seconds == 2.0
    assert prepared.removed_fraction == 0.0


def test_long_clear_silence_is_skipped_without_uncertainty_fallback() -> None:
    prepared = prepare_vad_audio(np.zeros(20 * SAMPLE_RATE, dtype=np.float32), [])

    assert requires_unfiltered_chunk_fallback(
        prepared,
        clear_silence_rms_threshold=0.002,
        max_removed_fraction_before_fallback=0.98,
    ) is False


def test_nontrivial_audio_removed_entirely_uses_unfiltered_fallback() -> None:
    prepared = prepare_vad_audio(np.ones(SAMPLE_RATE, dtype=np.float32) * 0.02, [])

    assert requires_unfiltered_chunk_fallback(
        prepared,
        clear_silence_rms_threshold=0.002,
        max_removed_fraction_before_fallback=0.98,
    ) is True


def test_nearly_continuous_speech_keeps_audio_and_does_not_fallback() -> None:
    audio = np.ones(10 * SAMPLE_RATE, dtype=np.float32) * 0.05
    prepared = prepare_vad_audio(
        audio,
        [{"start": int(0.1 * SAMPLE_RATE), "end": int(9.9 * SAMPLE_RATE)}],
    )

    assert prepared.speech_audio_seconds == pytest.approx(9.8)
    assert prepared.removed_fraction == pytest.approx(0.02)
    assert requires_unfiltered_chunk_fallback(
        prepared,
        clear_silence_rms_threshold=0.002,
        max_removed_fraction_before_fallback=0.98,
    ) is False


def test_timestamp_restoration_after_vad_filtering() -> None:
    regions = [
        {"start": 2 * SAMPLE_RATE, "end": 4 * SAMPLE_RATE},
        {"start": 8 * SAMPLE_RATE, "end": 10 * SAMPLE_RATE},
    ]
    filtered_words = [TranscriptWord("target", 2.2, 2.6, 0.9)]

    restored = restore_vad_word_timestamps(filtered_words, regions)

    assert restored[0].start == pytest.approx(8.2)
    assert restored[0].end == pytest.approx(8.6)


def test_no_match_or_baseline_disagreement_requests_global_fallback() -> None:
    config = ConservativeVADConfig()

    assert global_fallback_reason(
        vad_found_match=False,
        vad_matches_baseline=False,
        config=config,
    ) is not None
    assert global_fallback_reason(
        vad_found_match=True,
        vad_matches_baseline=False,
        config=config,
    ) is not None
    assert global_fallback_reason(
        vad_found_match=True,
        vad_matches_baseline=True,
        config=config,
    ) is None
