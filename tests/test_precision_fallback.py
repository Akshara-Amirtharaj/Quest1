from pathlib import Path
from unittest.mock import patch

import pytest

from dialogue_locator.caption_verification import verify_caption_candidates
from dialogue_locator.config import V2Config
from dialogue_locator.errors import V0Error
from dialogue_locator.matching import DEFAULT_FUZZY_THRESHOLD, find_dialogue
from dialogue_locator.models import (
    CaptionCandidate,
    DialogueMatch,
    TranscriptWord,
    Transcription,
)
from dialogue_locator.precision import PrecisionFallbackTranscriber
from dialogue_locator.precision import (
    DEFAULT_PRECISION_TRIGGER_THRESHOLD,
    precision_fallback_eligible,
)


def _transcription(*tokens: str) -> Transcription:
    words = [
        TranscriptWord(token, index * 0.25, (index + 1) * 0.25, 0.9)
        for index, token in enumerate(tokens)
    ]
    return Transcription(" ".join(tokens), words, "en", 0.99)


class _Transcriber:
    def __init__(self, transcription: Transcription) -> None:
        self.transcription = transcription
        self.calls: list[Path] = []
        self.last_cache_hit = False

    def __call__(self, audio_path: Path) -> Transcription:
        self.calls.append(audio_path)
        return self.transcription


def _fallback(
    base: _Transcriber,
    precision: _Transcriber,
    query: str = "How's it looking, Barley?",
    trigger: float = DEFAULT_PRECISION_TRIGGER_THRESHOLD,
) -> tuple[PrecisionFallbackTranscriber, list[int]]:
    factory_calls: list[int] = []

    def factory() -> _Transcriber:
        factory_calls.append(1)
        return precision

    return (
        PrecisionFallbackTranscriber(
            query=query,
            fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD,
            precision_trigger_threshold=trigger,
            base_transcriber=base,
            base_model_name="base.en",
            precision_transcriber_factory=factory,
            precision_model_name="distil-large-v3",
            scope="candidate_window",
        ),
        factory_calls,
    )


def test_base_success_skips_and_does_not_lazy_load_distil() -> None:
    base = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    precision = _Transcriber(_transcription("unused"))
    fallback, factory_calls = _fallback(base, precision)

    result = fallback(Path("window.wav"))

    assert result is base.transcription
    assert factory_calls == []
    assert precision.calls == []
    assert fallback.precision_loaded is False
    assert fallback.precision_fallback_used is False
    assert fallback.precision_fallback_eligible is False
    assert fallback.asr_model_used == "base.en"
    assert fallback.base_match_score == 100.0
    assert fallback.precision_match_score is None


def test_base_failure_triggers_distil_and_returns_precision_match() -> None:
    base = _Transcriber(_transcription("Is", "it", "looking,", "Bali?"))
    precision = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    fallback, factory_calls = _fallback(base, precision)

    result = fallback(Path("window.wav"))
    match = find_dialogue(
        "How's it looking, Barley?",
        result.words,
        DEFAULT_FUZZY_THRESHOLD,
    )

    assert factory_calls == [1]
    assert fallback.precision_loaded is True
    assert fallback.precision_fallback_used is True
    assert fallback.precision_fallback_eligible is True
    assert fallback.precision_trigger_threshold == 45.0
    assert fallback.precision_scope == "candidate_window"
    assert fallback.asr_model_used == "distil-large-v3"
    assert fallback.base_match_score is not None
    assert fallback.base_match_score == pytest.approx(78.0487804878)
    assert fallback.base_match_score < DEFAULT_FUZZY_THRESHOLD
    assert fallback.precision_match_score == 100.0
    assert match.matched_text == "How's it looking, Barley?"


def test_distil_failure_preserves_normal_no_match_semantics() -> None:
    base = _Transcriber(_transcription("Is", "it", "looking,", "Bali?"))
    precision = _Transcriber(_transcription("Nothing", "related", "was", "said."))
    fallback, _ = _fallback(base, precision)

    result = fallback(Path("window.wav"))

    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue(
            "How's it looking, Barley?",
            result.words,
            DEFAULT_FUZZY_THRESHOLD,
        )
    assert fallback.precision_fallback_used is True
    assert fallback.precision_match_score is not None
    assert fallback.precision_match_score < DEFAULT_FUZZY_THRESHOLD


def test_absent_target_does_not_become_a_precision_false_positive() -> None:
    base = _Transcriber(_transcription("You"))
    precision = _Transcriber(_transcription("Thank", "you."))
    fallback, factory_calls = _fallback(
        base,
        precision,
        query="Please remember to close the laboratory door.",
    )

    result = fallback(Path("window.wav"))

    assert factory_calls == []
    assert fallback.precision_fallback_used is False
    assert fallback.precision_fallback_eligible is False
    assert fallback.precision_fallback_skip_reason == (
        "base_match_score_below_precision_trigger"
    )
    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue(
            "Please remember to close the laboratory door.",
            result.words,
            DEFAULT_FUZZY_THRESHOLD,
        )


def test_original_selected_transcript_text_is_preserved() -> None:
    base = _Transcriber(
        _transcription("The", "company", "reported", "revenue", "of", "$20", "million.")
    )
    precision = _Transcriber(_transcription("unused"))
    query = "The company reported revenue of twenty million dollars."
    fallback, _ = _fallback(base, precision, query=query)

    transcription = fallback(Path("window.wav"))
    match = find_dialogue(query, transcription.words, DEFAULT_FUZZY_THRESHOLD)

    assert transcription.text == "The company reported revenue of $20 million."
    assert match.matched_text == "The company reported revenue of $20 million."
    assert fallback.precision_fallback_used is False


def test_existing_threshold_is_unchanged_and_controls_fallback() -> None:
    base = _Transcriber(_transcription("Is", "it", "looking,", "Bali?"))
    precision = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    fallback, _ = _fallback(base, precision)

    fallback(Path("window.wav"))

    assert DEFAULT_FUZZY_THRESHOLD == 85.0
    assert fallback.fuzzy_threshold == 85.0
    assert fallback.base_match_score is not None
    assert fallback.base_match_score < 85.0
    assert fallback.precision_fallback_used is True


def test_low_score_candidate_skips_precision_without_becoming_a_match() -> None:
    base = _Transcriber(
        _transcription("The", "company", "reported", "unrelated", "results.")
    )
    precision = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    fallback, factory_calls = _fallback(base, precision)

    result = fallback(Path("window.wav"))

    assert fallback.base_match_score is not None
    assert fallback.base_match_score < fallback.precision_trigger_threshold
    assert fallback.precision_fallback_eligible is False
    assert fallback.precision_fallback_used is False
    assert fallback.precision_loaded is False
    assert factory_calls == []
    with pytest.raises(V0Error, match="Dialogue not found"):
        find_dialogue(
            "How's it looking, Barley?",
            result.words,
            DEFAULT_FUZZY_THRESHOLD,
        )


def test_explicit_full_audio_precision_is_not_suppressed_by_candidate_gate() -> None:
    base = _Transcriber(
        _transcription("The", "company", "reported", "unrelated", "results.")
    )
    precision = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    factory_calls: list[int] = []

    def factory() -> _Transcriber:
        factory_calls.append(1)
        return precision

    fallback = PrecisionFallbackTranscriber(
        query="How's it looking, Barley?",
        fuzzy_threshold=85.0,
        precision_trigger_threshold=45.0,
        base_transcriber=base,
        base_model_name="base.en",
        precision_transcriber_factory=factory,
        precision_model_name="distil-large-v3",
        scope="full_audio",
    )

    fallback(Path("full-audio.wav"))

    assert fallback.base_match_score is not None
    assert fallback.base_match_score < 45.0
    assert fallback.precision_fallback_eligible is True
    assert fallback.precision_fallback_used is True
    assert factory_calls == [1]


def test_precision_trigger_and_match_threshold_equality_boundaries() -> None:
    assert precision_fallback_eligible(45.0, 45.0, 85.0) is True
    assert precision_fallback_eligible(44.999, 45.0, 85.0) is False
    assert precision_fallback_eligible(85.0, 45.0, 85.0) is False


def test_score_equal_to_match_threshold_is_base_accepted() -> None:
    base = _Transcriber(_transcription("borderline", "base", "transcript"))
    precision = _Transcriber(_transcription("unused"))
    fallback, factory_calls = _fallback(base, precision)

    with (
        patch("dialogue_locator.precision._best_score", return_value=85.0),
        patch(
            "dialogue_locator.precision.find_dialogue",
            return_value=DialogueMatch("borderline", 0.0, 1.0, "fuzzy", 85.0),
        ),
    ):
        result = fallback(Path("window.wav"))

    assert result is base.transcription
    assert fallback.base_match_score == 85.0
    assert fallback.precision_fallback_used is False
    assert fallback.precision_fallback_eligible is False
    assert factory_calls == []


def test_score_equal_to_precision_trigger_runs_precision() -> None:
    base = _Transcriber(_transcription("Is", "it", "looking,", "Bali?"))
    precision = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    trigger = 78.04878048780488
    fallback, factory_calls = _fallback(base, precision, trigger=trigger)

    fallback(Path("window.wav"))

    assert fallback.base_match_score == pytest.approx(trigger)
    assert fallback.precision_fallback_eligible is True
    assert fallback.precision_fallback_used is True
    assert factory_calls == [1]


def test_precision_configuration_defaults_to_bounded_only() -> None:
    config = V2Config()

    assert config.asr_precision_fallback is True
    assert config.precision_asr_model == "distil-large-v3"
    assert config.precision_trigger_threshold == 45.0
    assert config.full_audio_precision_fallback is False

    with pytest.raises(ValueError, match="requires asr_precision_fallback"):
        V2Config(
            asr_precision_fallback=False,
            full_audio_precision_fallback=True,
        )
    with pytest.raises(ValueError, match="less than or equal"):
        V2Config(
            verification_fuzzy_threshold=85.0,
            precision_trigger_threshold=85.01,
        )


def test_caption_precision_retry_reuses_the_same_bounded_audio_window(tmp_path: Path) -> None:
    base = _Transcriber(_transcription("Is", "it", "looking,", "Bali?"))
    precision = _Transcriber(_transcription("How's", "it", "looking,", "Barley?"))
    fallback, _ = _fallback(base, precision)
    extracted: list[tuple[float | None, float | None]] = []

    def extract(
        _: Path,
        output: Path,
        __: str,
        *,
        start_time: float | None = None,
        duration: float | None = None,
    ) -> Path:
        extracted.append((start_time, duration))
        output.write_bytes(b"audio")
        return output

    candidate = CaptionCandidate(
        "How's it looking, Barley?",
        10.0,
        12.0,
        "exact",
        100.0,
        "en",
        "manual",
    )
    verification, processed = verify_caption_candidates(
        [candidate],
        "How's it looking, Barley?",
        tmp_path / "media.mp4",
        30.0,
        0.0,
        tmp_path,
        "ffmpeg",
        fallback,
        V2Config(verification_margins=(2.0,)),
        audio_extractor=extract,
    )

    assert verification is not None
    assert extracted == [(8.0, 6.0)]
    assert base.calls == [tmp_path / "verification.wav"]
    assert precision.calls == [tmp_path / "verification.wav"]
    assert processed == 12.0
    assert verification.match.matched_text == "How's it looking, Barley?"


def test_low_score_caption_candidate_does_not_block_later_valid_candidate(
    tmp_path: Path,
) -> None:
    transcriptions = iter(
        (
            _transcription("Completely", "unrelated", "window", "content."),
            _transcription("How's", "it", "looking,", "Barley?"),
        )
    )

    class SequenceTranscriber:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def __call__(self, audio_path: Path) -> Transcription:
            self.calls.append(audio_path)
            return next(transcriptions)

    base = SequenceTranscriber()
    precision = _Transcriber(_transcription("unused"))
    factory_calls: list[int] = []

    def factory() -> _Transcriber:
        factory_calls.append(1)
        return precision

    fallback = PrecisionFallbackTranscriber(
        query="How's it looking, Barley?",
        fuzzy_threshold=85.0,
        precision_trigger_threshold=45.0,
        base_transcriber=base,
        base_model_name="base.en",
        precision_transcriber_factory=factory,
        precision_model_name="distil-large-v3",
        scope="candidate_window",
    )

    def extract(_: Path, output: Path, __: str, **___: float) -> Path:
        output.write_bytes(b"audio")
        return output

    candidates = [
        CaptionCandidate("wrong", 2.0, 3.0, "fuzzy", 90.0, "en", "manual"),
        CaptionCandidate(
            "How's it looking, Barley?",
            10.0,
            12.0,
            "exact",
            100.0,
            "en",
            "manual",
        ),
    ]
    verification, processed = verify_caption_candidates(
        candidates,
        "How's it looking, Barley?",
        tmp_path / "media.mp4",
        30.0,
        0.0,
        tmp_path,
        "ffmpeg",
        fallback,
        V2Config(verification_margins=(0.0,)),
        audio_extractor=extract,
    )

    assert verification is not None
    assert verification.candidate is candidates[1]
    assert len(base.calls) == 2
    assert factory_calls == []
    assert precision.calls == []
    assert processed == 3.0
