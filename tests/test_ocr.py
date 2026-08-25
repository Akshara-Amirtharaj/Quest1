from dialogue_locator.models import CandidateVideoFrame, OCRLine
from dialogue_locator.ocr import find_first_visible_frame, match_visible_text


def _frame(index: int, timestamp: float) -> CandidateVideoFrame:
    return CandidateVideoFrame(index, index * 100, "1/1000", timestamp, f"image-{index}")


def test_exact_visible_match_normalizes_case_punctuation_and_lines() -> None:
    match = match_visible_text(
        "My mind rebels at stagnation.",
        [OCRLine("MY MIND"), OCRLine("rebels,  at stagnation!")],
    )

    assert match is not None
    assert match.match_type == "exact"
    assert match.score == 100.0


def test_fuzzy_visible_match_accepts_minor_ocr_error() -> None:
    match = match_visible_text(
        "My mind rebels at stagnation",
        [OCRLine("My mind rebeis at stagnation")],
    )

    assert match is not None
    assert match.match_type == "fuzzy"
    assert match.score >= 85.0


def test_gradual_dialogue_stops_on_first_complete_frame() -> None:
    frames = [_frame(index, float(index)) for index in range(4)]
    visible = {
        "image-0": [OCRLine("My mind")],
        "image-1": [OCRLine("My mind rebels")],
        "image-2": [OCRLine("My mind rebels at stagnation")],
        "image-3": [OCRLine("My mind rebels at stagnation")],
    }

    frame, match, processed = find_first_visible_frame(
        "My mind rebels at stagnation",
        frames,
        lambda image: visible[image],
    )

    assert frame == frames[2]
    assert match is not None
    assert processed == 3


def test_no_visible_match_processes_candidate_frames_only() -> None:
    frames = [_frame(0, 1.0), _frame(1, 1.1)]

    frame, match, processed = find_first_visible_frame(
        "target dialogue",
        frames,
        lambda _: [OCRLine("unrelated scene text")],
    )

    assert frame is None
    assert match is None
    assert processed == 2
