from __future__ import annotations

import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from rapidfuzz import fuzz

from .matching import normalize_text
from .models import CandidateVideoFrame, OCRLine, OCRTextMatch


PADDLE_DETECTION_MODEL = "PP-OCRv5_mobile_det"
PADDLE_RECOGNITION_MODEL = "en_PP-OCRv5_mobile_rec"
PADDLE_MODEL_DESCRIPTION = f"{PADDLE_DETECTION_MODEL} + {PADDLE_RECOGNITION_MODEL}"


class OCRReader(Protocol):
    model_description: str

    def __call__(self, image: object) -> list[OCRLine]: ...


class PaddleOCRReader:
    model_description = PADDLE_MODEL_DESCRIPTION

    def __init__(self, cache_dir: Path) -> None:
        self._engine = None
        self._cache_dir = cache_dir

    def _get_engine(self):
        if self._engine is None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self._cache_dir.resolve()))
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed. Install project dependencies with "
                    "'python -m pip install -e .'."
                ) from exc
            self._engine = PaddleOCR(
                text_detection_model_name=PADDLE_DETECTION_MODEL,
                text_recognition_model_name=PADDLE_RECOGNITION_MODEL,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        return self._engine

    def __call__(self, image: object) -> list[OCRLine]:
        import numpy as np

        results = self._get_engine().predict(input=np.asarray(image))
        lines: list[OCRLine] = []
        for result in results:
            payload = getattr(result, "json", result)
            if callable(payload):
                payload = payload()
            if not isinstance(payload, dict):
                continue
            payload = payload.get("res", payload)
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            for index, text in enumerate(texts):
                if not str(text).strip():
                    continue
                score = float(scores[index]) if index < len(scores) else None
                lines.append(OCRLine(str(text), score))
        return lines


def _score(query: str, candidate: str) -> float:
    return max(
        float(fuzz.ratio(query, candidate)),
        float(fuzz.ratio(query.replace(" ", ""), candidate.replace(" ", ""))),
    )


def match_visible_text(
    query: str,
    lines: list[OCRLine],
    fuzzy_threshold: float = 85.0,
) -> OCRTextMatch | None:
    normalized_query = normalize_text(query)
    if not normalized_query:
        raise ValueError("Target dialogue must contain at least one letter or number.")
    visible_text = " ".join(line.text.strip() for line in lines if line.text.strip())
    normalized_visible = normalize_text(visible_text)
    if not normalized_visible:
        return None

    query_tokens = normalized_query.split()
    visible_tokens = normalized_visible.split()
    query_length = len(query_tokens)
    compact_query = "".join(query_tokens)
    for start in range(len(visible_tokens)):
        candidate_tokens = visible_tokens[start : start + query_length]
        if candidate_tokens == query_tokens or "".join(candidate_tokens) == compact_query:
            return OCRTextMatch(" ".join(candidate_tokens), "exact", 100.0)

    length_delta = max(1, min(3, math.ceil(query_length * 0.25)))
    minimum_length = max(1, query_length - length_delta)
    maximum_length = query_length + length_delta
    best: tuple[float, str] | None = None
    for start in range(len(visible_tokens)):
        for length in range(minimum_length, maximum_length + 1):
            candidate_tokens = visible_tokens[start : start + length]
            if len(candidate_tokens) != length:
                break
            candidate = " ".join(candidate_tokens)
            score = _score(normalized_query, candidate)
            if score >= fuzzy_threshold and (best is None or score > best[0]):
                best = score, candidate
    if best is None:
        return None
    return OCRTextMatch(best[1], "fuzzy", best[0])


def find_first_visible_frame(
    query: str,
    frames: Iterable[CandidateVideoFrame],
    reader: OCRReader,
    fuzzy_threshold: float = 85.0,
) -> tuple[CandidateVideoFrame | None, OCRTextMatch | None, int]:
    processed = 0
    for frame in frames:
        processed += 1
        match = match_visible_text(query, reader(frame.image), fuzzy_threshold)
        if match is not None:
            return frame, match, processed
    return None, None, processed
