from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dialogue_locator.models import DialogueMatch
from experiments.audio_localization_lightweight_locator.localization import CandidateWindow


@dataclass(frozen=True)
class AnchorDetection:
    anchor: str
    start: float
    end: float


@dataclass(frozen=True)
class KWSCandidateRegion:
    index: int
    start: float
    end: float
    detections: tuple[AnchorDetection, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def verification_window(self) -> CandidateWindow:
        text = " / ".join(dict.fromkeys(item.anchor for item in self.detections))
        raw_start = min(item.start for item in self.detections)
        raw_end = max(item.end for item in self.detections)
        return CandidateWindow(
            index=self.index,
            start=self.start,
            end=self.end,
            locator_match=DialogueMatch(text, raw_start, raw_end, "kws", 100.0),
        )


def group_detections(
    detections: Iterable[AnchorDetection],
    *,
    audio_duration: float,
    grouping_gap: float,
    margin_before: float,
    margin_after: float,
) -> tuple[KWSCandidateRegion, ...]:
    if audio_duration <= 0:
        raise ValueError("audio_duration must be greater than zero.")
    if min(grouping_gap, margin_before, margin_after) < 0:
        raise ValueError("grouping gap and candidate margins cannot be negative.")
    ordered = sorted(detections, key=lambda item: (item.start, item.end, item.anchor))
    if not ordered:
        return ()
    groups: list[list[AnchorDetection]] = [[ordered[0]]]
    group_end = ordered[0].end
    for detection in ordered[1:]:
        if detection.start - group_end <= grouping_gap:
            groups[-1].append(detection)
            group_end = max(group_end, detection.end)
        else:
            groups.append([detection])
            group_end = detection.end
    return tuple(
        KWSCandidateRegion(
            index=index,
            start=max(0.0, min(item.start for item in group) - margin_before),
            end=min(audio_duration, max(item.end for item in group) + margin_after),
            detections=tuple(group),
        )
        for index, group in enumerate(groups)
    )
