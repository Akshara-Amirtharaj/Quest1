from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from experiments.audio_localization_chunked.metrics import speedup_ratio

from .manifest import FrozenMatrixManifest, MatrixCase


def evaluate_strategy_record(
    case: MatrixCase,
    strategy: str,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    detected = _number(record, "detected_timestamp_seconds")
    error = record.get("error_reason") if record else "missing result record"
    no_match = detected is None and _is_no_match_error(error)
    matched = detected is not None
    delta = (
        abs(detected - case.expected_timestamp_seconds)
        if detected is not None and case.expected_timestamp_seconds is not None
        else None
    )
    within_tolerance = (
        delta <= case.tolerance_seconds
        if delta is not None and case.tolerance_seconds is not None
        else None
    )
    if case.expected_result == "NO_MATCH":
        correct = no_match
        false_positive = matched
        false_negative = False
    else:
        correct = matched and within_tolerance is True
        false_positive = False
        false_negative = not matched
    return {
        "case_id": case.case_id,
        "strategy": strategy,
        "media_identity": case.media_identity,
        "media_duration_seconds": _number(record, "media_duration_seconds"),
        "target": case.target,
        "expected_result": case.expected_result,
        "expected_first_timestamp_seconds": case.expected_timestamp_seconds,
        "ground_truth_tolerance_seconds": case.tolerance_seconds,
        "detected_first_timestamp_seconds": detected,
        "absolute_timestamp_delta_seconds": delta,
        "within_tolerance": within_tolerance,
        "correct_first_occurrence": correct,
        "result": "MATCH" if matched else ("NO_MATCH" if no_match else "ERROR"),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "matched_text": record.get("matched_text") if record else None,
        "match_type": record.get("match_type") if record else None,
        "match_score": _number(record, "match_score"),
        "total_wall_clock_seconds": _number(record, "total_wall_clock_seconds"),
        "asr_wall_clock_seconds": _number(record, "asr_wall_clock_seconds"),
        "expensive_asr_audio_seconds": _number(
            record, "expensive_asr_audio_seconds_processed"
        ),
        "expensive_asr_audio_percentage": _percentage(
            _number(record, "expensive_asr_audio_seconds_processed"),
            _number(record, "media_duration_seconds"),
        ),
        "chunks_processed": _number(record, "chunks_processed"),
        "chunks_total": _number(record, "chunks_total"),
        "early_stop_triggered": record.get("early_stop_triggered") if record else None,
        "early_stop_chunk_index": record.get("early_stop_chunk_index") if record else None,
        "stop_timestamp_seconds": (
            record.get("early_stop_coverage_timestamp_seconds") if record else None
        ),
        "fallback_used": record.get("fallback_used") if record else None,
        "peak_rss_bytes": record.get("peak_rss_bytes") if record else None,
        "failure_reason": error,
        "run_number": 1,
    }


def build_comparison(
    manifest: FrozenMatrixManifest,
    baseline_report: dict[str, Any],
    chunked_report: dict[str, Any],
) -> dict[str, Any]:
    baseline_index = _index(baseline_report)
    chunked_index = _index(chunked_report)
    strategy_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for case in manifest.audio_cases:
        baseline = evaluate_strategy_record(case, "full_asr", baseline_index.get(case.case_id))
        chunked = evaluate_strategy_record(case, "chunked_asr", chunked_index.get(case.case_id))
        strategy_rows.extend((baseline, chunked))
        baseline_wall = baseline["total_wall_clock_seconds"]
        chunked_wall = chunked["total_wall_clock_seconds"]
        baseline_audio = baseline["expensive_asr_audio_seconds"]
        chunked_audio = chunked["expensive_asr_audio_seconds"]
        comparisons.append(
            {
                "case_id": case.case_id,
                "tags": list(case.tags),
                "expected_result": case.expected_result,
                "expected_first_timestamp_seconds": case.expected_timestamp_seconds,
                "baseline_result": baseline["result"],
                "chunked_result": chunked["result"],
                "baseline_correct": baseline["correct_first_occurrence"],
                "chunked_correct": chunked["correct_first_occurrence"],
                "baseline_detected_timestamp_seconds": baseline[
                    "detected_first_timestamp_seconds"
                ],
                "chunked_detected_timestamp_seconds": chunked[
                    "detected_first_timestamp_seconds"
                ],
                "baseline_wall_clock_seconds": baseline_wall,
                "chunked_wall_clock_seconds": chunked_wall,
                "speedup_ratio": speedup_ratio(baseline_wall, chunked_wall),
                "wall_time_reduction_percent": _reduction(baseline_wall, chunked_wall),
                "baseline_expensive_asr_seconds": baseline_audio,
                "chunked_expensive_asr_seconds": chunked_audio,
                "expensive_asr_reduction_percent": _reduction(
                    baseline_audio, chunked_audio
                ),
                "baseline_match_score": baseline["match_score"],
                "chunked_match_score": chunked["match_score"],
                "baseline_failure_reason": baseline["failure_reason"],
                "chunked_failure_reason": chunked["failure_reason"],
            }
        )
    speedups = [row["speedup_ratio"] for row in comparisons if row["speedup_ratio"] is not None]
    wall_reductions = [
        row["wall_time_reduction_percent"]
        for row in comparisons
        if row["wall_time_reduction_percent"] is not None
    ]
    asr_reductions = [
        row["expensive_asr_reduction_percent"]
        for row in comparisons
        if row["expensive_asr_reduction_percent"] is not None
    ]
    baseline_rows = [row for row in strategy_rows if row["strategy"] == "full_asr"]
    chunked_rows = [row for row in strategy_rows if row["strategy"] == "chunked_asr"]
    return {
        "benchmark_id": manifest.benchmark_id,
        "inventory": {
            "total_benchmark_cases": len(manifest.cases),
            "audio_comparison_cases": len(manifest.audio_cases),
            "ocr_full_pipeline_only_cases": sum(not case.is_audio_case for case in manifest.cases),
            "public_benchmark_cases": sum(case.source_kind == "public" for case in manifest.cases),
            "controlled_fixture_cases": sum(
                case.source_kind == "controlled" for case in manifest.cases
            ),
            "unique_media_sources": manifest.unique_media_count,
        },
        "effective_configuration": {
            "model": manifest.defaults.model,
            "device": manifest.defaults.device,
            "compute_type": manifest.defaults.compute_type,
            "language": manifest.defaults.language,
            "fuzzy_threshold": manifest.defaults.fuzzy_threshold,
            "chunked_asr": {
                "chunk_duration_seconds": manifest.chunked_asr.chunk_duration_seconds,
                "overlap_seconds": manifest.chunked_asr.overlap_seconds,
                "transcript_context_seconds": manifest.chunked_asr.transcript_context_seconds,
            },
        },
        "strategy_results": strategy_rows,
        "comparisons": comparisons,
        "aggregate": {
            "audio_cases_attempted": len(manifest.audio_cases),
            "baseline_correct_cases": sum(row["correct_first_occurrence"] for row in baseline_rows),
            "chunked_correct_cases": sum(row["correct_first_occurrence"] for row in chunked_rows),
            "baseline_false_positives": sum(row["false_positive"] for row in baseline_rows),
            "chunked_false_positives": sum(row["false_positive"] for row in chunked_rows),
            "baseline_false_negatives": sum(row["false_negative"] for row in baseline_rows),
            "chunked_false_negatives": sum(row["false_negative"] for row in chunked_rows),
            "median_speedup_ratio": _median(speedups),
            "median_wall_time_reduction_percent": _median(wall_reductions),
            "median_expensive_asr_reduction_percent": _median(asr_reductions),
            "total_baseline_expensive_asr_seconds": _sum_numbers(
                row["expensive_asr_audio_seconds"] for row in baseline_rows
            ),
            "total_chunked_expensive_asr_seconds": _sum_numbers(
                row["expensive_asr_audio_seconds"] for row in chunked_rows
            ),
        },
        "excluded_from_audio_aggregates": [
            case.case_id for case in manifest.cases if not case.is_audio_case
        ],
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    comparisons = summary["comparisons"]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparisons[0]) if comparisons else [])
        writer.writeheader()
        writer.writerows(comparisons)
    (output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Frozen benchmark: {summary['benchmark_id']}",
        "",
        "| Case | Expected | Baseline | Chunked | Baseline time | Chunked time | Speedup | Baseline ASR | Chunked ASR | ASR reduction | Correct? |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            "| {case_id} | {expected_result} | {baseline_result} | {chunked_result} | "
            "{baseline_wall_clock_seconds} | {chunked_wall_clock_seconds} | {speedup_ratio} | "
            "{baseline_expensive_asr_seconds} | {chunked_expensive_asr_seconds} | "
            "{expensive_asr_reduction_percent} | {baseline_correct}/{chunked_correct} |".format(
                **{key: _display(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(summary["aggregate"], indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["case_id"]: record for record in report.get("cases", [])}


def _number(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None or isinstance(record.get(key), bool):
        return None
    try:
        return float(record[key])
    except (KeyError, TypeError, ValueError):
        return None


def _is_no_match_error(error: Any) -> bool:
    text = str(error or "").casefold()
    return "dialogue not found" in text or "no timestamped words" in text


def _percentage(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator * 100.0


def _reduction(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _sum_numbers(values: Any) -> float:
    return sum(value for value in values if value is not None)


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "—" if value is None else str(value)
