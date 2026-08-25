from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import V2Config, V3Config
from .errors import V0Error
from .matching import DEFAULT_FUZZY_THRESHOLD
from .pipeline import run_v0, run_v1, run_v2, run_v3, run_v4
from .precision import DEFAULT_PRECISION_MODEL, DEFAULT_PRECISION_TRIGGER_THRESHOLD
from .transcription import DEFAULT_MODEL


def build_parser(*, milestone: str = "V4") -> argparse.ArgumentParser:
    description = (
        "Quest1 V4: hardened caption/ASR localization with optional visible-text verification."
        if milestone == "V4"
        else "Quest1 V3: candidate-window visible-text verification with spoken fallback."
    )
    parser = argparse.ArgumentParser(description=description)
    _add_v2_arguments(parser)
    parser.add_argument("--ocr-search-margin", type=float, default=1.0)
    parser.add_argument("--ocr-fuzzy-threshold", type=float, default=85.0)
    return parser


def _add_v2_arguments(parser: argparse.ArgumentParser) -> None:
    _add_v1_arguments(parser)
    parser.add_argument("--language", help="BCP-47-style target language hint, such as en or en-US")
    parser.add_argument("--caption-fuzzy-threshold", type=float, default=85.0)
    parser.add_argument("--subtitle-window-size", type=int, default=4)
    parser.add_argument("--verification-margins", type=_parse_margins, default=(2.0, 5.0))


def build_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quest1 V2: use captions to localize, then verify spoken dialogue with ASR."
    )
    _add_v2_arguments(parser)
    return parser


def _add_v1_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("url", help="Public http(s) video URL")
    parser.add_argument("dialogue", help="Target spoken dialogue")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/media"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/models"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--precision-mode",
        choices=("default", "whisperx"),
        default="default",
        help="Keep faster-whisper timestamps or optionally refine them with WhisperX",
    )
    parser.add_argument(
        "--asr-precision-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Lazily retry rejected English candidate windows with the precision ASR model"
        ),
    )
    parser.add_argument(
        "--precision-asr-model",
        default=DEFAULT_PRECISION_MODEL,
        help="faster-whisper checkpoint used only after the default ASR rejects a match",
    )
    parser.add_argument(
        "--precision-trigger-threshold",
        type=float,
        default=DEFAULT_PRECISION_TRIGGER_THRESHOLD,
        help=(
            "Minimum rejected base-ASR match score that is eligible for precision fallback"
        ),
    )
    parser.add_argument(
        "--full-audio-precision-fallback",
        action="store_true",
        help=(
            "Explicitly allow the precision ASR model to retry full audio after base ASR fails"
        ),
    )
    parser.add_argument("--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD)
    parser.add_argument(
        "--cookies-from-browser",
        choices=("chrome", "edge", "firefox", "brave", "opera", "vivaldi", "safari"),
        help="Use cookies from a signed-in browser profile for providers that require authentication",
    )
    parser.add_argument("--cookies-file", type=Path, help="Netscape-format cookies file for yt-dlp")


def _parse_margins(value: str) -> tuple[float, ...]:
    try:
        margins = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("margins must be comma-separated numbers") from exc
    if not margins:
        raise argparse.ArgumentTypeError("at least one verification margin is required")
    return margins


def main(argv: list[str] | None = None) -> int:
    return _visual_main(run_v4, argv)


def v3_main(argv: list[str] | None = None) -> int:
    return _visual_main(run_v3, argv, milestone="V3")


def _visual_main(runner, argv: list[str] | None = None, *, milestone: str = "V4") -> int:
    args = build_parser(milestone=milestone).parse_args(argv)
    try:
        config = V2Config(
            caption_fuzzy_threshold=args.caption_fuzzy_threshold,
            verification_fuzzy_threshold=args.fuzzy_threshold,
            subtitle_window_size=args.subtitle_window_size,
            verification_margins=args.verification_margins,
            asr_precision_fallback=args.asr_precision_fallback,
            precision_asr_model=args.precision_asr_model,
            precision_trigger_threshold=args.precision_trigger_threshold,
            full_audio_precision_fallback=args.full_audio_precision_fallback,
        )
        result = runner(
            args.url,
            args.dialogue,
            args.work_dir,
            args.output_dir,
            args.model_cache,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            v2_config=config,
            v3_config=V3Config(
                search_margin=args.ocr_search_margin,
                fuzzy_threshold=args.ocr_fuzzy_threshold,
            ),
            cookies_from_browser=args.cookies_from_browser,
            cookie_file=args.cookies_file,
            precision_mode=args.precision_mode,
        )
    except (V0Error, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def v2_main(argv: list[str] | None = None) -> int:
    args = build_v2_parser().parse_args(argv)
    try:
        config = V2Config(
            caption_fuzzy_threshold=args.caption_fuzzy_threshold,
            verification_fuzzy_threshold=args.fuzzy_threshold,
            subtitle_window_size=args.subtitle_window_size,
            verification_margins=args.verification_margins,
            asr_precision_fallback=args.asr_precision_fallback,
            precision_asr_model=args.precision_asr_model,
            precision_trigger_threshold=args.precision_trigger_threshold,
            full_audio_precision_fallback=args.full_audio_precision_fallback,
        )
        result = run_v2(
            args.url,
            args.dialogue,
            args.work_dir,
            args.output_dir,
            args.model_cache,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            config=config,
            cookies_from_browser=args.cookies_from_browser,
            cookie_file=args.cookies_file,
            precision_mode=args.precision_mode,
        )
    except (V0Error, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def build_v1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quest1 V1: full-audio ASR dialogue localization."
    )
    _add_v1_arguments(parser)
    parser.add_argument("--language", help="ASR language hint for multilingual models")
    return parser


def v1_main(argv: list[str] | None = None) -> int:
    args = build_v1_parser().parse_args(argv)
    try:
        result = run_v1(
            args.url,
            args.dialogue,
            args.work_dir,
            args.output_dir,
            args.model_cache,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            fuzzy_threshold=args.fuzzy_threshold,
            language=args.language,
            cookies_from_browser=args.cookies_from_browser,
            cookie_file=args.cookies_file,
            precision_mode=args.precision_mode,
            asr_precision_fallback=args.asr_precision_fallback,
            precision_asr_model=args.precision_asr_model,
            precision_trigger_threshold=args.precision_trigger_threshold,
            full_audio_precision_fallback=args.full_audio_precision_fallback,
        )
    except (V0Error, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def build_v0_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quest1 V0: acquire and inspect media, then decode one frame."
    )
    parser.add_argument("url", help="Public http(s) video URL")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/media"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--cookies-from-browser")
    parser.add_argument("--cookies-file", type=Path)
    return parser


def v0_main(argv: list[str] | None = None) -> int:
    args = build_v0_parser().parse_args(argv)
    try:
        result = run_v0(
            args.url,
            args.work_dir,
            args.output_dir,
            cookies_from_browser=args.cookies_from_browser,
            cookie_file=args.cookies_file,
        )
    except V0Error as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0
