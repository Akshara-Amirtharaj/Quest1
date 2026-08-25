from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from dialogue_locator.dependencies import require_external_tools

from .manifest import (
    FROZEN_CHUNK_DURATION_SECONDS,
    FROZEN_OVERLAP_SECONDS,
    FROZEN_TRANSCRIPT_CONTEXT_SECONDS,
)


TARGET = "The company reported revenue of twenty million dollars."
FIXTURE_DIR = Path(".cache/benchmark-fixtures")


@dataclass(frozen=True)
class FixtureMetadata:
    fixture_id: str
    path: str
    duration_seconds: float
    target: str
    target_onsets_seconds: tuple[float, ...]
    target_speech_duration_seconds: float | None
    unrelated_speech_onsets_seconds: tuple[float, ...]
    has_captions: bool
    generator: str
    frozen_chunk_duration_seconds: float
    frozen_overlap_seconds: float
    frozen_transcript_context_seconds: float
    background: str


def derive_boundary_onset(
    chunk_duration_seconds: float = FROZEN_CHUNK_DURATION_SECONDS,
) -> float:
    return chunk_duration_seconds - 1.5


def generate_controlled_fixtures(output_dir: Path = FIXTURE_DIR) -> dict[str, FixtureMetadata]:
    tools = require_external_tools()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_wav, voice = _synthesize(TARGET, output_dir / "target-speech.wav")
    unrelated_wav, _ = _synthesize(
        "Welcome to this quiet benchmark recording.",
        output_dir / "unrelated-speech.wav",
    )
    target_duration = _wav_duration(target_wav)
    boundary_onset = derive_boundary_onset()
    if not (boundary_onset < FROZEN_CHUNK_DURATION_SECONDS < boundary_onset + target_duration):
        raise RuntimeError(
            "Synthesized target does not cross the frozen 120-second boundary; "
            f"onset={boundary_onset}, duration={target_duration}."
        )

    specs = {
        "fixture_no_captions_absent_long_silence": {
            "duration": 75.0,
            "speech": [(unrelated_wav, 35.0)],
            "target_onsets": (),
            "unrelated_onsets": (35.0,),
            "background": "deterministic silence with one unrelated spoken section",
        },
        "fixture_noisy_speech": {
            "duration": 45.0,
            "speech": [(target_wav, 20.0)],
            "target_onsets": (20.0,),
            "unrelated_onsets": (),
            "background": "deterministic 220/330 Hz low-level tone mixture",
        },
        "fixture_chunk_boundary": {
            "duration": 150.0,
            "speech": [(target_wav, boundary_onset)],
            "target_onsets": (boundary_onset,),
            "unrelated_onsets": (),
            "background": "silence",
        },
        "fixture_repeated_target": {
            "duration": 60.0,
            "speech": [(target_wav, 10.0), (target_wav, 45.0)],
            "target_onsets": (10.0, 45.0),
            "unrelated_onsets": (),
            "background": "silence",
        },
    }
    generated: dict[str, FixtureMetadata] = {}
    for fixture_id, spec in specs.items():
        path = output_dir / f"{fixture_id}.mp4"
        _render_fixture(
            ffmpeg=tools.ffmpeg,
            output_path=path,
            duration=float(spec["duration"]),
            speech=list(spec["speech"]),
            noisy=fixture_id == "fixture_noisy_speech",
        )
        metadata = FixtureMetadata(
            fixture_id=fixture_id,
            path=str(path.resolve()),
            duration_seconds=float(spec["duration"]),
            target=TARGET,
            target_onsets_seconds=tuple(spec["target_onsets"]),
            target_speech_duration_seconds=(
                target_duration if spec["target_onsets"] else None
            ),
            unrelated_speech_onsets_seconds=tuple(spec["unrelated_onsets"]),
            has_captions=False,
            generator=f"Windows SAPI ({voice}) + FFmpeg",
            frozen_chunk_duration_seconds=FROZEN_CHUNK_DURATION_SECONDS,
            frozen_overlap_seconds=FROZEN_OVERLAP_SECONDS,
            frozen_transcript_context_seconds=FROZEN_TRANSCRIPT_CONTEXT_SECONDS,
            background=str(spec["background"]),
        )
        generated[fixture_id] = metadata
    metadata_path = output_dir / "fixtures-metadata.json"
    metadata_path.write_text(
        json.dumps({key: asdict(value) for key, value in generated.items()}, indent=2),
        encoding="utf-8",
    )
    return generated


def _synthesize(text: str, output_path: Path) -> tuple[Path, str]:
    output_path.unlink(missing_ok=True)
    escaped_path = str(output_path.resolve()).replace("'", "''")
    escaped_text = text.replace("'", "''")
    script = (
        "$voice = New-Object -ComObject SAPI.SpVoice; "
        "$voice.Rate = 0; $voice.Volume = 100; "
        "$stream = New-Object -ComObject SAPI.SpFileStream; "
        f"$stream.Open('{escaped_path}', 3, $false); "
        "$voice.AudioOutputStream = $stream; "
        f"[void]$voice.Speak('{escaped_text}'); "
        "$description = $voice.Voice.GetDescription(); "
        "$stream.Close(); Write-Output $description"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 46:
        raise RuntimeError(f"Windows SAPI fixture synthesis failed: {completed.stderr.strip()}")
    return output_path.resolve(), completed.stdout.strip() or "default voice"


def _render_fixture(
    *,
    ffmpeg: str,
    output_path: Path,
    duration: float,
    speech: list[tuple[Path, float]],
    noisy: bool,
) -> None:
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x18202a:s=320x180:r=2:d={duration:g}",
        "-f",
        "lavfi",
        "-i",
        (
            f"aevalsrc=0.035*sin(2*PI*220*t)+0.025*sin(2*PI*330*t):s=16000:d={duration:g}"
            if noisy
            else f"anullsrc=r=16000:cl=mono:d={duration:g}"
        ),
    ]
    unique_inputs: list[Path] = []
    for path, _ in speech:
        if path not in unique_inputs:
            unique_inputs.append(path)
            command.extend(["-i", str(path)])

    filters: list[str] = []
    delayed_labels: list[str] = []
    occurrence_counts: dict[Path, int] = {path: 0 for path in unique_inputs}
    totals: dict[Path, int] = {
        path: sum(candidate == path for candidate, _ in speech) for path in unique_inputs
    }
    source_labels: dict[tuple[Path, int], str] = {}
    for input_index, path in enumerate(unique_inputs, start=2):
        count = totals[path]
        if count == 1:
            source_labels[(path, 0)] = f"{input_index}:a"
        else:
            labels = [f"split_{input_index}_{index}" for index in range(count)]
            filters.append(f"[{input_index}:a]asplit={count}" + "".join(f"[{label}]" for label in labels))
            for index, label in enumerate(labels):
                source_labels[(path, index)] = label
    for occurrence_index, (path, onset) in enumerate(speech):
        local_index = occurrence_counts[path]
        occurrence_counts[path] += 1
        source = source_labels[(path, local_index)]
        label = f"delayed_{occurrence_index}"
        delay_ms = round(onset * 1000)
        filters.append(f"[{source}]adelay={delay_ms}:all=1[{label}]")
        delayed_labels.append(label)
    mix_inputs = "[1:a]" + "".join(f"[{label}]" for label in delayed_labels)
    filters.append(
        f"{mix_inputs}amix=inputs={1 + len(delayed_labels)}:duration=first:normalize=0[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "mpeg4",
            "-q:v",
            "12",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-t",
            f"{duration:g}",
            "-metadata",
            "comment=Generated deterministic Quest1 frozen benchmark fixture",
            str(output_path),
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"FFmpeg fixture generation failed: {completed.stderr.strip()}")


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()
