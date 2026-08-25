from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dialogue_locator.matching import normalize_text

from .candidates import AnchorDetection


class SherpaKWSUnavailable(RuntimeError):
    """The optional sherpa-onnx backend or its model is unavailable."""


@dataclass(frozen=True)
class SherpaKWSModel:
    model_dir: Path
    encoder_filename: str = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    decoder_filename: str = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    joiner_filename: str = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    tokens_filename: str = "tokens.txt"
    bpe_model_filename: str = "bpe.model"

    @property
    def description(self) -> str:
        return self.model_dir.name


@dataclass(frozen=True)
class SherpaKWSOptions:
    num_threads: int = 1
    keywords_score: float = 2.0
    keywords_threshold: float = 0.15
    num_trailing_blanks: int = 1


class SherpaKeywordSpotter:
    def __init__(self, model: SherpaKWSModel, options: SherpaKWSOptions) -> None:
        self.model = model
        self.options = options

    def detect(
        self,
        audio_path: Path,
        anchors: tuple[str, ...],
        temporary_dir: Path,
    ) -> tuple[AnchorDetection, ...]:
        if importlib.util.find_spec("sherpa_onnx") is None:
            raise SherpaKWSUnavailable(
                "sherpa-onnx is not installed; install the optional experiment runtime."
            )
        files = {
            "encoder": self.model.model_dir / self.model.encoder_filename,
            "decoder": self.model.model_dir / self.model.decoder_filename,
            "joiner": self.model.model_dir / self.model.joiner_filename,
            "tokens": self.model.model_dir / self.model.tokens_filename,
            "bpe_model": self.model.model_dir / self.model.bpe_model_filename,
        }
        for label, path in files.items():
            if not path.is_file():
                raise SherpaKWSUnavailable(f"Sherpa KWS {label} file is missing: {path}")
        if not audio_path.is_file():
            raise SherpaKWSUnavailable(f"KWS audio file is missing: {audio_path}")

        keyword_path = temporary_dir / "anchors-tokenized.txt"
        try:
            from sherpa_onnx.utils import text2token

            encoded = text2token(
                [anchor.upper() for anchor in anchors],
                tokens=str(files["tokens"]),
                tokens_type="bpe",
                bpe_model=str(files["bpe_model"]),
            )
            keyword_path.write_text(
                "\n".join(" ".join(str(token) for token in item) for item in encoded) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(_script("sherpa-onnx-keyword-spotter.exe")),
                    f"--encoder={files['encoder']}",
                    f"--decoder={files['decoder']}",
                    f"--joiner={files['joiner']}",
                    f"--tokens={files['tokens']}",
                    f"--keywords-file={keyword_path}",
                    f"--num-threads={self.options.num_threads}",
                    f"--keywords-score={self.options.keywords_score}",
                    f"--keywords-threshold={self.options.keywords_threshold}",
                    f"--num-trailing-blanks={self.options.num_trailing_blanks}",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise SherpaKWSUnavailable(f"Sherpa KWS execution failed: {detail}")
            known = {normalize_text(anchor): anchor for anchor in anchors}
            return tuple(
                _parse_detections(completed.stdout + "\n" + completed.stderr, known)
            )
        except SherpaKWSUnavailable:
            raise
        except Exception as exc:
            raise SherpaKWSUnavailable(f"Sherpa KWS execution failed: {exc}") from exc


def _script(name: str) -> Path:
    path = Path(sys.executable).resolve().parent / name
    if not path.is_file():
        raise SherpaKWSUnavailable(f"Sherpa executable is missing: {path}")
    return path


def _parse_detections(text: str, known: dict[str, str]) -> list[AnchorDetection]:
    detections: list[AnchorDetection] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            keyword = str(payload["keyword"]).strip()
            timestamps = [float(value) for value in payload["timestamps"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not timestamps:
            continue
        start_time = float(payload.get("start_time", 0.0) or 0.0)
        offset = start_time if timestamps[0] < start_time else 0.0
        anchor = known.get(normalize_text(keyword), normalize_text(keyword))
        detections.append(
            AnchorDetection(anchor, timestamps[0] + offset, timestamps[-1] + offset)
        )
    return sorted(detections, key=lambda item: (item.start, item.end, item.anchor))
