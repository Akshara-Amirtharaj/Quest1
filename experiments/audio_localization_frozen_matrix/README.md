# Frozen audio-localization benchmark matrix

This experiment compares only the existing full-ASR baseline with chronological chunked ASR frozen at 120-second chunks, 5-second overlap, and 15-second transcript context. Production V4 is not imported differently or modified. VAD, lightweight localization, KWS, and OCR are excluded from audio aggregates.

The tracked manifest contains 12 registered cases: 11 audio comparisons and one separately registered OCR/full-pipeline case. Four Tears of Steel targets reuse one cached source, and two CS50 targets reuse one cached source. Generated fixtures and benchmark results are ignored.

## Setup and validation

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix validate
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix generate-fixtures
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix prepare
.venv\Scripts\python.exe -m pytest tests/experiments/test_frozen_matrix.py tests/experiments/test_chunked_asr.py tests/experiments/test_reusable_chunk_audio.py -q --basetemp=.pytest_tmp_frozen_docs
```

`generate-fixtures` uses the installed Windows SAPI voice and FFmpeg. Exact insertion onsets, synthesized duration, voice, and frozen configuration are written to `.cache/benchmark-fixtures/fixtures-metadata.json`.

## Run commands

Run one case against baseline:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix run --case-id tears_beginning --strategy baseline --run-id tears-beginning-baseline
```

Run the same case against frozen chunking, using that baseline:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix run --case-id tears_beginning --strategy chunked --baseline-results experiments/audio_localization_frozen_matrix/results/tears-beginning-baseline/baseline.json --run-id tears-beginning-chunked
```

Run one case against both strategies:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix run --case-id tears_beginning --strategy both --run-id tears-beginning-both
```

Run all public audio cases (the OCR-only case is filtered automatically from strategy execution):

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix run --source-kind public --strategy both --run-id public-audio-frozen
```

Run all controlled audio fixtures:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix run --source-kind controlled --strategy both --run-id controlled-audio-frozen
```

Run the complete frozen audio matrix:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix run --strategy both --run-id frozen-matrix-v1
```

Register/validate the OCR-only Big Buck Bunny case without including it in audio execution:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_frozen_matrix validate --case-id big_buck_bunny_visible_title
```

## Results

Each complete run directory contains `baseline.json`, `chunked.json`, `comparison.json`, `comparison.csv`, and `summary.md`. Media preparation status is written to `experiments/audio_localization_frozen_matrix/results/media-preparation.json`. Generated frames remain under the ignored results tree.

The benchmark uses one warm-up sanity pass and one measured run per case/strategy. Three measured repetitions of the multi-hour CPU-only matrix are intentionally not performed; this policy is recorded in the summary.

To remove only generated controlled fixtures:

```powershell
Remove-Item -LiteralPath ".cache/benchmark-fixtures" -Recurse -Force
```

Public media uses the existing `.cache/media` acquisition cache. Inspect exact paths in `media-preparation.json` before removing individual cached files; do not delete the entire cache blindly.
