# Chronological chunked-ASR experiment

This strategy keeps the production `base.en` faster-whisper recognizer, production text normalization/RapidFuzz matcher, media acquisition, ffprobe cache, and PTS-based frame resolution. It changes only experimental orchestration: audio is extracted and transcribed in chronological overlapping windows, searched after each completed chunk, and stopped once the earliest accepted occurrence is available.

Production V4 does not import this package and is unchanged. The strategy adds no VAD, KWS, lightweight localizer, speech model, or dependency.

## Configuration

URLs, targets, model settings, and reference timestamps remain in JSON. Chunk values live under `chunked_asr`:

```json
{
  "chunked_asr": {
    "chunk_duration_seconds": 8,
    "overlap_seconds": 2
  }
}
```

`overlap_seconds` must be non-negative and smaller than the chunk duration. The CLI can override either value for controlled sweeps.

## Run baseline and chunked strategy

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_baseline `
  --manifest experiments/audio_localization_chunked/manifest.example.json `
  --output experiments/audio_localization_chunked/results/full-asr-baseline.json

.venv\Scripts\python.exe -m experiments.audio_localization_chunked `
  --manifest experiments/audio_localization_chunked/manifest.example.json `
  --baseline-results experiments/audio_localization_chunked/results/full-asr-baseline.json `
  --output experiments/audio_localization_chunked/results/chunked-asr.json
```

Optional overrides:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_chunked `
  --manifest experiments/audio_localization_chunked/manifest.example.json `
  --baseline-results experiments/audio_localization_chunked/results/full-asr-baseline.json `
  --chunk-duration 12 `
  --overlap 3
```

## Correctness and metrics

Word timestamps from each chunk are shifted by the chunk start and the media audio-stream start offset. Overlapping observations of the same normalized word and spoken interval are deduplicated, then the accumulated chronological words are searched with the production matcher. Accumulation lets a target span two chunks; chronological processing and immediate matching preserve first-occurrence semantics.

The structured JSON includes total/ASR wall time, actual WAV seconds sent to ASR, chunks processed/total, early-stop chunk and coverage timestamp, detected timestamp and baseline delta, raw match text/type/score, first-occurrence agreement, percentage ASR audio avoided, total/ASR speedup ratios, peak process-tree RSS, cache provenance, fallback information, and error reason. Numeric seconds are authoritative and parallel `*_hms` fields are for display.

Generated JSON and frames under `results/` are ignored by Git.
