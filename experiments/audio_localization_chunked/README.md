# Chronological chunked-ASR experiment

This strategy keeps the production `base.en` faster-whisper recognizer, production text normalization/RapidFuzz matcher, media acquisition, ffprobe cache, and PTS-based frame resolution. It changes only experimental orchestration: FFmpeg extracts one reusable mono 16 kHz PCM WAV, chronological overlapping slices are read from that open source, and one loaded faster-whisper model transcribes them until the earliest accepted occurrence is available.

Production V4 does not import this package and is unchanged. The strategy adds no VAD, KWS, lightweight localizer, speech model, or dependency.

## Configuration

URLs, targets, model settings, and reference timestamps remain in JSON. Chunk values live under `chunked_asr`:

```json
{
  "chunked_asr": {
    "chunk_duration_seconds": 120,
    "overlap_seconds": 5,
    "transcript_context_seconds": 15
  }
}
```

`overlap_seconds` must be non-negative and smaller than the chunk duration. `transcript_context_seconds` controls the bounded previous-transcript tail used to recover phrases that cross a chunk boundary. The CLI can override all three values for controlled sweeps.

The default 120/5 configuration is the best measured CPU setting on the 90-minute Torovian case. Keep these values configurable: target position, hardware, and model can change the optimum.

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
  --overlap 3 `
  --transcript-context 15
```

## Correctness and metrics

Word timestamps from each chunk are shifted by the chunk start and the media audio-stream start offset. The current chunk is searched immediately. A bounded canonical tail is also joined to it at the overlap midpoint, choosing one ASR observation on either side of that time seam rather than interleaving contradictory overlap transcripts. This recovers boundary-spanning targets, keeps matching work bounded, and preserves chronological first-occurrence semantics.

The structured JSON includes total/ASR wall time, actual WAV seconds sent to ASR, unique chronological coverage, overlap seconds/percentage, one-time audio extraction calls/time, chunks processed/total, early-stop chunk and coverage timestamp, detected timestamp and baseline delta, raw match text/type/score, first-occurrence agreement, percentage ASR audio avoided, total/ASR speedup ratios, peak process-tree RSS, cache provenance, fallback information, and error reason. Numeric seconds are authoritative and parallel `*_hms` fields are for display.

First-occurrence agreement is based on successful chronological verification and the configured baseline timestamp tolerance. Transcript wording and raw match scores remain diagnostics and do not independently change occurrence identity. `baseline_configuration_parity` records whether media, input, ASR settings, matcher/threshold, and runtime environment are identical.

Generated JSON and frames under `results/` are ignored by Git.
