# Full-ASR baseline benchmark

This experiment measures the unchanged production `dialogue_locator.pipeline.run_v1` baseline. Production modules do not import this package, and the harness does not implement VAD, chunking, lightweight ASR, keyword spotting, or any alternate localization behavior.

## Run

From the repository root with the project virtual environment active:

```powershell
python -m experiments.audio_localization_baseline `
  --manifest experiments/audio_localization_baseline/manifest.example.json `
  --output experiments/audio_localization_baseline/results/full-asr-baseline.json
```

The manifest owns all URLs, targets, model settings, paths, language hints, and expected production baselines. Add cases to the JSON rather than editing Python.

By default, `reuse_transcript_cache` is `false`: downloaded media, ffprobe metadata, and model files are still reused, but transcript-result caching is bypassed inside the experiment so ASR time is measured instead of hidden by a prior result. Set it to `true` to benchmark warm transcript-cache behavior; a cache hit then reports zero ASR calls and zero expensive ASR audio seconds.

## Output

The JSON report contains environment metadata and one structured record per case:

- `total_wall_clock_seconds`: complete production `run_v1` call, including acquisition/cache lookup, probing, audio extraction, ASR, matching, and frame extraction.
- `asr_wall_clock_seconds`: time spent inside the actual faster-whisper transcriber. Cache hashing and other pipeline work are excluded.
- `media_duration_seconds`: ffprobe container duration used by production.
- `audio_duration_seconds`: duration of the generated mono 16 kHz baseline WAV.
- `expensive_asr_audio_seconds_processed`: sum of WAV seconds supplied to actual faster-whisper calls. This is zero for a transcript-cache hit.
- `detected_timestamp_seconds`, `matched_text`, `match_type`, and `match_score`: unchanged production result.
- `first_occurrence_matches_production_baseline`: timestamp tolerance plus normalized detected-text comparison against the optional manifest reference; `null` if no reference is supplied.
- `peak_rss_bytes`: best-effort peak RSS of the benchmark process and child processes, sampled through optional `psutil`; `null` if unavailable.
- `fallback_used` and `fallback_reason`: currently records direct-HTTP acquisition or precision fallback. Full ASR itself is the selected baseline, not labeled as a fallback.
- `error_reason`: exception type and message while preserving later cases in the report.
- cache provenance and the effective model/device/language configuration.

Result files are ignored by Git under `results/`; only the manifest and harness belong on the experiment branch.
