# Lightweight Whisper locator experiment

This experiment scans the complete audio with a configurable small faster-whisper model,
turns locator matches into chronological candidate windows, and accepts a result only after
the unchanged accurate baseline model verifies it. Candidate timestamps are expanded and
clamped in audio-relative time, then restored to absolute media timestamps after verification.

If the locator fails, finds no candidates, or no candidate verifies, the accurate model runs
on the full audio as the correctness fallback. Production V4 is not imported as a new code
path or modified.

The example uses `tiny.en` as the locator and `base.en` as the accurate model. Both use the
same device, compute type, language, matcher, and fuzzy threshold from the benchmark case.

Run after producing the three existing strategy result files:

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_lightweight_locator `
  --manifest experiments/audio_localization_lightweight_locator/manifest.example.json `
  --baseline-results experiments/audio_localization_chunked_vad/results/full-asr-baseline.json `
  --chunked-results experiments/audio_localization_chunked_vad/results/chunked-asr.json `
  --vad-results experiments/audio_localization_chunked_vad/results/chunked-vad-asr.json `
  --output experiments/audio_localization_lightweight_locator/results/lightweight-locator.json
```

The first run downloads only the configured locator model if it is absent. Temporary full and
candidate WAV files are deleted after every case; acquired media and model files use the
existing caches.
