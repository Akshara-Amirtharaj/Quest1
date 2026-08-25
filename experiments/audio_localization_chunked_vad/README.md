# Conservative VAD + chunked-ASR experiment

This third strategy reuses the existing chronological 8-second/2-second-overlap chunk search, production `base.en` faster-whisper model, matcher, thresholds, acquisition, and PTS frame resolution. Inside each chunk it uses faster-whisper's bundled Silero VAD ONNX model to remove clear non-speech before ASR. Production V4 is unchanged.

No VAD package, speech model, lightweight localizer, KWS system, or production integration is added.

## Conservative defaults

- Speech threshold: `0.35`
- Negative threshold: `0.20`
- Minimum speech: `0 ms`
- Minimum silence: `1000 ms`
- Speech padding: `500 ms` on both sides
- Clear-silence RMS threshold: `0.002`
- Per-chunk fallback when nontrivial audio loses more than `98%`
- Full unfiltered chunked fallback on no match or baseline disagreement

All parameters are manifest-controlled. Low threshold, zero minimum speech, long silence requirement, and generous padding bias toward recall.

Word times from the compressed speech waveform are restored through faster-whisper's `SpeechTimestampsMap`, then the existing chunk-start/audio-stream offsets are applied. Chunk overlap and chronological early stopping are unchanged.

## Three-way benchmark

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_baseline `
  --manifest experiments/audio_localization_chunked_vad/manifest.example.json `
  --output experiments/audio_localization_chunked_vad/results/full-asr-baseline.json

.venv\Scripts\python.exe -m experiments.audio_localization_chunked `
  --manifest experiments/audio_localization_chunked_vad/manifest.example.json `
  --baseline-results experiments/audio_localization_chunked_vad/results/full-asr-baseline.json `
  --output experiments/audio_localization_chunked_vad/results/chunked-asr.json

.venv\Scripts\python.exe -m experiments.audio_localization_chunked_vad `
  --manifest experiments/audio_localization_chunked_vad/manifest.example.json `
  --baseline-results experiments/audio_localization_chunked_vad/results/full-asr-baseline.json `
  --chunked-results experiments/audio_localization_chunked_vad/results/chunked-asr.json `
  --output experiments/audio_localization_chunked_vad/results/chunked-vad-asr.json
```

The VAD report records full original duration, VAD-examined duration, retained speech, removal percentage, VAD and ASR wall time, expensive ASR seconds including fallback work, chunk counts, timestamp correctness, fallback provenance, speedups against both references, and peak RSS.

Generated JSON and frames under `results/` are ignored by Git.
