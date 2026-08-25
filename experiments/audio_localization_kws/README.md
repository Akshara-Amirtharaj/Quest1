# Open-vocabulary KWS experiment

This isolated strategy generates up to three deterministic adjacent-word anchors, tokenizes
them for sherpa-onnx, runs the official English GigaSpeech 3.3M streaming KWS model, groups
nearby detections, and verifies candidate windows chronologically with the unchanged accurate
faster-whisper model. A KWS miss, unavailable runtime/model, failed candidate, or target absence
always invokes accurate full-ASR fallback.

On the tested Windows build, sherpa-onnx's bundled native keyword-spotter executable emitted
correct detections, while the equivalent 1.13.6 Python `KeywordSpotter` wrapper emitted none
even for the model's known-positive sample. The experiment therefore tokenizes anchors through
the Python package but invokes the bundled native spotter executable.

The optional experiment environment used:

```powershell
.venv\Scripts\python.exe -m pip install sherpa-onnx==1.13.6 sherpa-onnx-bin==1.13.6 sentencepiece pypinyin
```

The model is not downloaded automatically. Place the official
`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01` model under `.cache/models`, or change
`kws.model_dir` in the manifest. No sherpa package is added to production dependencies.

```powershell
.venv\Scripts\python.exe -m experiments.audio_localization_kws `
  --manifest experiments/audio_localization_kws/manifest.example.json `
  --baseline-results experiments/audio_localization_chunked_vad/results/full-asr-baseline.json `
  --chunked-results experiments/audio_localization_chunked_vad/results/chunked-asr.json `
  --vad-results experiments/audio_localization_chunked_vad/results/chunked-vad-asr.json `
  --locator-results experiments/audio_localization_lightweight_locator/results/lightweight-locator.json `
  --output experiments/audio_localization_kws/results/kws-accurate-asr.json
```
