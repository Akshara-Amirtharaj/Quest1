# Quest1 Dialogue Locator

Quest1 accepts a public video URL and target dialogue, locates the first spoken occurrence, resolves the corresponding frame from actual presentation timestamps, and can verify the earliest nearby frame where the complete dialogue is visibly burned into the video. V4 is the final hardening milestone: the V0–V3 architecture remains intact, with optional precision alignment, explicit confidence, file caching, multilingual model selection, and clearer provenance.

## Problem statement

Given a public video URL and a target line of dialogue, find the first occurrence of that dialogue in the spoken audio and return the corresponding video frame. The solution must remain useful when captions are missing, inaccurate, unavailable, or rate-limited, and frame selection must use real media timestamps rather than an assumed constant frame rate.

## Solution overview

Quest1 uses a progressive evidence pipeline: provider captions first localize inexpensive candidates, short-window speech recognition verifies them, and full-audio ASR is the fallback. PyAV resolves the earliest frame at or after the matched word timestamp using presentation timestamps. Bounded PaddleOCR can replace that spoken-dialogue frame with the earliest nearby frame containing the complete visible phrase. The structured result exposes raw evidence, confidence, provenance, processing cost, and cache reuse.

Detailed design decisions, algorithms, trade-offs, and edge cases are documented in [APPROACH.md](APPROACH.md). AI-assistance disclosure is in [AI_USAGE.md](AI_USAGE.md), with the recorded prompt history in [prompts.md](prompts.md).

## Final architecture and evidence flow

```text
public URL
  -> V0: yt-dlp acquisition (direct-media requests fallback) + ffprobe
  -> V2: manual captions, then automatic captions
       -> chronological caption candidates -> short-window faster-whisper verification
       -> full-audio faster-whisper fallback when captions are absent/unusable
  -> V1: normalized exact match, then RapidFuzz; retain all practical occurrences
       -> first occurrence by default
       -> optional WhisperX word-boundary alignment
  -> PyAV seek/decode using PTS * time_base (never timestamp * FPS)
  -> V3: lazy OCR only around the candidate interval
       -> first frame containing the complete normalized/fuzzy visible phrase
       -> spoken frame remains the fallback
  -> V4: confidence + provenance + safe file-cache reuse
```

The evidence hierarchy is manual captions → automatic captions → full ASR. Captions only localize a candidate: faster-whisper still verifies the spoken text. V3/V4 attempt visible-text verification only inside the bounded candidate interval and retain the spoken frame when no OCR match exists. WhisperX, when requested, aligns already-produced faster-whisper words; it does not replace the default recognizer.

Heavy components are lazy. Faster-whisper runs only when caption verification or ASR fallback needs it, PaddleOCR initializes only after spoken localization reaches the bounded V3/V4 inspection stage and cached OCR is insufficient, and WhisperX is imported only in precision mode.

## Milestones

- **V0:** provider-neutral public URL acquisition, direct-media HTTP fallback, ffprobe inspection, and PyAV sampling.
- **V1:** full-audio faster-whisper transcription, cross-segment exact/fuzzy dialogue matching, and PTS-based target-frame extraction.
- **V2:** manual/automatic caption acquisition, chronological candidates, short-window ASR verification, and full-ASR fallback.
- **V3:** bounded candidate-window PaddleOCR and earliest complete visible-text frame selection, with spoken-only fallback.
- **V4:** optional WhisperX alignment, rule-based confidence, file caching, multilingual configuration, multiple-candidate detection, clearer edge cases/provenance, regression tests, and final documentation.

No later milestone is defined by this project.

## Repository layout

```text
Quest1/
├── src/dialogue_locator/   # installable application package
├── tests/                  # offline unit and regression tests
├── output/.gitkeep         # generated frames are ignored
├── README.md               # problem, setup, usage, and limitations
├── APPROACH.md             # architecture and engineering decisions
├── AI_USAGE.md             # AI-assistance disclosure
├── prompts.md              # recorded AI prompt history
├── pyproject.toml          # package metadata and dependencies
├── requirements.txt        # pip-compatible dependency list
└── .gitignore              # excludes media, models, environments, and outputs
```

## Setup

Requirements:

- Python 3.10+
- FFmpeg and ffprobe on `PATH`
- Internet access for initial media/model acquisition

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

PaddleOCR uses the pretrained `PP-OCRv5_mobile_det` and `en_PP-OCRv5_mobile_rec` models. faster-whisper defaults to the English `base.en` CTranslate2 checkpoint (about 148 MB), using CPU `int8` by default. These provide a practical accuracy/disk compromise and are downloaded once into `.cache/models`.

WhisperX is deliberately optional because its alignment dependencies and models are substantially heavier:

```powershell
python -m pip install -e ".[precision]"
```

The normal installation and pipeline work without it.

## Commands

```powershell
# Acquisition/probe only
quest1-v0 "https://example.com/public-video"

# Full-audio ASR localization
quest1-v1 "https://example.com/public-video" "target spoken dialogue"

# Caption-first localization with ASR verification
quest1-v2 "https://example.com/public-video" "target spoken dialogue" --language en

# Candidate-window visible-text verification
quest1-v3 "https://example.com/public-video" "target dialogue" --language en

# Final V4-hardened pipeline
quest1 "https://example.com/public-video" "target dialogue" --language en
```

Useful V4 options:

```powershell
quest1 URL DIALOGUE `
  --language en `
  --model base.en `
  --precision-mode default `
  --device cpu `
  --compute-type int8 `
  --fuzzy-threshold 85 `
  --caption-fuzzy-threshold 85 `
  --verification-margins 2,5 `
  --ocr-search-margin 1 `
  --ocr-fuzzy-threshold 85 `
  --work-dir .cache/media `
  --model-cache .cache/models `
  --output-dir output
```

Use `--precision-mode whisperx` when tighter word boundaries justify the extra dependency. The pipeline first transcribes with faster-whisper, then lazily attempts WhisperX alignment. If WhisperX is absent or alignment fails, it emits a warning, records `precision_fallback_reason`, and continues with the original faster-whisper timestamps.

URLs accidentally pasted as Markdown links are normalized before validation. Public provider/page URLs supported by yt-dlp and direct public media URLs are accepted; inputs are not restricted to `.mp4`. For legitimate signed-in access, use `--cookies-from-browser edge` or `--cookies-file PATH`. Credentials are never embedded by the application.

## Matching, occurrences, and frame selection

Dialogue and detected text are Unicode-normalized, case-folded, stripped of punctuation, and whitespace-collapsed. Timestamped ASR words are flattened across segment boundaries. Exact token windows are collected chronologically; if none exist, RapidFuzz checks small word-count variations for minor ASR/query errors. Internally, practical non-overlapping occurrences are retained in timestamp order, while the CLI continues to return the earliest occurrence by default through `occurrence_count` and the first match's real transcription text.

The first matched word timestamp is offset by the audio stream's start time when present. PyAV seeks to a preceding keyframe and decodes forward. The chosen frame is the earliest decoded frame whose `PTS * time_base` is at or after that timestamp. This handles variable frame rate and non-zero stream starts without calculating `timestamp × FPS`.

For visible text, V3/V4 decodes only a configurable interval around the localized dialogue and OCRs frames chronologically. It stops at the first frame where the complete target passes exact or accepted fuzzy matching. If no visible match exists, the earlier spoken-dialogue frame is returned unchanged.

## Confidence semantics

`confidence` is an explainable category, not a synthetic weighted score. Raw ASR, caption, and OCR match types/scores remain separate in the JSON.

| Category | Meaning |
|---|---|
| `HIGH` | Independent sources corroborate exact evidence: caption + short-window ASR, or spoken localization + exact OCR. |
| `MEDIUM` | A single exact ASR match, or corroborating sources where at least one accepted match is fuzzy. |
| `LOW` | ASR-only fuzzy evidence, any accepted evidence below the normal 85 threshold, or caption/ASR/OCR conflict. |

`confidence_reason` names the rule that produced the category. `evidence_conflict` is explicit, so conflicting caption, ASR, or OCR text cannot silently receive high confidence. Thresholds remain configurable for matching, but lowering them can produce a deliberately low-confidence accepted result.

## Cache reuse and disk behavior

All caching is file-based under `.cache`; there is no database or Redis.

- yt-dlp reuses already-acquired media in `.cache/media`.
- Successfully retrieved subtitle files remain beside the acquired media.
- ffprobe metadata is cached against resolved path, size, and modification time.
- Transcripts are keyed by audio content plus model, package/configuration, language, compute mode, and precision identity. Re-created temporary WAVs with identical content reuse safely.
- OCR results are keyed by exact frame pixels, dimensions/mode, and OCR model identity.

Full yt-dlp metadata is not persisted as an authoritative cache because provider caption URLs can be signed and expire. Temporary verification/full-ASR audio is deleted after use. The application selects only one ASR model for a run and never pre-downloads a set of models.

Cache provenance is returned as `transcript_cache_hit`, `media_metadata_cache_hit`, and `ocr_cache_hits`.

## Multilingual behavior

The optimized English path stays unchanged: no language, `en`, or an English regional tag uses `base.en`. A non-English `--language` request with the default model selects the multilingual `base` checkpoint for that run only. An explicitly requested multilingual model such as `small` is respected. An explicitly English-only `.en` model with a non-English language fails clearly instead of producing a misleading result.

Only the selected model is downloaded on demand, avoiding disk waste. Automatic language detection is still available when no language is supplied. The bundled PaddleOCR recognition model is English-specific, so non-English audio localization can work while visible-text verification for other scripts remains limited.

## Structured output

Successful V4 JSON includes:

- actual `matched_text`, dialogue start/end, and `occurrence_count`
- frame index, PTS, time base, timestamp, and saved path
- `localization_source`, `verification_source`, and `frame_match_type`
- raw ASR/caption/OCR match types and scores
- `confidence`, `confidence_reason`, and `evidence_conflict`
- ASR model/language, alignment source, precision mode/fallback reason
- processed audio/frame counts and cache-hit provenance

Expected failures—no match, missing audio/video, invalid media, failed ASR, or acquisition errors—produce a clear JSON error on stderr and a non-zero exit code.

## Tests

```powershell
pytest
```

The offline suite covers V0–V3 regressions plus V4 confidence rules, low/conflicting evidence, repeated/multiple occurrences, multilingual model configuration, transcript/metadata/OCR reuse, and WhisperX's optional fallback. Network/model end-to-end checks are separate because providers and model hosts are mutable.

## Modules

- `acquisition.py`, `inspection.py`, `captions.py`, `subtitles.py`: V0/V2 media and caption inputs
- `audio.py`, `transcription.py`: temporary speech audio, faster-whisper, and optional WhisperX alignment
- `matching.py`, `caption_matching.py`, `caption_verification.py`: normalized chronological evidence matching
- `frames.py`, `ocr.py`: PTS-based frame decoding and lazy candidate-window OCR
- `cache.py`: schema-versioned atomic JSON caches and safe content identities
- `confidence.py`: explicit HIGH/MEDIUM/LOW evidence rules
- `pipeline.py`: shared V0–V4 orchestration and fallbacks
- `cli.py`: input validation, options, structured output/errors

## Known limitations

- Providers can change extractors, require authentication, rate-limit captions, geo-block content, or use DRM. YouTube timed-text URLs may return HTTP 429; the pipeline handles this cleanly and falls back to ASR.
- The direct HTTP fallback cannot extract media hidden behind a provider HTML page. HTML/error responses are rejected, and TLS verification is never disabled globally.
- Signed caption URLs expire, which is why full provider metadata is not treated as a durable cache.
- Whisper/faster-whisper word times are estimates. WhisperX can improve boundaries, but is optional, heavier, language/model dependent, and can itself fail.
- Fuzzy matching is inherently ambiguous for very short or substantially incorrect targets. Low confidence and raw scores should be inspected.
- Multiple exact occurrences are robustly retained; fuzzy occurrence enumeration is best-effort and defaults still return the first accepted candidate.
- The standard PaddleOCR recognizer is English-specific. Full-frame OCR is bounded but can still be CPU-intensive for long intervals or high-frame-rate video.
- Embedded-only subtitles are inventoried but are not extracted as V2 caption candidates.
- Only one URL is processed per invocation; playlists are disabled.

See `prompts.md` for the AI-assistance disclosure and verbatim prompt history.
