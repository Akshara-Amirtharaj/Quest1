# Approach and Architecture

## Objective

Quest1 receives a public video URL and target dialogue, finds the earliest spoken occurrence, and saves the corresponding video frame. It must work across provider pages and direct media URLs, tolerate missing or unreliable captions, and preserve timestamp correctness for variable-frame-rate media.

## Design principles

1. Prefer inexpensive evidence before expensive processing.
2. Treat captions as localization hints, not ground truth.
3. Verify spoken dialogue using timestamped audio recognition.
4. Resolve frames from presentation timestamps instead of `timestamp × FPS`.
5. Bound OCR to the already-localized interval.
6. Preserve raw evidence and explain confidence rather than hiding uncertainty.
7. Cache deterministic work without introducing a database or cloud service.

## Processing pipeline

```text
URL + target dialogue
  -> validate and normalize the URL
  -> reuse cached media or acquire with yt-dlp/direct HTTP
  -> inspect streams and timing with ffprobe
  -> require decodable video and audio
  -> discover manual and automatic caption tracks
     -> normalize and search caption windows
     -> short-window faster-whisper verification
     -> if no verified candidate: full-audio faster-whisper
  -> normalize timestamped words and exact/fuzzy match
  -> choose the earliest accepted occurrence
  -> optionally refine word times with WhisperX
  -> PyAV keyframe seek and PTS-based forward decode
  -> bounded chronological PaddleOCR search
     -> visible-text frame when found
     -> spoken-dialogue frame otherwise
  -> emit one structured result with evidence and provenance
```

## Media acquisition

`yt-dlp` handles provider pages and selects video up to 720p when practical, with a best-available fallback. Direct HTTP media URLs use a streamed `requests` fallback with TLS verification and content-type checks. Authentication can be supplied through browser or Netscape cookie files.

Successful URL acquisitions are mapped to local media files. This allows a completed download to remain usable when the original provider later becomes unavailable. Legacy files created before the URL index are adopted safely using provider and media identifiers.

Playlists are intentionally disabled; one URL is processed per invocation.

## Metadata and caption strategy

ffprobe supplies stream presence, codecs, dimensions, duration, time bases, start times, and embedded-subtitle inventory. Provider metadata supplies manual and automatic caption tracks.

Manual captions are attempted before automatic captions. Supported formats include JSON3, WebVTT, SRT, TTML, ASS, and SSA. Downloaded tracks are cached, parsed into chronological cues, and searched across small multi-cue windows so phrases split across subtitle events remain matchable.

Caption comparison supports:

- exact normalized equality;
- the complete query as a substring;
- RapidFuzz similarity above a configurable threshold.

A caption match only proposes a time interval. faster-whisper must still find the target dialogue inside a short audio window around that interval. Increasing verification margins handle caption timing drift. Unavailable, empty, malformed, rate-limited, or unverified captions fall back to full-audio ASR.

## Speech recognition and matching

Audio is converted temporarily to 16 kHz mono PCM for faster-whisper. The default English configuration is `base.en` on CPU with `int8`; non-English language requests automatically select the multilingual `base` model unless another multilingual model is supplied.

Word timestamps are flattened across ASR segment boundaries. Text normalization uses Unicode NFKC normalization, case folding, punctuation removal, and whitespace collapse. Matching first collects exact token windows. If none exist, RapidFuzz examines small word-count variations to tolerate insertions, omissions, and recognition errors.

Practical non-overlapping occurrences are retained chronologically. The first occurrence is returned by default and `occurrence_count` reports how many were found.

WhisperX is an optional alignment refinement. It runs after faster-whisper transcription but before final frame resolution. If unavailable or unsuccessful, the pipeline records the reason and safely retains faster-whisper timings.

## Frame resolution and OCR

PyAV seeks backward to a preceding keyframe, decodes forward, and selects the first decoded frame whose `PTS × time_base` is at or after the matched first-word timestamp. This works with variable frame rates and avoids fabricated global frame calculations.

Visible-text verification decodes only a configurable interval around the spoken match. PaddleOCR checks frames chronologically and stops at the first exact or accepted fuzzy occurrence of the complete target. If no visible match exists or OCR is unavailable, the spoken-dialogue frame remains the result.

## Confidence and provenance

Confidence is rule-based and explainable:

- **HIGH:** independent exact caption/ASR or spoken/OCR corroboration;
- **MEDIUM:** exact ASR alone, or corroboration containing a fuzzy match;
- **LOW:** fuzzy ASR alone, weak accepted evidence, or evidence conflict.

The JSON result retains individual caption, ASR, and OCR match scores along with localization source, verification source, alignment source, model, language, timestamps, processed duration/frame counts, and cache hits.

## Caching

The implementation uses schema-versioned files rather than a database:

- URL-to-media acquisition mapping;
- downloaded subtitle files;
- ffprobe results keyed by file identity;
- transcripts keyed by audio content and ASR configuration;
- OCR results keyed by exact frame pixels and OCR model identity.

Temporary extracted audio is deleted after use. Media, model weights, generated frames, and cache files are excluded from Git.

## Failure handling

Expected failures return clear structured errors: invalid URLs, acquisition failures, missing streams, malformed captions, ASR failure, no accepted dialogue match, and frame decoding failure. Caption and OCR failures degrade to stronger available evidence rather than masking a valid spoken match.

## Current limitations and next optimization work

- Embedded subtitle streams are inventoried but not yet extracted as caption candidates.
- OCR currently runs automatically in V4; explicit `off`, `auto`, and `required` modes are not yet exposed.
- Full-audio fallback completes the transcription before searching; sequential chunking could return the first occurrence earlier.
- OCR checks every decoded frame in the bounded interval; sampling, duplicate-frame suppression, subtitle-region cropping, and batching could reduce CPU cost.
- An OCR success can leave both the initial spoken frame and visible frame on disk even though the result points to only one.
- Alternate same-language caption tracks are acquisition fallbacks rather than independently searched tracks.
- WhisperX is optional and is not part of the default installation.

These items are intentionally documented as future optimization work and are not hidden by the current test suite.

## Verification

The offline suite covers acquisition, inspection, caption parsing/matching, short-window verification, ASR matching, PTS frame resolution, OCR, confidence rules, multilingual configuration, caching, occurrence detection, and V0–V4 regressions. Live provider checks are supplemental because provider availability and signed caption URLs change independently of the code.
