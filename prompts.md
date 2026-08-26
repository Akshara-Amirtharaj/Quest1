# Quest1 Dialogue Frame Finder --- Prompts Used

This is a short record of the main GPT and Codex prompts that shaped the
project. The prompts are shortened to show what I was trying to achieve
and what part of the pipeline was built from each one.

## Pipeline evolution

`Media foundation` → `Full-audio ASR` → `Caption-first localization` →
`Caption hardening` → `OCR / visible text` → `Pipeline hardening` →
`Matching + edge cases` → `Performance experiments` → `Frontend + API` →
`Final integration`

------------------------------------------------------------------------

## 1. Plan the complete pipeline

**Prompt**

> Given a public video URL and target dialogue, design a pipeline that
> can find the first occurrence, timestamp and corresponding frame. Use
> captions, audio/ASR and OCR as different evidence sources. Correctness
> comes first, then optimization. Do not hardcode the sample video or
> dialogue.

**What part of the pipeline was built**

-   Defined the overall progressive-evidence architecture.
-   Captions were treated as cheap localization evidence, ASR as spoken
    verification, and OCR as visible-text evidence.
-   PyAV PTS was chosen for final frame timing instead of
    `timestamp × FPS`.

------------------------------------------------------------------------

## 2. V0 --- Media foundation

**Prompt**

> Implement only V0 first. Acquire a public video using yt-dlp, inspect
> it with ffprobe, discover available captions and decode a sample frame
> with PyAV. Keep the project small and modular. Do not add ASR or OCR
> yet.

**What part of the pipeline was built**

-   URL → media acquisition → metadata/stream inspection.
-   Caption/subtitle availability detection.
-   PyAV frame decoding with PTS and time-base metadata.
-   Basic dependency checks, errors and tests.

------------------------------------------------------------------------

## 3. Fix direct media acquisition failures

**Prompt**

> yt-dlp is failing for some direct video URLs because of SSL/provider
> issues. Check whether direct HTTP download can be used as a safe
> fallback without replacing yt-dlp or hardcoding a provider. Make the
> smallest change and keep normal URLs working.

**What part of the pipeline was built**

-   Added/hardened direct-media acquisition fallback.
-   Kept yt-dlp as the main provider/page acquisition path.
-   Improved acquisition failure handling without changing later stages.

------------------------------------------------------------------------

## 4. V1 --- First complete spoken-dialogue pipeline

**Prompt**

> Build the first complete working solution using audio only. Extract
> speech-friendly audio, run faster-whisper with timestamps, normalize
> and fuzzy-match the target, find the first occurrence, then use PyAV
> PTS to extract the corresponding frame. Keep captions and OCR out of
> V1.

**What part of the pipeline was built**

-   Media → FFmpeg audio extraction → faster-whisper transcription.
-   Exact/normalized/fuzzy dialogue matching with RapidFuzz.
-   Multi-segment matching and first-occurrence selection.
-   ASR timestamp → PyAV PTS frame → saved image.

------------------------------------------------------------------------

## 5. V2 --- Caption-first localization

**Prompt**

> Use captions when available to avoid transcribing the full video.
> Captions should only locate candidate regions, not act as ground
> truth. Verify each promising region using short-window ASR. If
> captions fail or are unavailable, fall back to the existing full-audio
> ASR path.

**What part of the pipeline was built**

-   Caption discovery, parsing and target matching.
-   Chronological caption candidate generation.
-   Short-window ASR verification with progressive widening.
-   Reliable full-ASR fallback when caption localization fails.

------------------------------------------------------------------------

## 6. Harden caption acquisition

**Prompt**

> The pipeline says captions are unavailable even when the video has
> them. If one subtitle format fails, try other formats/tracks for the
> same language. Handle 429, expired URLs and invalid subtitle responses
> cleanly. Preserve manual captions → auto captions → ASR fallback.

**What part of the pipeline was built**

-   More reliable manual/automatic caption retrieval.
-   Alternative subtitle-format fallback.
-   Invalid/HTML/expired caption response rejection.
-   Caption failure no longer breaks the ASR fallback.

------------------------------------------------------------------------

## 7. V3 --- Visible-text / OCR path

**Prompt**

> Add support for dialogue that is visibly present in video frames.
> First use captions/ASR to narrow the candidate interval, then run
> PaddleOCR only around that region. Return the earliest frame where the
> complete target is visible. Do not OCR the whole video.

**What part of the pipeline was built**

-   Optional visible-text verification path using OCR.
-   Local frame decoding around an already localized interval.
-   Exact/fuzzy OCR matching and earliest complete visible-frame
    selection.
-   Spoken-dialogue behavior remained independent of OCR.

------------------------------------------------------------------------

## 8. V4 --- Production hardening

**Prompt**

> Harden the working V0--V3 pipeline without rewriting it. Add
> confidence/provenance, caching, multilingual configuration,
> repeated-dialogue handling and better edge cases. Heavy models should
> be lazy and optional. Keep first occurrence as the default result.

**What part of the pipeline was built**

-   Confidence and provenance fields for results.
-   Lightweight reuse/caching of expensive artifacts.
-   Better handling of repeated/no-match/low-confidence cases.
-   Multilingual and optional precision-path support.
-   Heavy stages kept lazy so normal spoken requests do not require
    them.

------------------------------------------------------------------------

## 9. Harden dialogue matching

**Prompt**

> Test matching against realistic failures: punctuation, apostrophes,
> small ASR errors, numbers/currency, long targets, dialogue split
> across segments, repeated dialogue, short ambiguous phrases, stutters
> and noisy speech. Preserve the earliest valid occurrence and avoid
> semantic/LLM matching.

**What part of the pipeline was built**

-   Stronger normalization and fuzzy matching.
-   Neighbouring-word/segment candidate construction.
-   Better first-occurrence semantics.
-   Matching became tolerant to small transcript differences without
    becoming overly permissive.

------------------------------------------------------------------------

## 10. Validate exact frame timing

**Prompt**

> Make sure the returned frame is based on actual decoded video timing,
> not nominal FPS arithmetic. Test variable-frame-rate and
> unusual/non-zero timelines. The frame should correspond to the
> localized dialogue start as closely as the media timestamps allow.

**What part of the pipeline was built**

-   PTS/time-base remained the authoritative frame timing.
-   VFR/non-zero timestamp cases were hardened.
-   Frame extraction stayed separate from ASR timestamp estimation.

------------------------------------------------------------------------

## 11. Explore faster localization

**Prompt**

> The working pipeline is accurate but slow on long videos. Explore
> cheaper ways to locate likely regions before expensive ASR:
> chronological chunks, overlap, VAD, lightweight Whisper or keyword
> spotting. Do not sacrifice correctness; uncertain paths must fall back
> to the reliable pipeline.

**What part of the pipeline was built**

-   Created a separate performance/benchmark track.
-   Explored chunked ASR, VAD, lightweight localization and KWS.
-   Production behavior was not replaced just because an optimization
    looked faster.

------------------------------------------------------------------------

## 12. Benchmark chunked ASR properly

**Prompt**

> Compare the full-ASR baseline against a frozen chunked-ASR
> configuration across beginning, middle, end, absent, repeated and
> boundary cases. Do not tune parameters per sample. Measure
> correctness, first occurrence, runtime and expensive ASR audio
> processed.

**What part of the pipeline was built**

-   Controlled baseline-vs-chunked benchmark.
-   Reusable media/cache for repeatable tests.
-   Optimization decisions were based on measured correctness and
    runtime rather than intuition.

------------------------------------------------------------------------

## 13. Build a thin frontend + API

**Prompt**

> Build a simple demo frontend with video URL, target dialogue and
> optional language. Show loading/error states and return frame,
> timestamp, matched text, confidence and source. Add only a thin
> FastAPI adapter around the existing Python pipeline; do not duplicate
> the search logic.

**What part of the pipeline was built**

-   Browser UI for submitting dialogue-frame searches.
-   Thin FastAPI `/api/find` integration with the existing core.
-   Generated frames served through the API.
-   Frontend remained separate from localization logic.

------------------------------------------------------------------------

## 14. Fix frontend/backend integration

**Prompt**

> The frontend/backend must be stable first; speed can come later. Check
> stale worktree/import issues, duplicate requests and why ordinary
> spoken requests are triggering OCR. Keep one request per click and
> make sure the API calls the correct production pipeline.

**What part of the pipeline was built**

-   Removed stale worktree import risk from runtime.
-   Kept one frontend submit → one API request.
-   Ordinary spoken localization stopped unnecessarily initializing OCR.
-   Stable frontend → API → production pipeline call chain.

------------------------------------------------------------------------

## 15. Improve frontend errors

**Prompt**

> Do not show raw technical errors to the user. Handle missing
> URL/dialogue, invalid URL, unavailable media, dialogue not found and
> backend failure with clear messages. Treat "dialogue not found" as a
> normal result. Remove internal V4/pipeline labels from the UI.

**What part of the pipeline was built**

-   Clean domain-level API/UI errors.
-   `NO_MATCH` separated from actual processing failures.
-   Internal paths/provider errors were not exposed.
-   Final demo UI was simplified.

------------------------------------------------------------------------

## 16. Recheck caption-first behavior in the final app

**Prompt**

> This video clearly has captions but the final pipeline is falling into
> long ASR. Trace caption discovery and retrieval before optimizing
> anything else. Make sure usable captions actually enter caption
> matching and bounded ASR verification before full-audio fallback.

**What part of the pipeline was built**

-   Final integration rechecked caption-first routing.
-   Caption availability/retrieval failures became visible instead of
    silently appearing as "no captions".
-   Preserved the intended path: caption candidate → bounded ASR → full
    ASR only if required.

------------------------------------------------------------------------

## Final production flow

``` text
Video URL + target dialogue
        ↓
Media acquisition + inspection
        ↓
Caption discovery
        ↓
Useful caption candidate?
   ┌────Yes────┐
   ↓           │
Caption match  │
   ↓           │
Bounded ASR verification
   │           │
   └──fails────┘
        ↓
Full / broader ASR fallback
        ↓
Robust target matching
        ↓
First valid occurrence
        ↓
Dialogue start timestamp
        ↓
PyAV PTS-based frame resolution
        ↓
Frame + timestamp + matched text
+ confidence/provenance
        ↓
Thin FastAPI layer
        ↓
Frontend
```

## Explored separately / not required in the normal spoken path

-   Chunked-ASR optimization.
-   Voice activity detection (VAD).
-   Open-vocabulary keyword spotting (KWS).
-   Lightweight-ASR localization.
-   OCR is a separate visible-text modality and should not run for an
    ordinary spoken-only result.
-   More expensive precision/alignment paths remain optional rather than
    mandatory.
