# Approach

This shows how I built the solution step by step. I did not start with
the final pipeline directly. At each stage, I added one part, tested it,
found the next limitation, and improved it.

---

## 1. Initial idea

### What we did

First, I split the problem into two parts:

1. Find where the target dialogue occurs.
2. Resolve that timestamp to the correct video frame.

Captions, speech and visible text were considered as possible evidence.

### What improved

This separated dialogue localization from exact frame extraction.

### Flow

```text
Video URL + Target Dialogue
            |
            v
      Acquire Media
            |
            v
 Find Dialogue Timestamp
            |
            v
 Resolve Decoded Frame
            |
            v
 Timestamp + Exact Frame
```

---

## 2. V0 - Media foundation

### What we did

Before adding ASR or OCR, I first made sure the media itself could be
handled properly.

- yt-dlp acquired one local media file.
- ffprobe inspected video, audio and embedded subtitle streams.
- yt-dlp metadata supplied platform and automatic caption inventories.
- PyAV decoded one sample video frame.

### What improved

All later stages now had one common media and inspection layer.

### Flow

```text
Video URL
    |
    v
Validate URL + External Tools
    |
    v
yt-dlp Media Acquisition
    |
    v
Local Media + Provider Metadata
    |
    +--> ffprobe --> Video / Audio / Embedded Subtitles
    |                Codec / Dimensions / FPS / Time Base
    |
    +--> Caption Discovery --> Manual / Automatic Tracks
    |
    v
PyAV Decode First Video Frame
    |
    v
V0 Media Result + Sample Frame
```

---

## 3. Acquisition hardening

### What we did

Some direct media URLs could fail through the normal yt-dlp path even
when the media was accessible.

I kept yt-dlp as the main path, added URL-indexed cache reuse, rejected
incomplete artifacts, and used streamed direct HTTP only as a fallback.

### What improved

A provider or direct-media issue no longer required changing the rest
of the pipeline, and partial files could not become authoritative media.

### Flow

```text
Video URL
    |
    v
Validate Public URL
    |
    v
Cached Complete Media?
   / \
 yes  no
  |    |
  |    v
  |  yt-dlp Acquisition
  |    / \
  | success failure
  |    |      |
  |    |      v
  |    |  Streamed Direct HTTP
  |    |      / \
  |    |   success failure
  |    |      |      |
  +----+------+      v
       |       Acquisition Error
       v
Confirmed Complete Local Media
       |
       v
ffprobe / Later Pipeline Stages
```

---

## 4. V1 - First complete spoken-dialogue pipeline

### What we did

Once media handling worked, I built the first complete path using spoken
audio.

The media was inspected, speech audio was extracted, faster-whisper
produced timestamped words, and matching returned occurrences in
chronological order. The first occurrence was resolved with PyAV.

### What improved

The project could now go from URL + dialogue all the way to a spoken
timestamp and exact decoded frame.

### Flow

```text
Video URL + Target Dialogue
            |
            v
 Acquire Complete Local Media
            |
            v
 ffprobe: Require Audio + Video
            |
            v
 FFmpeg Extract Mono 16 kHz WAV
            |
            v
 faster-whisper + Word Timestamps
            |
            v
 Normalize + Exact/Fuzzy Word Windows
            |
            v
 All Accepted Occurrences in Time Order
            |
            v
 First Occurrence Start Timestamp
            |
            v
 PyAV Backward Seek + Decode Forward
            |
            v
 First Frame at/after Timestamp by PTS x time_base
```

---

## 5. V2 - Caption-first localization

### What we did

Full ASR worked, but transcribing an entire long video was unnecessary
when captions could first narrow the search.

Captions became localization evidence, not timing ground truth. Manual
tracks are searched before automatic tracks. Each chronological caption
candidate is checked by ASR on increasing bounded audio windows. If no
candidate verifies, the path rejoins the full-audio ASR fallback.

### What improved

Usable captions reduced ASR work while actual audio still verified the
dialogue and supplied the final spoken timestamp.

### Flow

```text
Video URL + Target Dialogue
            |
            v
 Acquire + Inspect Complete Media
            |
            v
 Discover / Select Caption Tracks
            |
            v
 Manual Tracks, then Automatic Tracks
            |
            v
 Download + Parse + Caption Window Match
            |
      caption candidates?
          /       \
        yes        no / unusable / 429
         |                 |
         v                 |
 Chronological Candidates  |
         |                 |
         v                 |
 FFmpeg Bounded Audio      |
         |                 |
         v                 |
 Base ASR Verification     |
      /       \            |
 verified   all rejected   |
    |            \         |
    |             v        v
    |        Full-Audio ASR
    |             |
    |             v
    |       Transcript Matching
    |             |
    +-------------+
          |
          v
 Spoken Match Timestamp
          |
          v
 PyAV PTS Frame Resolution
```

---

## 6. Caption acquisition hardening

### What we did

A video could advertise captions while one format or response was
expired, malformed, empty, HTML, rate-limited or otherwise unusable.

Caption selection keeps manual before automatic, groups tracks by
language, prefers a cached usable file, and tries format variants until
one parses. Candidates from usable languages are merged chronologically.

### What improved

A single bad subtitle response no longer discarded every caption route,
while a 429 or exhausted track set still fell back cleanly to ASR.

### Flow

```text
Caption Inventory + Requested Language
                 |
                 v
      Select Matching Languages
                 |
                 v
        Manual Source First
                 |
                 v
 Group Tracks by Language
                 |
                 v
 Cached Track First, then Format Variants
          /              \
      parses            fails
        |                 |
        v                 v
 Caption Windows     Next Variant / Language
        |                 |
        |          source exhausted?
        |             /       \
        |           no         yes
        |           |           |
        |           +-----------+--> Automatic Source
        |                            |
        +----------------------------+
                     |
                     v
        Merge Candidates Chronologically
                     |
          candidates? / no candidates or 429
              |                 |
              v                 v
      Bounded ASR Verify   Full-ASR Fallback
```

---

## 7. V3 - OCR / visible-text verification

### What we did

OCR was added after spoken localization, not as a second whole-video
localizer.

V3 first runs the complete V2 path and already has an authoritative
spoken frame. It then decodes only a small interval around the spoken
match. If PaddleOCR finds the complete target, that earlier visible-text
frame replaces the spoken frame. Any OCR failure keeps the spoken result.

### What improved

Visible text could refine the selected frame without making OCR a
required dependency or scanning the whole video.

### Flow

```text
Video URL + Target Dialogue
            |
            v
 Complete V2 Spoken Localization
            |
            +--> Spoken Match + Spoken PTS Frame (safe fallback)
            |
            v
 Match Interval +/- OCR Margin
            |
            v
 PyAV Seek + Decode Bounded Frames
            |
            v
 Lazy Cached PaddleOCR + Visible-Text Match
       /                         \
 exact/fuzzy visible match   no match / OCR error
       |                         |
       v                         v
 Earliest Visible Frame      Keep Spoken Frame
       \                         /
        +-----------+-----------+
                    |
                    v
 V3 Result + Frame-Match Provenance
```

---

## 8. V4 - Pipeline hardening

### What we did

V4 kept the V3 algorithm and exposed it as the final CLI pipeline.
Hardening added cached evidence, confidence/source details, language and
alignment configuration, occurrence reporting, conflict reporting and
safe optional-component fallbacks.

### What improved

The CLI result became easier to audit and failures in optional OCR or
alignment no longer erased an already valid spoken result.

### Flow

```text
CLI: URL + Target + Optional Language / Precision Mode
                         |
                         v
                      run_v4
                         |
                         v
                      run_v3
                         |
                         v
          Complete V2 Spoken Pipeline
                         |
                         +--> Caption evidence or Full-ASR evidence
                         +--> First spoken occurrence
                         +--> Spoken PTS frame
                         |
                         v
             Optional Bounded OCR Refinement
                  /                  \
             visible match       unavailable / miss
                  |                  |
                  +--------+---------+
                           |
                           v
 Frame + Timestamp + Original ASR Text + Occurrences
 Confidence + Evidence Sources + Cache / Model Provenance
```

---

## 9. Matching and precision hardening

### What we did

I hardened matching without changing the target thresholds. Matching
normalizes Unicode, punctuation, numbers and currency, searches
variable word windows across segment boundaries, and preserves the
earliest accepted occurrence.

For English caption-candidate windows, a rejected base-ASR result can
trigger lazy `distil-large-v3` ASR only when its best score is inside the
configured uncertainty band. Full-audio precision retry remains opt-in.

### What improved

Small transcript differences and recoverable base-model misses stopped
causing unnecessary failures, while expensive precision ASR stayed
bounded and lazy.

### Flow

```text
Target + Timestamped ASR Words
              |
              v
 Unicode / Punctuation / Number Normalization
              |
              v
 Variable Cross-Segment Word Windows
              |
              v
 Base-ASR Accepted?
       /                \
     yes                 no
      |                   |
      |          Candidate-window precision enabled
      |          and best score in trigger band?
      |                /             \
      |              yes              no
      |               |                |
      |               v                v
      |       Lazy Precision ASR   Preserve Rejection
      |               |
      +-------+-------+
              |
              v
 Accepted Matches in Chronological Order
              |
              v
 First Accepted Occurrence + Original Selected-ASR Text
```

---

## 10. Exact frame timing

### What we did

I used decoded presentation timing instead of timestamp x FPS.

PyAV seeks backward to a preceding keyframe, decodes forward, and picks
the first frame whose real `PTS x time_base` is at or after the dialogue
timestamp. The final PNG is written atomically.

### What improved

Frame selection now follows real variable-frame-rate timing and a
partial image cannot become the published result.

### Flow

```text
Dialogue Start Timestamp
           |
           v
Seek Target = max(0, Timestamp - Margin)
           |
           v
PyAV Backward Keyframe Seek
           |
           v
Decode Frames Forward
           |
           v
For Each Frame: Presentation Time = PTS x time_base
           |
    before target? ---- yes ----> Decode Next Frame
           |
          no
           |
           v
First Frame at/after Target
           |
           v
Atomic PNG Publish + PTS / Time Base / Timestamp
```

---

## 11. ASR optimization exploration - EXPERIMENTAL

### What we did

I kept long-audio optimization work outside production and explored
four separate strategies:

- chronological chunked ASR with overlap and early stopping;
- chunked ASR with conservative Silero VAD filtering;
- lightweight Whisper localization followed by production-ASR verification;
- open-vocabulary KWS candidates followed by production-ASR verification.

Each experiment reused production matching or verification where its
runner actually called it. These strategies were not connected to
`run_v2`, `run_v3` or `run_v4`.

### What improved

Performance ideas could be measured without changing the reliable
production control flow.

### Flow

```text
                         Long Audio
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
 Chronological Chunks   Chunks + VAD     Lightweight Locator
 + Overlap + Context    + Base ASR       or Open KWS
          |                  |                  |
          v                  v                  v
 Production Matcher    Production Matcher   Candidate Regions
          |                  |                  |
          |                  |                  v
          |                  |         Production-ASR Verification
          |                  |                  |
          +------------------+------------------+
                             |
                             v
                     Experimental Result

Status: EXPERIMENTAL - not called by the production pipeline.
```

---

## 12. Frozen chunked-ASR benchmark - EXPERIMENTAL

### What we did

Instead of changing many parameters at once, I froze one chunked setup
and compared it against the full-ASR baseline on the same manifest and
fixtures.

The matrix covered beginning, middle, end, repeated, absent,
chunk-boundary and controlled-noise cases. Later diagnostic work stayed
inside the experiment tree.

### What improved

This gave accuracy, first-occurrence and processing-cost evidence before
any decision about production integration.

### Flow

```text
Frozen Manifest + Ground Truth / Controlled Fixtures
                         |
              +----------+----------+
              |                     |
              v                     v
       Full-ASR Baseline      Frozen Chunked Runner
              |             Chunk + Overlap + Context
              |                     |
              v                     v
       Baseline Records       Candidate Records
              \                     /
               +---------+---------+
                         |
                         v
        Compare Match / No-Match / First Occurrence
        Timestamp Error / Elapsed Time / Audio Used

Status: EXPERIMENTAL - benchmark evidence only.
```

---

## 13. Frontend + FastAPI integration

### What we did

I added a simple frontend and a thin FastAPI layer. The API validates the
request, creates per-request output storage, calls the existing Python
pipeline, registers the generated frame behind an opaque ID, and adapts
the pipeline result into the public response.

### What improved

The project became easy to demonstrate without duplicating dialogue
localization logic in the web layer.

### Flow

```text
Browser Form: Video URL + Dialogue
                 |
                 v
Client Validation + One POST /api/find
                 |
                 v
FastAPI FindRequest Validation
                 |
                 v
Per-Request Work / Output Directories
                 |
                 v
Existing Python Pipeline
                 |
                 v
Frame Registry + Structured Response Adapter
                 |
        +--------+--------+
        |                 |
        v                 v
JSON Match/Evidence   GET /api/frames/{opaque-id}
        |                 |
        +--------+--------+
                 |
                 v
Frontend Frame + Timestamp + Confidence + Source
```

---

## 14. Frontend/API hardening

### What we did

Once the frontend was connected, I fixed integration issues instead of
changing the core algorithm.

The packaged FastAPI app became the root-installable public demo. The
browser blocks duplicate in-flight submissions, while the API calls
`run_v2` directly for spoken-dialogue requests. It does not call
`run_v4`, so PaddleOCR is not loaded by the web route.

### What improved

The demo now has one request, one production backend call and one result
without starting an unnecessary heavy OCR path.

### Flow

```text
Frontend Submit
      |
      v
Request Already In Flight?
   /             \
 yes              no
  |                |
ignore             v
             POST /api/find
                    |
                    v
           FastAPI find_frame()
                    |
                    v
                 run_v2
                    |
          Caption / ASR Spoken Path
                    |
                    v
             One Spoken Result
                    |
                    v
        Adapt JSON + Serve Registered PNG

No run_v3/run_v4 call and no OCR in the public web route.
```

---

## 15. User-friendly error handling

### What we did

Raw validation, yt-dlp, media and Python errors should not be shown
directly in the frontend.

The browser handles simple missing-input errors. FastAPI maps request
validation and pipeline failures to stable public codes, logs internal
details, and always returns a safe message.

### What improved

Dialogue absence is now distinct from an unavailable video or an
unexpected processing failure.

### Flow

```text
Browser Input
    |
    +--> Missing / Invalid Field --> Inline Public Message
    |
    v
FastAPI Request Validation
    |
    +--> Validation Error --> Required / Invalid Public Code
    |
    v
run_v2
  /   \
success failure
  |      |
  |      v
  |   Classify V0Error Message
  |      |
  |      +--> Dialogue Not Found (404)
  |      +--> Video Unavailable (422)
  |      +--> Processing Failed (500)
  |      |
  |      v
  |   Safe JSON Error
  |      |
  +------+
         |
         v
Frontend Result or Friendly Error State
```

```text
Missing URL       -> Video URL required
Invalid URL       -> Invalid video URL
Missing dialogue  -> Dialogue required
No match          -> Dialogue not found
Unavailable video -> Video unavailable
Unexpected error  -> Processing failed
```

---

## 16. Final caption-first and audio-first verification

### What we did

Near the end, I rechecked both optimized production acquisition routes.

Metadata inspection happens before a full download. If caption matching
finds no candidate, the pipeline downloads a genuine audio-only stream,
runs full ASR, and resolves the frame through the preserved remote
video-only URL. If remote seeking fails, it falls back to complete media.
If caption candidates exist, the current implementation acquires
complete media before bounded ASR verification.

### What improved

The final spoken pipeline can avoid a full-video download when captions
cannot localize the target, while preserving the same ASR matching and
PTS-correct frame fallback.

### Flow

```text
Video URL + Target Dialogue
            |
            v
yt-dlp Metadata-Only inspect_source()
      /                         \
 inspection works          inspection fails
      |                         |
      v                         v
Discover / Select Captions   Acquire Complete Media
      |                         |
      v                         +------> Standard V2 Caption/ASR Path
Pre-check Caption Matches
   /                 \
candidate exists     no candidate / caption failure
   |                         |
   v                         v
Acquire Complete Media    acquire_audio_only()
   |                         |
   v                         v
Run Caption Search Again  ffprobe Audio + Full-Audio ASR
   |                         |
   v                         v
Bounded Base ASR          Transcript Match + First Occurrence
   |                         |
   +--> Precision Retry      v
   |    only if eligible   Preserved Remote Video URL?
   |                       /                    \
verified / rejected      yes                    no
   |          |           |                      |
   |          v           v                      |
   |      Full-Audio ASR  PyAV Remote Seek       |
   |          |           + Decode Forward       |
   |          |           + PTS x time_base      |
   |          |              /       \           |
   |          |          success     failure     |
   |          |             |          |         |
   |          |             |          v         v
   |          |             |      Acquire Complete Media
   |          |             |          |
   +----------+-------------+----------+
                         |
                         v
              First Frame at/after Match
```

---

## Final validated production flow

The public application uses the spoken-dialogue `run_v2` route. OCR and
all long-audio experiments remain outside this web path.

### What we did

I traced the current public entry point from the browser submit handler
through FastAPI `find_frame()`, `run_v2()`, acquisition, localization,
matching, frame extraction and response adaptation.

### What improved

The final architecture now shows the real caption, audio-only, remote
seek and full-media fallback branches in their actual execution order.

### Flow

```text
Frontend submit handler
Video URL + Target Dialogue
            |
            v
POST /api/find --> FastAPI find_frame()
            |
            v
Validate Request + Create Request ID / Output Directory
            |
            v
run_v2()
            |
            v
Validate URL + Require FFmpeg / ffprobe
            |
            v
Metadata-Only Source Inspection
       /                             \
  available                        unavailable
      |                                |
      v                                v
Caption Inventory / Selection      [FULL MEDIA]
      |
      v
Caption Candidate Pre-check
   /                    \
found                    none
  |                       |
  v                       v
[FULL MEDIA]       acquire_audio_only()
                          |
                          v
                    ffprobe: audio
                          |
                          v
                   Full-Audio Base ASR
                          |
                          v
                    Robust Matching
                          |
                          v
                  Earliest Spoken Match
                          |
                          v
                Preserved Remote Video URL?
                     /             \
                   yes              no
                    |                |
                    v                v
              PyAV Remote Seek   [FULL MEDIA]
                 /       \
             success    failure
                |          |
                v          v
            [PUBLISH]  [FULL MEDIA]

[FULL MEDIA]
  |
  v
acquire_media(): cache -> yt-dlp -> direct HTTP fallback
  |
  v
Local Complete Media + ffprobe Audio/Video
  |
  v
Caption Search: Manual then Automatic
  |
  +--> no candidates / unusable / 429
  |              |
  |              v
  |       Full-Audio Base ASR
  |              |
  |              v
  |       Robust Matching
  |              |
  |              v
  |       Earliest Spoken Match
  |
  +--> chronological caption candidates
                 |
                 v
          Bounded Audio Windows
                 |
                 v
          Base ASR Verification
            /               \
        accepted           rejected
           |                  |
           |          uncertainty-band score?
           |              /            \
           |            yes             no
           |             |               |
           |             v               v
           |       Precision ASR     Next Candidate
           |             |               |
           |          accepted?       all rejected
           |           /      \          |
           |         yes       no---------+
           |          |                   |
           +----------+                   v
           |                       Full-Audio Base ASR
           |                              |
           |                              v
           |                       Robust Matching
           |                              |
           +--------------+---------------+
                          |
                          v
     Earliest Accepted Spoken Match + Original Selected-ASR Text
                          |
                          v
              PyAV Local Backward Keyframe Seek
              Decode Forward until PTS x time_base >= Start
                          |
                          v
                     [PUBLISH]

[PUBLISH]
     |
     v
Exactly One Authoritative Frame
                     |
                     v
Atomic dialogue_frame.png
                     |
                     v
V1Result: Match / Occurrences / Confidence / Provenance
                     |
                     v
FastAPI _adapt_result() + Opaque Frame Registry
                     |
          +----------+----------+
          |                     |
          v                     v
JSON Response              GET Registered PNG
          \                     /
           +---------+---------+
                     |
                     v
Frontend Frame + Timestamp + Match + Confidence + Source

Not in this web route:
  run_v3/run_v4 bounded OCR
  chunking / VAD / lightweight locator / KWS experiments
```

## In short

```text
Media Foundation
      |
      v
Full-Audio Spoken Localization
      |
      v
Caption-First Localization + Bounded ASR Verification
      |
      v
Matching / First-Occurrence / Precision Hardening
      |
      v
Optional Bounded OCR in V3/V4 CLI Only
      |
      v
Separate Optimization Experiments
      |
      v
Frontend + Thin FastAPI run_v2 Route
      |
      v
Audio-First Acquisition + Remote PTS Seek Fallback
      |
      v
Final Public Error Handling
```

The main thing I kept throughout the project was that optimization
should not reduce correctness. Cheaper localization and acquisition
paths are useful, but when they are uncertain the pipeline can still
fall back to the more reliable path.
