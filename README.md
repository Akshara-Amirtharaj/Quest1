# Dialogue Frame Finder — Quest1

Quest1 finds the **first valid video frame where a target line is spoken**.
Give it a public video URL and the dialogue you remember; it returns the spoken
time range, a decoded frame, the matched text, and confidence with evidence
provenance.

The pipeline runs locally and does not require a paid API or API key. It uses
platform captions when available, verifies them against the audio, and falls
back to speech recognition when captions cannot be used.

---

## Key Features

- **Caption-first localization** — manual English captions are preferred, then
  automatic captions; a caption candidate is accepted only after bounded ASR
  verification.
- **Reliable ASR fallback** — when captions are missing, unusable, rate-limited,
  or do not verify, `faster-whisper` searches the spoken audio.
- **Robust dialogue matching** — Unicode, punctuation, common numeric and
  currency forms are normalized before chronological exact/fuzzy word-window
  matching.
- **Bounded precision retry** — uncertain English caption-window results can
  lazily retry with `distil-large-v3`; unrelated low-score audio does not trigger
  the larger model.
- **Correct frame timing** — PyAV seeks to a preceding keyframe and decodes
  forward using `PTS × time_base`, avoiding inaccurate timestamp-by-FPS math on
  variable-frame-rate media.
- **Auditable results** — output includes start/end timestamps, frame PTS and
  time base, match score, confidence, evidence sources, model provenance, and
  cache information.
- **Integrated web demo** — one FastAPI process serves both the API and the
  bundled HTML/CSS/JavaScript frontend; no Node.js build is required.

---

## Architecture

```text
Video URL + target dialogue
            │
            ▼
Metadata, stream and caption inspection
            │
            ▼
Usable caption candidate?
     ┌──────┴──────┐
    yes            no
     │              │
     ▼              ▼
Bounded audio     Audio-first acquisition
ASR verification Full-audio ASR
     │              │
     └──────┬───────┘
            ▼
Exact/fuzzy matching in chronological order
            │
            ▼
First valid spoken occurrence
            │
            ▼
PyAV seek + decode by PTS × time_base
            │
            ▼
Timestamp + frame + confidence/provenance
```

If audio-only localization succeeds but remote frame access fails, Quest1
downloads the full media and resolves the frame locally. The `quest1` CLI can
also perform bounded visible-text OCR; the spoken-dialogue web route uses
`run_v2` and intentionally does not load PaddleOCR.

---

## Demo

[Watch the Quest1 demo on Google Drive](https://drive.google.com/file/d/1l6q4ttKL1qBKSFGXUT6mr04WbLazZsit/view?usp=sharing)

---

## Example

Public sample used during project validation:

```text
Video:   https://www.youtube.com/watch?v=R6MlUcmOul8
Dialogue: You're a jerk, Thom.
```

Run the spoken-dialogue pipeline:

```powershell
quest1-v2 "https://www.youtube.com/watch?v=R6MlUcmOul8" "You're a jerk, Thom." --language en
```

The same URL and dialogue can be entered in the web interface. Live provider
availability varies, so a run may use caption localization or the documented
ASR fallback.

---

## Prerequisites

- Python 3.10 or newer
- FFmpeg and ffprobe available on `PATH`
- Internet access for public media and first-run model downloads

Verify FFmpeg before installation:

```powershell
ffmpeg -version
ffprobe -version
```

---

## Installation

```powershell
git clone https://github.com/Akshara-Amirtharaj/Quest1.git
cd Quest1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The default installation includes the production CLI, FastAPI demo, ASR, and
OCR dependencies. Optional WhisperX alignment is installed separately:

```powershell
python -m pip install -e ".[precision]"
```

---

## Running the Application

### Web frontend and API

```powershell
quest1-api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The frontend and backend
run together in this process. Runtime media, models, transcripts, and generated
frames are stored under `.cache/demo` by default and are ignored by Git.

### CLI

Final CLI with optional bounded OCR refinement:

```powershell
quest1 "PUBLIC_VIDEO_URL" "target dialogue" --language en
```

Spoken-dialogue path without OCR:

```powershell
quest1-v2 "PUBLIC_VIDEO_URL" "target dialogue" --language en
```

Earlier versioned entry points (`quest1-v0`, `quest1-v1`, and `quest1-v3`) are
retained for development-history verification.

---

## Output

A successful result reports:

- dialogue start and end in seconds and `HH:MM:SS.mmm`;
- resolved frame timestamp, frame index, PTS, and time base;
- matched source text, exact/fuzzy type, and score;
- caption, ASR, and optional OCR evidence;
- confidence category and explanation;
- ASR model, precision-fallback, alignment, and cache provenance;
- generated PNG path in the CLI or an opaque frame URL in the API.

The web API returns stable public error codes for invalid input, unavailable
media, dialogue not found, and unexpected processing failures. Provider details
and local filesystem paths are not exposed to the browser.

---

## Confidence and Evidence

Confidence is based on **independent evidence**, not only the numeric match
score. For example, an exact ASR match without caption or OCR corroboration is
reported as medium confidence, while agreeing caption and ASR evidence can
produce high confidence. Conflicting or weak evidence is reported explicitly.

The result also states how it was localized and verified, such as:

```text
localization_source: caption
verification_source: asr
frame_match_type: spoken_dialogue
```

---

## Project Structure

```text
Quest1/
├── src/
│   ├── dialogue_locator/
│   │   ├── acquisition.py          # Metadata, media and audio-first acquisition
│   │   ├── captions.py             # Caption inventory normalization
│   │   ├── subtitles.py            # Track selection, download and parsing
│   │   ├── caption_matching.py      # Caption-window candidate search
│   │   ├── caption_verification.py  # Bounded ASR verification
│   │   ├── transcription.py         # faster-whisper and optional alignment
│   │   ├── precision.py             # Lazy precision-ASR fallback
│   │   ├── matching.py              # Exact/fuzzy chronological matching
│   │   ├── frames.py                # PTS-based PyAV frame resolution
│   │   ├── ocr.py                   # Optional bounded visible-text refinement
│   │   ├── cache.py                 # Metadata, transcript and OCR caches
│   │   ├── confidence.py            # Evidence-based confidence
│   │   ├── errors.py                # Structured failure classification
│   │   ├── pipeline.py              # V0–V4 orchestration
│   │   └── cli.py                   # Versioned command-line entry points
│   └── dialogue_locator_demo/
│       ├── api.py                   # Thin FastAPI adapter
│       └── static/                  # Bundled frontend
├── tests/                           # Deterministic production/API regression tests
├── APPROACH.md                      # Incremental architecture and decisions
├── DEMO.md                          # Demo/API usage notes
├── AI_USAGE.md                      # AI-assistance disclosure
├── prompts.md                       # Concise prompt-driven development history
└── pyproject.toml                   # Package, dependencies and entry points
```

---

## Tests

```powershell
python -m pytest -q
python -m compileall -q src tests
git diff --check
```

The final production suite covers acquisition, caption selection and fallback,
ASR verification, matching edge cases, precision fallback, caching, structured
errors, PTS frame timing, OCR fallback behavior, and frontend/API integration.

---

## Limitations

- Only publicly reachable URLs or direct media URLs can be processed without
  cookies; login-protected, DRM-protected, removed, or region-blocked media may
  be unavailable.
- Provider caption endpoints can return expired data, HTTP 429, timeouts, empty
  tracks, or unsupported formats. Quest1 falls back to ASR when possible, but
  provider/network failures can still prevent acquisition.
- CPU ASR on long videos without usable captions can take several minutes.
- The default web demo searches English dialogue. Other supported language
  choices are available through the CLI with a compatible ASR model.
- OCR is bounded around an already localized spoken interval; it is not a
  whole-video text-search engine.
- Chunked ASR, VAD, KWS, and lightweight localizer work was benchmarked
  experimentally and is not connected to the final production pipeline.

---

## Documentation

- [APPROACH.md](APPROACH.md) — pipeline evolution and design decisions
- [DEMO.md](DEMO.md) — frontend/API setup and behavior
- [AI_USAGE.md](AI_USAGE.md) — AI-assistance disclosure
- [prompts.md](prompts.md) — concise development history mapped to pipeline work
