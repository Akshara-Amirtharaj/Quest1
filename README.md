# Dialogue Frame Finder

Given a public video URL and target dialogue, Quest1 finds the first valid spoken occurrence and returns its timestamp and corresponding video frame. It uses captions when useful, verifies speech with ASR, and falls back to broader ASR when required.

## Final pipeline

```text
Video URL + target dialogue
        ↓
Metadata, stream and caption inspection
        ↓
Usable caption candidate?
   ├─ yes → caption localization → bounded ASR verification
   └─ no  → audio-first acquisition → full-audio ASR
        ↓
Normalized exact/fuzzy matching → first valid occurrence
        ↓
PyAV seek/decode using PTS × time_base
        ↓
Timestamp + frame + match score + confidence/provenance
```

If bounded remote frame access fails after audio-only localization, Quest1 downloads the full media and resolves the frame locally. The `quest1`/V3 path can additionally look for visible burned-in text with OCR; the bundled spoken-dialogue web UI uses V2 and does not load OCR unnecessarily.

## Setup

Prerequisites:

- Python 3.10 or newer
- FFmpeg and ffprobe available on `PATH`
- Internet access for public media and first-run ASR model downloads

```powershell
git clone https://github.com/Akshara-Amirtharaj/Quest1.git
cd Quest1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No Node.js installation or frontend build is required; FastAPI serves the bundled HTML, CSS and JavaScript.

## Run

Start the API and frontend together:

```powershell
quest1-api
```

Open `http://127.0.0.1:8000`.

The CLI is also available:

```powershell
quest1 "PUBLIC_VIDEO_URL" "target dialogue" --language en
```

For spoken-dialogue localization without optional OCR:

```powershell
quest1-v2 "PUBLIC_VIDEO_URL" "target dialogue" --language en
```

## Example

This public sample is registered in the project benchmark history:

```text
Video: https://www.youtube.com/watch?v=R6MlUcmOul8
Dialogue: You're a jerk, Thom.
```

CLI example:

```powershell
quest1-v2 "https://www.youtube.com/watch?v=R6MlUcmOul8" "You're a jerk, Thom." --language en
```

Provider availability and caption endpoints can change, so live runs may use the ASR fallback.

## Output

Successful results include:

- dialogue start/end and frame timestamps
- extracted frame path or safe API frame URL
- matched transcript text, match type and score
- localization and verification sources
- confidence, confidence reason and evidence provenance
- frame PTS and time base

Expected failures use structured error codes. The web UI maps missing/invalid input, unavailable video, dialogue-not-found and unexpected processing failures to concise public messages without exposing backend paths or provider diagnostics.

## Tests

```powershell
python -m pytest -q
python -m compileall -q src tests
```

See [APPROACH.md](APPROACH.md) for design details, [DEMO.md](DEMO.md) for the API contract, [AI_USAGE.md](AI_USAGE.md) for AI-assistance disclosure, and [prompts.md](prompts.md) for development history.
