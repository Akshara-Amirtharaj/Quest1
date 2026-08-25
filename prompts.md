# AI and Coding-Assistant Prompts

This file contains only prompts from the Codex task titled **“Implement V0 media localization.”** Prompts from the later manual-verification task are intentionally excluded.

Prompts are preserved verbatim where captured. The initial V0 prompt was recovered from the original user-pasted attachment associated with that task rather than reconstructed from memory.

## Initial architecture and V0 implementation prompt

```text
I am building a software engineering assessment project for Quest1.

You currently have no context about the project, so read this carefully before making any changes.

Do not implement the entire application immediately.

First understand the complete problem and architecture, then implement only V0 as described at the end.

# Problem

Input:

* a publicly accessible video URL
* a target dialogue string

The application must automatically find where that dialogue occurs in the media and return at minimum:

* timestamp
* frame number where applicable
* extracted or detected dialogue text
* corresponding video frame as an image

The supplied assessment example uses:

"My mind rebels at stagnation"

but the implementation must not hardcode this dialogue, video URL, timestamp, FPS, duration, resolution, language or video provider.

During evaluation, the interviewer may provide a completely different video URL and dialogue.

The user or interviewer must not manually inspect the video to locate the dialogue.

# Important interpretation

This is primarily a media localization problem.

The dialogue may be represented through different signals:

* platform captions
* embedded subtitle tracks
* auto generated captions
* spoken audio
* burned in visible captions
* multiple signals at once

The application should use these signals intelligently.

The goal is always correctness first, then optimization.

# Evidence model

Treat the available sources differently because they answer different questions.

## Captions and subtitles

Captions are cheap temporal evidence.

They can help us quickly determine where a dialogue might occur.

However, they are not guaranteed to match the actual speech.

They may be:

* incorrect
* auto generated
* paraphrased
* translated
* incomplete
* mistimed
* split differently
* in the wrong language

Therefore captions should not automatically be treated as ground truth for spoken dialogue.

## Audio and ASR

ASR provides evidence about what is actually spoken.

For spoken dialogue, audio verification should ultimately establish whether the target phrase is present.

## OCR

OCR provides evidence about what is actually visible in video pixels.

This is useful when dialogue appears as burned in text or when the requirement is specifically about visible dialogue.

## Video frame timestamps

Decoded frame presentation timestamps should be treated as the authoritative timing information for the video frames.

Do not rely only on:

frame = timestamp × FPS

because videos may use variable frame rate.

# Final intended strategy

The architecture should use progressive evidence escalation.

The general decision flow is:

video URL + target dialogue

→ acquire media

→ inspect available streams and metadata

→ inspect available captions and subtitles

→ normalize target dialogue

→ if useful captions exist:
search captions using exact or normalized matching
then fuzzy matching if required

→ if caption candidate exists:
treat it as a candidate region only
extract a small audio window around that region
run ASR on that small audio window
verify whether the actual spoken dialogue matches

→ if caption verification fails:
try the next chronological caption candidate

→ if all caption candidates fail:
run ASR on the full audio

→ if captions do not exist:
directly use full audio ASR

→ locate the target phrase in the timestamped ASR output

→ determine the estimated start time of the target dialogue

→ resolve that timestamp to the actual video frame using decoded frame PTS

→ extract and save the frame

→ return structured output

# Why captions are still useful

Do not always transcribe the entire video if captions already tell us where to look.

Example:

A 60 minute video contains a caption candidate around:

42:15 → 42:18

Instead of transcribing the entire 60 minute audio:

* extract perhaps 42:12 → 42:21
* run ASR only on that short audio region
* verify the target dialogue

If verification succeeds, continue with frame resolution.

This gives us correctness without unnecessary full video ASR.

# Spoken dialogue definition

If the target dialogue exists only in audio, define the final frame as:

The earliest decoded video frame whose presentation timestamp is greater than or equal to the estimated start timestamp of the first word of the matched spoken dialogue.

Important:

The video frame timing may be precise, but the ASR speech onset is still an estimate.

Do not pretend ASR timestamps are perfect ground truth.

# Visible dialogue definition

If the target dialogue is visually present in the video, define the final frame as:

The earliest decoded frame in which the complete target dialogue satisfies the accepted OCR match condition.

ASR or subtitles may help locate the candidate interval.

OCR should establish when visible text actually appears.

# Target dialogue matching

The supplied target may not exactly equal ASR, subtitle or OCR output.

Handle cases such as:

* capitalization
* punctuation
* whitespace
* contractions
* minor ASR errors
* minor OCR errors
* dialogue split across chunks
* target text itself being slightly inaccurate

Use simple reliable matching first.

Intended later strategy:

1. Unicode and text normalization
2. exact match
3. substring match
4. fuzzy matching using RapidFuzz
5. small sliding windows over neighbouring words or subtitle chunks

Do not use embeddings or LLM based semantic matching unless later testing proves it necessary.

# Intended technology stack

Use mature existing tools wherever possible.

Do not reimplement solved problems.

Primary choices:

* Python
* yt-dlp for media acquisition and platform metadata
* FFmpeg for audio and media processing
* ffprobe for stream inspection
* PyAV for video decoding, seeking and PTS based frame handling
* faster-whisper for ASR
* WhisperX only as an optional precision fallback later
* RapidFuzz for fuzzy text matching
* PaddleOCR later for visible text verification
* pytest for tests

Possible subtitle parsing library:

* pysubs2 or another lightweight mature parser if needed

Do not introduce a dependency unless it provides clear value.

# ASR strategy

Default later ASR:

faster-whisper

Use it because it provides:

* strong transcription
* local execution
* word timestamps
* good CPU and GPU performance
* simpler installation than a heavier alignment pipeline

WhisperX should not be mandatory.

Use WhisperX only if later testing shows that higher precision first word alignment is genuinely required.

# Audio optimization

When ASR is required, prefer speech appropriate audio such as:

* mono
* approximately 16 kHz
* model compatible format

Do not preserve unnecessarily high bitrate stereo audio just for speech recognition.

During early development, temporary WAV files are acceptable because they are easy to inspect and debug.

Later, piping audio directly may be considered if it materially improves performance.

# OCR strategy

Do not implement OCR yet.

Later use a pretrained OCR system such as PaddleOCR.

Do not:

* train an OCR model
* OCR every frame
* assume subtitles always appear at the bottom
* use a vision LLM for every frame

OCR should normally operate only near an already localized candidate interval.

If no other temporal evidence exists, use coarse to fine frame sampling.

# Frame handling

Use PyAV as the authoritative frame timing layer.

Use:

* frame PTS
* stream time base
* decoded frame presentation timestamp

Do not assume constant frame rate.

Do not decode the entire video from the beginning when a candidate timestamp is already known.

Later, seek close to the candidate region and decode only a small interval.

# Media quality optimization

Do not automatically download the highest available video resolution.

Choose a practical media format that balances:

* sufficient visual quality
* sufficient audio quality
* download size
* decode cost
* later OCR requirements

Prefer approximately 720p when appropriate.

Do not hardcode 720p as a requirement.

If 720p is unavailable, select an appropriate fallback.

If later OCR requires better visual quality, higher quality may be selected adaptively.

# Optimization principle

At every stage ask:

* can this work be avoided
* can a cheaper signal narrow the search first
* can we process a smaller time window
* can we seek instead of decoding sequentially
* can we reuse an existing artifact
* can a mature library solve this already
* can a heavy model be loaded only when required

Optimize primarily by eliminating unnecessary work.

Do not prematurely micro optimize Python code.

# No database

Do not add a database.

The current application is:

video URL + dialogue
→ process
→ result

It is stateless.

Temporary local artifacts are sufficient.

Possible directories:

.cache/
output/

A database would only become justified if future requirements include:

* users
* processing history
* persistent jobs
* analytics
* shared caching
* distributed workers

Do not add:

* PostgreSQL
* MongoDB
* Redis

# Do not add unnecessary architecture

Do not introduce:

* microservices
* message queues
* Kubernetes
* cloud services
* distributed workers
* LLM agents
* embeddings
* vector databases
* custom OCR models
* custom ASR models
* custom edit distance implementations

This should remain a small modular Python application.

# Intended internal architecture

The final application may eventually contain responsibilities such as:

media acquisition
media inspection
subtitle processing
audio extraction
ASR
OCR
text normalization
dialogue matching
candidate selection
frame resolution
result formatting

Do not turn all of these into classes automatically.

Use functions and dataclasses where sufficient.

Introduce interfaces only where replaceability genuinely helps.

# Common candidate representation

Later evidence providers should ideally produce a common structure similar to:

Candidate

* text
* start_time
* end_time
* source
* score
* verified
* verification_source

Do not overdesign this yet.

Final Result may eventually contain:

* matched_text
* timestamp
* frame_index
* frame_path
* confidence
* sources

# Confidence strategy

Do not invent arbitrary mathematical confidence formulas.

Prefer simple categories such as:

HIGH
MEDIUM
LOW

and preserve underlying raw scores separately.

Examples:

HIGH
caption candidate + strong ASR verification

MEDIUM
strong ASR only

LOW
weak ASR or conflicting evidence

Do not implement this fully yet.

# Important failure cases for later

The architecture should eventually tolerate:

* invalid URL
* inaccessible media
* expired URL
* missing video stream
* missing audio
* no captions
* incorrect captions
* wrong subtitle language
* multiple subtitle tracks
* commentary or SDH subtitles
* background music
* multiple speakers
* poor audio quality
* repeated dialogue
* inaccurate target dialogue
* variable frame rate
* non zero media timestamps
* no match
* multiple possible matches
* low confidence results

Do not attempt to solve every one of these in V0.

# Development plan

We will build incrementally.

## V0 — media foundation

Goal:

Prove that we can reliably acquire and inspect media and decode a frame.

## V1 — first complete working solution

media
→ full audio ASR using faster-whisper
→ target dialogue matching
→ candidate speech timestamp
→ PyAV frame resolution
→ frame image
→ required output

This gives us the first complete working application.

## V2 — caption optimization

caption discovery
→ caption search
→ candidate interval
→ short-window ASR verification
→ full ASR fallback

This reduces unnecessary transcription.

## V3 — visible text support

candidate interval
→ local video decoding
→ PaddleOCR
→ first complete visible match

Also add coarse OCR fallback if needed.

## V4 — hardening

Possible additions:

* WhisperX alignment
* confidence handling
* lightweight caching
* adaptive resolution
* stronger edge case handling
* multiple occurrence support
* multilingual handling

Do not implement V1, V2, V3 or V4 now.

# CURRENT TASK

Implement only V0.

Do not proceed to later milestones automatically.

# V0 requirements

## 1. Project structure

Create a small Python project structure suitable for later expansion.

Keep it clean and minimal.

Example direction only:

src/
main.py
media/
acquisition.py
inspection.py
frames.py
models/
media_info.py

tests/
output/
README.md
requirements.txt
prompts.md

You may improve this structure if there is a simpler sensible option.

Do not overengineer it.

## 2. CLI input

Accept a publicly accessible video URL.

A CLI is sufficient.

Do not build a frontend or API.

## 3. Media acquisition

Use yt-dlp.

The media acquisition layer should:

* obtain useful platform metadata
* select practical media quality
* avoid automatically selecting maximum resolution
* prefer around 720p when appropriate
* gracefully fall back
* retain good enough audio for future ASR
* avoid duplicate downloads during one execution

Use a sensible yt-dlp format selection expression rather than custom format-selection algorithms unless necessary.

## 4. Media inspection

Use ffprobe.

Use structured JSON output.

Extract at minimum:

* duration
* video stream exists
* audio stream exists
* subtitle stream exists
* width
* height
* video codec
* audio codec
* average frame rate metadata
* real frame rate metadata if available
* stream time base
* stream start time where available

Represent missing metadata safely.

## 5. Caption discovery

Use yt-dlp metadata where possible to determine:

* manually available subtitles
* automatic captions
* available languages
* useful track metadata

Do not match captions yet.

Do not run ASR.

Do not run OCR.

## 6. Decode sample frame

Use PyAV.

Open the acquired video and decode at least one sample video frame.

Capture:

* sequential decoded frame index
* frame PTS
* frame time base
* presentation timestamp

Do not use frame index × FPS as the authoritative timestamp.

## 7. Save sample frame

Save the decoded sample frame to the output directory.

Use PNG or JPEG.

The goal is simply to prove that video decoding and image extraction work.

## 8. Structured result

Print or return a structured JSON style V0 result containing at minimum:

* source_url
* media_path
* duration
* has_video
* has_audio
* embedded_subtitles
* platform_subtitles
* automatic_captions
* available_caption_languages
* width
* height
* video_codec
* audio_codec
* avg_frame_rate
* real_frame_rate
* video_time_base
* video_start_time
* sample_frame_index
* sample_frame_pts
* sample_frame_timestamp
* sample_frame_path

Keep the representation understandable.

## 9. Dependency checks

Before processing, check important external dependencies such as:

* ffmpeg
* ffprobe

Provide a clear error if they are unavailable.

Do not silently fail.

## 10. Error handling

Handle cleanly:

* invalid URL
* yt-dlp failure
* unsupported source
* network failure
* ffprobe failure
* missing video stream
* PyAV failure
* frame decode failure
* output directory failure

Use clear errors.

Do not create an excessive custom exception hierarchy.

## 11. Tests

Add meaningful tests for deterministic V0 logic.

Tests should not all require a network connection.

Consider unit tests for:

* ffprobe JSON parsing
* missing audio/video fields
* rational frame rate parsing
* timestamp conversion
* caption metadata normalization
* missing metadata

If appropriate, include a separate optional integration test for a real video URL.

Do not make the normal test suite download a large video.

## 12. README

Document:

* actual project problem
* architecture direction
* what V0 implements
* prerequisites
* Python setup
* yt-dlp requirement
* FFmpeg/ffprobe requirement
* how to run V0
* expected output
* output directories
* current limitations
* V1 to V4 overview

Keep the README concise but sufficient for another engineer to run the project.

## 13. Prompt documentation

Quest1 explicitly requires the prompts used with LLMs/coding assistants to be documented in the GitHub repository.

Create:

prompts.md

Do not fabricate prompts that were used before this.

Add a clear section for:

* architecture/design prompts
* Codex implementation prompts

You may either include this exact Codex prompt or create a clearly marked section where I can paste the exact original prompt manually.

Preserve prompts verbatim if they are added.

## 14. Verification

After implementation:

* run the unit tests
* run the CLI
* if network access is available, test against the supplied sample video
* verify yt-dlp acquisition
* verify ffprobe output
* verify caption discovery
* verify PyAV frame decoding
* verify the saved frame exists
* inspect whether timing fields are populated correctly

Do not claim a test succeeded if you could not execute it.

Clearly mention environmental restrictions.

## 15. Before modifying code

First inspect the current repository.

If files already exist:

* understand the current structure
* preserve working code
* avoid unnecessary rewrites
* do not delete unrelated files
* adapt the implementation to what is already present

Do not assume the repository is empty.

## 16. Coding quality

Keep the code:

* readable
* typed where useful
* modular
* testable
* easy to explain in an interview

Avoid:

* giant functions
* excessive classes
* clever abstractions
* unnecessary dependencies
* premature optimization

Add comments only where the reasoning is not obvious.

## 17. Final response

After V0 is fully implemented, report:

1. files created
2. files modified
3. V0 architecture
4. dependencies added
5. commands required to run it
6. tests executed
7. test results
8. sample execution result if available
9. any source/media problems encountered
10. assumptions made
11. limitations remaining
12. what V1 should implement next

Most importantly:

Do not proceed to V1.

Stop after V0 is implemented, tested and verified.
```

## Subsequent implementation prompts from the same task

### 2026-08-24 - Codex - V0 direct HTTP acquisition fallback

> We are getting an SSL certificate error from yt-dlp for the given video URL.
>
> Before changing the architecture, check whether using requests can help with media acquisition.
>
> For a direct video URL, something like:
>
> import requests
>
> url = "[https://example.com/interview.mp4](https://example.com/interview.mp4)"
>
> response = requests.get(url, stream=True)
>
> with open("interview\.mp4", "wb") as f:
> for chunk in response.iter\_content(chunk\_size=1024 \* 1024):
> if chunk:
> f.write(chunk)
>
> print("Video downloaded")
>
> Check whether this approach is useful for our V0 acquisition layer.
>
> Important:
>
> - input should accept any valid public video URL, not only .mp4 URLs
> - do not hardcode any provider or URL format
> - keep yt-dlp if it is needed for provider/page URLs
> - requests can be used as a fallback if appropriate
> - do not disable SSL verification globally
> - make the smallest change required
> - test the current failing URL after the change
>
> Do not start V1 yet.
>
> First explain briefly whether this approach will actually help, then make the required V0 change and test it.

### 2026-08-24 - Codex - V1 spoken-audio localization

> V0 is complete and tested. Now implement V1.
>
> Goal:
> Given a video URL and a target dialogue, find the first occurrence of that dialogue in the spoken audio and extract the corresponding video frame.
>
> For V1, ignore captions and OCR. We first need a reliable end-to-end audio-based solution.
>
> Use the existing V0 acquisition, ffprobe and PyAV code. Do not rewrite working V0 functionality.
>
> Requirements:
>
> 1. Update the CLI to accept:
>    - video URL
>    - target dialogue
>
> Example:
> quest1 "[https://ok.ru/video/248244667877](https://ok.ru/video/248244667877) " "My mind rebels at stagnation"
>
> 2. Extract audio from the acquired media using FFmpeg in a format suitable for speech recognition.
>    Prefer mono 16 kHz audio unless faster-whisper works better directly with the existing media.
>
> 3. Use faster-whisper for speech recognition.
>    Use an existing pretrained Whisper model. Do not build or train any speech model.
>
> Because I have limited disk space, choose a reasonably small model initially that still gives good English transcription accuracy.
> Explain which model you choose and why.
>
> 4. Generate timestamped transcription.
>
> Enable word-level timestamps if practical because we eventually need the start time of the first word of the target dialogue.
>
> 5. Implement dialogue matching.
>
> Normalize both the requested dialogue and transcript for:
>
> - case
> - punctuation
> - whitespace
>
> First try exact/normalized matching.
>
> If that fails, use RapidFuzz for minor ASR differences.
>
> The target dialogue may span multiple ASR segments, so do not assume it exists inside one segment.
>
> Do not use an LLM or embeddings for matching.
>
> 6. Find the FIRST valid occurrence of the target dialogue.
>
> If multiple occurrences exist, return the earliest one.
>
> 7. Once the dialogue is found, determine the estimated start timestamp of its first spoken word.
>
> Do not calculate the final frame using timestamp × FPS.
>
> 8. Use the existing PyAV functionality to seek near that timestamp and decode frames using actual PTS/time\_base.
>
> Return the earliest decoded frame whose presentation timestamp is >= the estimated dialogue start timestamp.
>
> 9. Save that frame to output.
>
> 10. Return structured output containing at least:
>
> {
> "query": "...",
> "matched\_text": "...",
> "dialogue\_start": ...,
> "dialogue\_end": ...,
> "frame\_index": ...,
> "frame\_pts": ...,
> "frame\_timestamp": ...,
> "frame\_path": "...",
> "match\_type": "exact/fuzzy"
> }
>
> The matched\_text should come from the transcription rather than simply echoing the input query.
>
> 11. Handle:
>
> - no audio stream
> - ASR failure
> - dialogue not found
> - minor transcription differences
> - dialogue spanning ASR segments
> - multiple occurrences
>
> 12. Keep the implementation modular and simple.
>
> Do not add:
>
> - captions/subtitle matching
> - OCR
> - WhisperX
> - database
> - frontend
> - API
> - embeddings/LLMs
> - unnecessary abstractions
>
> 13. Add tests for dialogue normalization and matching without requiring Whisper to run for every unit test.
>
> 14. Test V1 end to end on a small public video where the spoken dialogue is known.
>
> Also run all existing V0 tests and make sure nothing regresses.
>
> 15. Be careful about disk usage.
>     Do not download multiple Whisper models.
>     Do not leave unnecessary temporary audio/video files.
>     Reuse the media already acquired during the pipeline where possible.
>
> Before coding, briefly inspect the current V0 structure and tell me your implementation plan.
>
> Then implement V1, run the tests and report:
>
> - files changed
> - model chosen and model size
> - how matching works
> - how word timestamps map to the final frame
> - tests/results
> - end-to-end result
> - any problems or assumptions
>
> Do not implement V2 yet.

### 2026-08-24 - Codex - V2 caption-first localization

> V0 and V1 are complete.
>
> Implement only V2 now.
>
> Goal:
> Use captions/subtitles, when available, to reduce how much audio needs to be transcribed while preserving correctness.
>
> Important rule:
>
> Captions are not ground truth for spoken dialogue.
>
> Use them only to locate candidate time ranges.
>
> Verify promising caption candidates against the actual audio using ASR.
>
> If captions are unavailable, unsuitable, or fail verification, fall back to the existing V1 full-audio ASR path.
>
> Do not implement OCR or WhisperX yet.
>
> ## Requirements
>
> 1. Inspect the existing V0 and V1 code first.
>
> Preserve all working functionality.
>
> Do not rewrite working modules unless there is a clear design problem.
>
> 2. Keep the project modular and follow normal Python project conventions.
>
> Prefer:
>
> * small focused modules
> * clear type hints
> * dataclasses only where useful
> * dependency injection only where it genuinely helps testing
> * standard library solutions where sufficient
> * existing mature libraries instead of custom parsers or algorithms
>
> Avoid:
>
> * giant service classes
> * unnecessary factories/interfaces
> * duplicated logic
> * provider-specific hardcoding
> * hardcoded timestamps, languages, FPS, URLs or subtitle positions
>
> 3. Subtitle/caption discovery
>
> Reuse the existing yt-dlp metadata from V0.
>
> Detect available:
>
> * manual subtitles
> * automatic captions
> * languages
> * track metadata where available
>
> Choose suitable tracks based on the target dialogue language when metadata is available.
>
> Do not assume English permanently.
>
> If language cannot be determined reliably, handle it conservatively rather than hardcoding one track.
>
> 4. Subtitle parsing
>
> Use an existing lightweight subtitle parsing library if needed.
>
> Do not manually implement full SRT/VTT/ASS parsers.
>
> Support common formats exposed by yt-dlp where practical.
>
> Normalize parsed entries into a common internal representation such as:
>
> SubtitleEntry:
>
> * text
> * start_time
> * end_time
>
> Keep this representation source-independent.
>
> 5. Caption matching
>
> Reuse the normalization and matching logic already built in V1.
>
> Do not duplicate matching code.
>
> Search captions in this order:
>
> * normalized exact match
> * substring match
> * fuzzy match with RapidFuzz
>
> Handle dialogue split across neighbouring subtitle entries using a small sliding window.
>
> Do not hardcode one fixed number of entries if a simple configurable limit is cleaner.
>
> 6. Candidate generation
>
> Convert caption matches into timestamped candidate intervals.
>
> Candidates should be processed chronologically because the requirement is to return the first valid occurrence.
>
> Do not verify all candidates if the earliest one already succeeds.
>
> 7. Short-window ASR verification
>
> For each promising caption candidate:
>
> * add a small configurable margin before and after the candidate
> * extract/process only that audio window
> * run the existing faster-whisper pipeline on that small region
> * verify whether the target dialogue is actually spoken there
>
> Do not transcribe the whole video if a short-window verification succeeds.
>
> Keep the margin configurable rather than hardcoding it deeply in business logic.
>
> Choose sensible defaults and document them.
>
> 8. Progressive widening
>
> If the initial short verification window fails, do not immediately run full-video ASR.
>
> Use a simple progressive widening strategy, for example:
>
> small margin
> → larger margin
> → full-audio ASR fallback
>
> Keep the number of widening steps small and configurable.
>
> Avoid overengineering this.
>
> 9. Fallback behaviour
>
> Use the existing V1 path when:
>
> * no captions exist
> * no suitable-language captions exist
> * captions cannot be parsed
> * no caption candidate matches
> * all caption candidates fail ASR verification
> * caption metadata is unusable
>
> V1 must remain a reliable fallback.
>
> Do not duplicate the V1 implementation.
>
> 10. Matching result
>
> When ASR verifies the dialogue, the final matched text should come from the ASR transcript, not simply from the caption text or input query.
>
> Return useful source information such as:
>
> * localization_source: caption
> * verification_source: asr
>
> If V1 fallback was used:
>
> * localization_source: asr
> * verification_source: asr
>
> Keep this simple.
>
> 11. Frame resolution
>
> Reuse the existing V1 PyAV frame-resolution logic.
>
> Do not change the timing model.
>
> Continue using:
>
> * frame PTS
> * stream time base
> * decoded frame timestamp
>
> Do not use timestamp × FPS as authoritative timing.
>
> 12. Optimization
>
> At every stage minimize unnecessary work.
>
> Prefer:
>
> * caption lookup before full ASR
> * chronological early stopping
> * short audio windows
> * reuse of downloaded media
> * reuse of existing metadata
> * lazy loading of ASR if possible
> * no repeated ffprobe or media acquisition
> * no duplicate transcript generation
>
> Avoid:
>
> * running ASR and caption search in parallel unnecessarily
> * transcribing the whole file if short-window verification succeeds
> * loading models before they are needed
> * storing large temporary files unless required for debugging
>
> 13. Existing solutions first
>
> Before implementing custom logic, check whether existing libraries already provide the needed capability.
>
> Use mature existing tools for:
>
> * subtitle parsing
> * fuzzy matching
> * media extraction
> * audio decoding
>
> Do not implement:
>
> * custom subtitle parser
> * custom edit distance
> * custom speech recognition
> * custom media decoder
>
> 14. Configuration
>
> Do not scatter magic numbers through the code.
>
> Parameters such as:
>
> * fuzzy threshold
> * verification margin
> * widening margin
> * subtitle window size
>
> should live in a small configuration object/module or CLI options where appropriate.
>
> Defaults should be sensible but overridable.
>
> Do not introduce a large configuration framework.
>
> 15. Error handling
>
> Handle cleanly:
>
> * malformed subtitle track
> * unsupported subtitle format
> * empty subtitle track
> * subtitle download failure
> * no matching candidate
> * short-window ASR failure
> * all candidates rejected
>
> Do not let an optional caption path break the working V1 fallback unless the underlying media itself is unusable.
>
> 16. Tests
>
> Add focused unit tests for:
>
> * subtitle normalization
> * adjacent subtitle-window matching
> * exact caption candidate
> * fuzzy caption candidate
> * chronological candidate ordering
> * progressive widening
> * fallback to V1 when captions fail
> * no duplicate ASR invocation after successful verification
>
> Mock ASR and network/media calls where possible.
>
> Do not make the normal test suite depend on large media downloads or Whisper inference.
>
> 17. Integration test
>
> Run at least one end-to-end V2 test on a small public video with captions if practical.
>
> Verify:
>
> * captions are discovered
> * candidate interval is found
> * only a short audio region is sent to ASR
> * ASR verifies the dialogue
> * final frame is resolved using existing V1 logic
> * final output is correct
>
> Also rerun all V0 and V1 tests.
>
> 18. Code quality and structure
>
> Keep the folder structure clean and industry-standard.
>
> If new modules are needed, prefer responsibilities like:
>
> src/dialogue_locator/
> subtitles/
> parser.py
> matching.py
> localization/
> caption_locator.py
>
> Only create folders/modules if they improve separation of concerns.
>
> Do not restructure the whole repository just for V2.
>
> Use:
>
> * pathlib
> * logging instead of scattered print statements where appropriate
> * enums only where useful
> * explicit return types
> * clear names
> * small functions
> * minimal side effects
>
> 19. Documentation
>
> Update:
>
> * README with V2 behaviour
> * architecture flow
> * caption-first optimization
> * fallback to V1
> * limitations
> * configuration options
>
> Update prompts.md with this V2 prompt.
>
> 20. Final report
>
> After implementation, report:
>
> * files changed
> * dependencies added
> * caption parsing approach
> * candidate matching strategy
> * verification window strategy
> * how fallback to V1 works
> * how much audio was processed in the V2 integration test compared with full-video ASR
> * tests run and results
> * any assumptions or limitations
> * anything that should be fixed before V3
>
> Do not implement V3.

### 2026-08-24 - Codex - V2 caption acquisition hardening

> V2 fallback is working, but my manual test did not actually use captions.
>
> For the YouTube test video, the manual VTT track returned HTTP 429 and the automatic VTT track could not be parsed, so the pipeline fell back to full ASR.
>
> Please harden only the caption acquisition part of V2.
>
> If one subtitle track or format fails, try other available formats for the same suitable language before giving up.
>
> Use yt-dlp's available subtitle metadata rather than hardcoding YouTube-specific formats.
>
> Keep the current priority:
> manual captions first
> → automatic captions
> → full ASR fallback
>
> Do not break the existing fallback behavior.
>
> Also make sure:
>
> - a failed VTT does not immediately discard the whole language
> - unusable/HTML/invalid subtitle responses are rejected
> - 429 or expired caption URLs are handled cleanly
> - successful caption retrieval still feeds the existing caption matching and short-window ASR verification
>
> Make the smallest change required.
>
> Then test again on the same video and report whether:
>
> - localization\_source becomes caption
> - verification\_source remains asr
> - audio\_processed\_seconds becomes much lower than the full 19 seconds
>
> Do not start V3.

### 2026-08-24 - Codex - V3 visible-text frame localization

> V2 is complete. Now implement V3.
>
> Goal:
> Support cases where the target dialogue is visibly present in the video as burned-in text/captions and return the first frame where the complete target dialogue is visible.
>
> Important:
> Do not OCR the whole video by default.
>
> Use the existing V1/V2 localization first:
>
> - captions if available
> - otherwise ASR
>
> This gives us a candidate time interval.
>
> Then V3 should inspect only frames around that candidate interval.
>
> Requirements:
>
> 1. Keep all existing V0, V1 and V2 behavior working.
> Do not rewrite working code.
>
> 2. Add OCR using a strong pretrained existing model.
> Use PaddleOCR unless there is a clearly better practical option for this project.
>
> Do not train any OCR model.
>
> 3. Use PyAV to seek near the candidate interval and decode frames locally.
>
> Do not decode the entire video from the beginning.
>
> 4. Run OCR on candidate frames chronologically.
>
> The goal is to find:
>
> the earliest decoded frame where the complete target dialogue satisfies the accepted OCR match condition.
>
> 5. Reuse the existing text normalization and RapidFuzz matching logic where possible.
>
> OCR output may contain small errors, so support:
>
> - case differences
> - punctuation differences
> - spacing differences
> - minor OCR mistakes
>
> 6. Do not assume captions are always at the bottom of the frame.
>
> Initially OCR the full candidate frame if needed.
>
> If performance is poor, add a simple ROI optimization only if it is justified by testing.
>
> Do not hardcode one subtitle region for every video.
>
> 7. Stop as soon as the first valid visible match is found.
>
> If the target appears gradually across multiple frames, return the first frame where the complete dialogue is visible.
>
> 8. Keep spoken-only behavior unchanged.
>
> If no visible text is found, do not break the existing audio-based result.
>
> The system should still be able to return the spoken-dialogue frame using V1/V2.
>
> 9. Add a clear source/result distinction.
>
> For visible-text success, return something like:
>
> - localization\_source: caption or asr
> - verification\_source: ocr
> - frame\_match\_type: visible\_text
>
> For spoken-only fallback:
>
> - verification\_source: asr
>
> 10. Add tests without requiring PaddleOCR to run for every unit test.
>
> Mock OCR results where possible.
>
> Test:
>
> - exact visible match
> - fuzzy OCR match
> - dialogue appearing gradually
> - no visible match
> - first valid frame selection
> - V1/V2 regression
>
> 11. Keep performance in mind.
>
> Only load PaddleOCR when V3 actually needs it.
>
> Do not initialize it at startup.
>
> Do not OCR frames outside the candidate region.
>
> 12. Do not add:
>
> - WhisperX
> - database
> - frontend
> - API
> - embeddings
> - LLM vision
> - full-video OCR
> - unnecessary new dependencies
>
> 13. End-to-end test
>
> Use a small public video with visible captions or burned-in text.
>
> Verify that:
>
> - localization narrows the interval
> - OCR runs only near that interval
> - the first complete visible-text frame is selected
> - the frame image is saved
> - existing spoken-audio flow still works
>
> After implementation report:
>
> - files changed
> - PaddleOCR model/version used
> - how many frames were OCR-processed
> - how the first visible frame is selected
> - tests and results
> - one end-to-end example
> - limitations
>
> Do not implement V4 yet.

### 2026-08-24 - Codex - V4 hardening

> V3 is complete. Now implement V4 hardening only.
>
> Goal:
> Make the existing pipeline more robust and interview-ready without changing the core architecture.
>
> Do not rewrite working V0-V3 functionality.
>
> Focus on these improvements:
>
> 1. WhisperX precision mode
>
> - keep faster-whisper as default
> - add WhisperX only as an optional precision mode when more accurate word boundary alignment is needed
> - do not make WhisperX mandatory
> - if WhisperX is unavailable, the normal pipeline should still work
>
> 2. Confidence handling
>
> - add simple confidence categories such as HIGH, MEDIUM, LOW
> - do not invent arbitrary weighted formulas
> - preserve raw scores separately
> - confidence should consider evidence such as:
>   - exact/fuzzy match score
>   - caption + ASR agreement
>   - ASR-only match
>   - OCR verification
>   - conflicting evidence
>
> 3. Lightweight caching
>
> - reuse downloaded media
> - reuse transcripts where safe
> - reuse metadata where useful
> - avoid recomputing expensive ASR/OCR work during repeated tests
> - keep caching file-based only
> - no database or Redis
>
> 4. Multilingual support
>
> - keep English optimized path working
> - allow language selection
> - support a multilingual faster-whisper model when required
> - do not automatically download many models
> - avoid wasting disk space
>
> 5. Better edge-case handling
>    Handle cleanly:
>
> - no match
> - multiple matches
> - repeated dialogue
> - low confidence result
> - captions disagreeing with ASR
> - OCR and ASR disagreement
> - missing audio
> - missing video
> - invalid media
> - variable frame rate
> - non-zero start timestamps
> - target dialogue with small wording mistakes
>
> 6. Multiple occurrences
>
> - keep first occurrence as the default behavior
> - internally support detecting multiple candidates where practical
> - optionally expose all occurrences only if it can be added cleanly
> - do not complicate the default CLI
>
> 7. Better output
>    Return clear provenance fields such as:
>
> - localization_source
> - verification_source
> - match_type
> - match_score
> - confidence
> - confidence_reason
>
> Do not just echo the query as detected text.
>
> 8. Performance
>
> - keep heavy models lazy-loaded
> - do not run WhisperX unless needed
> - do not run OCR unless visual verification is needed
> - preserve caption-first / short-window ASR optimization
> - preserve chronological early stopping
> - clean temporary files
> - avoid duplicate model downloads
>
> 9. Tests
>    Add focused tests for:
>
> - confidence categories
> - multiple occurrences
> - low confidence
> - conflicting evidence
> - multilingual configuration
> - cache reuse
> - WhisperX optional fallback behavior
> - V0-V3 regressions
>
> 10. Documentation
>     Update README with:
>
> - final architecture
> - evidence hierarchy
> - V0-V4 milestones
> - supported inputs
> - limitations
> - confidence semantics
> - optional precision mode
> - known provider limitations
>
> 11. Do not add
>
> - database
> - Redis
> - cloud services
> - frontend
> - API
> - embeddings
> - LLM matching
> - microservices
> - queues
>
> After implementation, report:
>
> - files changed
> - V4 features added
> - how confidence works
> - how cache reuse works
> - how optional WhisperX mode works
> - multilingual behavior
> - tests/results
> - end-to-end examples
> - remaining limitations
>
> Do not add any new milestone after V4.
