# AI and Coding-Assistant Prompts

This file records prompts used to design or implement the repository. Prompts must be preserved verbatim when added; summaries must not be presented as original prompts.

## Architecture/design prompts

No standalone architecture/design prompt has been recorded in this repository yet. The current Codex implementation prompt contained substantial architecture direction, but it was one combined prompt rather than a separate design session.

If a separate design prompt is used later, paste the exact text below this line before submission.

<!-- Paste exact architecture/design prompts here, verbatim. -->

## Codex implementation prompts

The V0 implementation was created from a single long prompt supplied in the Codex task on 2026-08-24. It begins with:

> I am building a software engineering assessment project for Quest1.
>
> You currently have no context about the project, so read this carefully before making any changes.
>
> Do not implement the entire application immediately.
>
> First understand the complete problem and architecture, then implement only V0 as described at the end.

The prompt then defines the complete media-localization problem, evidence model, progressive caption/ASR/OCR strategy, V0-V4 plan, and the numbered V0 implementation requirements. This excerpt is a locator, **not** a claimed verbatim record of the complete prompt.

Before assessment submission, paste/export the complete original Codex task prompt verbatim below. It is intentionally not reconstructed from memory here, because doing so could fabricate or alter the actual prompt.

<!-- Paste the complete original Codex implementation prompt here, verbatim. -->

## Maintenance rule

For every future AI-assisted change, append the exact prompt, tool/assistant name, date, and the affected milestone. Do not rewrite old entries after the fact.

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

### 2026-08-25 - Codex - V4 verification

> v4 is implemented, can we check that

### 2026-08-25 - Codex - Cached-provider failure fix

> can you fix and run it

Context for this prompt was the immediately preceding BitChute terminal output: the first run was manually interrupted during full-audio ASR, and the next run reported the provider video unavailable even though the media file had already downloaded.

### 2026-08-25 - Codex - Pipeline completeness review

> have we implemented the below pipeline or is our existing implementation iss better and added even more better methods or optimisation techniques?? check if we're missing smth

The prompt included the proposed acquisition → metadata → caption → short-window/full-ASR → OCR/WhisperX → PTS-frame pipeline diagram.

### 2026-08-25 - Codex - Public repository checkpoint

> can we push whatever we did till now to github and then start optimizing?
> make sure our folder is in the required format
