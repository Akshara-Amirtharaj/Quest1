# AI-Assistance Disclosure

## Tools used

OpenAI Codex was used as a coding assistant during the staged V0–V4 implementation, debugging, test creation, manual verification, and documentation of this repository.

## How AI was used

AI assistance included:

- discussing and refining the progressive caption → ASR → OCR architecture;
- implementing code from user-provided milestone requirements;
- explaining errors from yt-dlp, FFmpeg, faster-whisper, PaddleOCR, and provider caption endpoints;
- proposing and implementing focused fixes after manual terminal tests;
- adding unit and regression tests;
- reviewing edge cases, performance limitations, and cache behavior;
- preparing project documentation.

The developer directed milestone scope, supplied manual test inputs, reviewed produced frames and terminal output, and decided when to advance between versions. Provider behavior and end-to-end results were checked with real media in addition to the offline suite.

## Prompt record

The recorded prompt history is maintained in [prompts.md](prompts.md). It preserves prompts verbatim where they were captured. Any explicitly marked locator or missing-original section is disclosed rather than reconstructed from memory.

## Verification responsibility

AI-generated or AI-modified code was not accepted solely from generated text. The repository includes automated tests, and major pipeline stages were manually exercised from PowerShell with public and locally served media. Known limitations remain documented in [APPROACH.md](APPROACH.md) and [README.md](README.md).
