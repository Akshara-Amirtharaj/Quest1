from __future__ import annotations

from .manifest import ConservativeVADConfig


def global_fallback_reason(
    *,
    vad_found_match: bool,
    vad_matches_baseline: bool | None,
    config: ConservativeVADConfig,
) -> str | None:
    if not vad_found_match and config.fallback_on_no_match:
        return "VAD path found no dialogue; ran unfiltered chronological chunked fallback."
    if (
        vad_found_match
        and vad_matches_baseline is False
        and config.fallback_on_baseline_mismatch
    ):
        return "VAD result disagreed with baseline; ran unfiltered chronological chunked fallback."
    return None
