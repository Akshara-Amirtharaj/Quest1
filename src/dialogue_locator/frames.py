from __future__ import annotations

from pathlib import Path

from .errors import V0Error
from .models import CandidateVideoFrame, ResolvedFrame, SampleFrame


def decode_sample_frame(media_path: Path, output_dir: Path) -> SampleFrame:
    try:
        import av
    except ImportError as exc:
        raise V0Error(
            "PyAV is not installed. Install project dependencies with "
            "'python -m pip install -e .'."
        ) from exc

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise V0Error(f"Could not create output directory {output_dir}: {exc}") from exc

    frame_path = output_dir / "sample_frame.png"
    try:
        with av.open(str(media_path)) as container:
            if not container.streams.video:
                raise V0Error("The acquired media has no decodable video stream.")
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                time_base = frame.time_base or stream.time_base
                timestamp = float(frame.pts * time_base) if frame.pts is not None and time_base else None
                frame.to_image().save(frame_path, format="PNG")
                return SampleFrame(
                    index=index,
                    pts=frame.pts,
                    time_base=str(time_base) if time_base else None,
                    timestamp=timestamp,
                    path=frame_path.resolve(),
                )
    except V0Error:
        raise
    except (OSError, ValueError) as exc:
        raise V0Error(f"PyAV could not decode a sample frame: {exc}") from exc
    except Exception as exc:
        # PyAV exception class names vary between supported releases.
        if exc.__class__.__module__.startswith("av"):
            raise V0Error(f"PyAV could not decode a sample frame: {exc}") from exc
        raise
    raise V0Error("The video stream opened, but no frame could be decoded.")


def resolve_frame_at_timestamp(
    media_path: Path,
    target_timestamp: float,
    output_dir: Path,
    seek_margin: float = 2.0,
) -> ResolvedFrame:
    if target_timestamp < 0:
        raise V0Error("Dialogue timestamp cannot be negative.")
    try:
        import av
    except ImportError as exc:
        raise V0Error("PyAV is required for target-frame resolution.") from exc

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise V0Error(f"Could not create output directory {output_dir}: {exc}") from exc

    frame_path = output_dir / "dialogue_frame.png"
    try:
        with av.open(str(media_path)) as container:
            if not container.streams.video:
                raise V0Error("The acquired media has no decodable video stream.")
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise V0Error("The video stream has no usable time base.")

            seek_timestamp = max(0.0, target_timestamp - seek_margin)
            seek_offset = int(seek_timestamp / float(stream.time_base))
            container.seek(seek_offset, stream=stream, backward=True, any_frame=False)

            for decode_index, frame in enumerate(container.decode(stream)):
                time_base = frame.time_base or stream.time_base
                if frame.pts is None or time_base is None:
                    continue
                timestamp = float(frame.pts * time_base)
                if timestamp + 1e-9 < target_timestamp:
                    continue
                frame.to_image().save(frame_path, format="PNG")
                return ResolvedFrame(
                    index=decode_index,
                    pts=int(frame.pts),
                    time_base=str(time_base),
                    timestamp=timestamp,
                    path=frame_path.resolve(),
                )
    except V0Error:
        raise
    except (OSError, ValueError) as exc:
        raise V0Error(f"PyAV could not resolve the dialogue frame: {exc}") from exc
    except Exception as exc:
        if exc.__class__.__module__.startswith("av"):
            raise V0Error(f"PyAV could not resolve the dialogue frame: {exc}") from exc
        raise
    raise V0Error("No decoded video frame exists at or after the dialogue timestamp.")


def iter_frames_in_interval(
    media_path: Path,
    start_timestamp: float,
    end_timestamp: float,
    seek_margin: float = 2.0,
):
    if start_timestamp < 0 or end_timestamp < start_timestamp:
        raise V0Error("The OCR candidate interval is invalid.")
    try:
        import av
    except ImportError as exc:
        raise V0Error("PyAV is required for OCR frame decoding.") from exc

    try:
        with av.open(str(media_path)) as container:
            if not container.streams.video:
                raise V0Error("The acquired media has no decodable video stream.")
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise V0Error("The video stream has no usable time base.")

            seek_timestamp = max(0.0, start_timestamp - seek_margin)
            seek_offset = int(seek_timestamp / float(stream.time_base))
            container.seek(seek_offset, stream=stream, backward=True, any_frame=False)

            for decode_index, frame in enumerate(container.decode(stream)):
                time_base = frame.time_base or stream.time_base
                if frame.pts is None or time_base is None:
                    continue
                timestamp = float(frame.pts * time_base)
                if timestamp + 1e-9 < start_timestamp:
                    continue
                if timestamp - 1e-9 > end_timestamp:
                    break
                yield CandidateVideoFrame(
                    index=decode_index,
                    pts=int(frame.pts),
                    time_base=str(time_base),
                    timestamp=timestamp,
                    image=frame.to_image(),
                )
    except V0Error:
        raise
    except (OSError, ValueError) as exc:
        raise V0Error(f"PyAV could not decode the OCR candidate interval: {exc}") from exc
    except Exception as exc:
        if exc.__class__.__module__.startswith("av"):
            raise V0Error(f"PyAV could not decode the OCR candidate interval: {exc}") from exc
        raise
