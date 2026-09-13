import cv2
import os
from typing import Optional, Callable, Dict


def get_video_info(video_path: str) -> Dict:
    """Extract metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with fps, width, height, frame_count, duration_sec,
        and codec. Empty dict if video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Decode fourcc
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        duration = 0.0
        if fps > 0:
            duration = round(frame_count / fps, 2)

        return {
            "fps": round(fps, 2) if fps > 0 else 0,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": duration,
            "codec": codec.strip()
        }
    finally:
        cap.release()


def process_video_preview(video_path: str, max_frames: int = 10):
    """Extract evenly-spaced frames from a video for preview.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of RGB numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return frames

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            # Try reading sequentially for videos without frame count
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames

        step = max(1, frame_count // max_frames)

        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()

    return frames


def process_video_frame(frame, operation_func):
    """Process a single video frame with the given function.

    Args:
        frame: BGR numpy array from cv2.VideoCapture.
        operation_func: Function to apply to the frame.

    Returns:
        Processed frame, or original on failure.
    """
    try:
        return operation_func(frame)
    except Exception:
        return frame


def process_full_video(video_path: str, output_path: str,
                       operation_func: Optional[Callable] = None,
                       progress_callback: Optional[Callable] = None) -> bool:
    """Process entire video frame-by-frame and export.

    Args:
        video_path: Input video path.
        output_path: Output video path.
        operation_func: Function to apply to each frame (operates on BGR).
        progress_callback: Called with (current_frame, total_frames).

    Returns:
        True if processing completed successfully.
    """
    cap = cv2.VideoCapture(video_path)
    out = None

    if not cap.isOpened():
        return False

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or fps != fps:  # NaN check
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (width, height), isColor=True)

        current = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if operation_func is not None:
                processed = operation_func(frame)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                elif processed.shape[2] == 4:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)
                out.write(processed)
            else:
                out.write(frame)

            current += 1
            if progress_callback and frame_count > 0:
                progress_callback(current, frame_count)

        return True
    except Exception:
        return False
    finally:
        cap.release()
        if out is not None:
            out.release()
