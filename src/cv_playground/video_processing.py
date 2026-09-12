import cv2
import tempfile
import os

def process_video_frame(frame, operation_func):
    """Process a single video frame with the given function."""
    try:
        return operation_func(frame)
    except Exception:
        # Fallback if processing fails
        return frame

def process_video_preview(video_path: str, max_frames: int = 10):
    """Extract a few frames from a video for preview purposes."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return frames

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // max_frames)

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames

def process_full_video(video_path: str, output_path: str, operation_func=None) -> bool:
    """Process entire video frame-by-frame and export to output_path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if operation_func is not None:
            processed = operation_func(frame)
            # Ensure processed frame matches output dimensions and channels
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif processed.shape[2] == 4: # Handle RGBA just in case
                processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)
            out.write(processed)
        else:
            out.write(frame)

    cap.release()
    out.release()
    return True
