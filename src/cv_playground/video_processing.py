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
