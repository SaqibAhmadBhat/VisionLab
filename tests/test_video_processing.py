import numpy as np
import pytest
import tempfile
import os
import cv2
from src.cv_playground import video_processing


@pytest.fixture
def synthetic_video_path():
    """Create a short synthetic video (10 frames, 100x100, solid colors)."""
    path = tempfile.mktemp(suffix=".avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path, fourcc, 10.0, (100, 100), isColor=True)

    for i in range(10):
        frame = np.full((100, 100, 3), fill_value=(i * 25) % 256,
                        dtype=np.uint8)
        out.write(frame)

    out.release()
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


def test_get_video_info(synthetic_video_path):
    info = video_processing.get_video_info(synthetic_video_path)
    assert info["width"] == 100
    assert info["height"] == 100
    assert info["fps"] > 0
    assert info["frame_count"] == 10
    assert info["duration_sec"] > 0


def test_get_video_info_invalid():
    info = video_processing.get_video_info("/nonexistent/video.mp4")
    assert info == {}


def test_preview_frames(synthetic_video_path):
    frames = video_processing.process_video_preview(
        synthetic_video_path, max_frames=3
    )
    assert len(frames) <= 3
    assert len(frames) > 0
    # Frames should be RGB
    assert frames[0].shape == (100, 100, 3)


def test_preview_invalid_path():
    frames = video_processing.process_video_preview("/nonexistent.mp4")
    assert frames == []


def test_full_video_processing(synthetic_video_path):
    out_path = tempfile.mktemp(suffix=".mp4")
    try:
        success = video_processing.process_full_video(
            synthetic_video_path, out_path
        )
        assert success is True
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def test_full_video_with_operation(synthetic_video_path):
    """Process video with grayscale conversion."""
    out_path = tempfile.mktemp(suffix=".mp4")
    try:
        def to_gray(frame):
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        success = video_processing.process_full_video(
            synthetic_video_path, out_path, operation_func=to_gray
        )
        assert success is True
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def test_full_video_invalid_input():
    out_path = tempfile.mktemp(suffix=".mp4")
    success = video_processing.process_full_video(
        "/nonexistent.mp4", out_path
    )
    assert success is False


def test_full_video_progress_callback(synthetic_video_path):
    """Progress callback should be called."""
    out_path = tempfile.mktemp(suffix=".mp4")
    calls = []

    def track(current, total):
        calls.append((current, total))

    try:
        video_processing.process_full_video(
            synthetic_video_path, out_path,
            progress_callback=track
        )
        assert len(calls) > 0
        # Last call should have current == total
        assert calls[-1][0] == calls[-1][1]
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
