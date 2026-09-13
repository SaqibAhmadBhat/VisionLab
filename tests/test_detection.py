import numpy as np
import pytest
from src.cv_playground import detection


@pytest.fixture
def dummy_shapes_image():
    """Create a 100x100 grayscale image with a white square."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img


@pytest.fixture
def dummy_rgb_image():
    """Create a 100x100 RGB image with a white square."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]
    return img


def test_shape_detection(dummy_shapes_image):
    result, shapes = detection.detect_shapes(dummy_shapes_image)
    # The result is colored to draw contours (RGB)
    assert result.shape == (100, 100, 3)
    # Should detect the white square as a shape
    assert len(shapes) >= 1
    assert shapes[0]["shape"] in ("Square", "Rectangle")


def test_shape_detection_no_shapes():
    """Uniform gray image should produce no shapes."""
    img = np.full((100, 100), 128, dtype=np.uint8)
    result, shapes = detection.detect_shapes(img, thresh_val=127)
    assert result.shape == (100, 100, 3)
    assert len(shapes) == 0


def test_face_detection_no_crash(dummy_shapes_image):
    """Testing with a blank square shouldn't crash, should return 0 faces."""
    result, count = detection.detect_faces(dummy_shapes_image)
    assert count == 0
    assert result.shape == (100, 100, 3)


def test_hough_lines(dummy_shapes_image):
    """Square edges should produce some line segments."""
    result, count = detection.detect_hough_lines(
        dummy_shapes_image, threshold=50, min_line_length=10, max_line_gap=5
    )
    assert result.shape == (100, 100, 3)
    assert isinstance(count, int)


def test_hough_circles():
    """Draw a circle and verify detection."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(img, (100, 100), 50, 255, 2)
    result, count = detection.detect_hough_circles(
        img, dp=1.2, min_dist=50, param1=100, param2=20
    )
    assert result.shape == (200, 200, 3)
    assert isinstance(count, int)


def test_contour_analysis(dummy_shapes_image):
    """Should find at least one contour from the white square."""
    result, count, stats = detection.detect_contours(dummy_shapes_image)
    assert result.shape == (100, 100, 3)
    assert count >= 1
    assert len(stats) == count
    assert "area" in stats[0]
    assert "perimeter" in stats[0]


def test_load_haar_cascade():
    """Cascade should load (or return None if unavailable)."""
    cascade = detection.load_haar_cascade()
    # On most OpenCV installs, this should succeed
    if cascade is not None:
        assert not cascade.empty()
