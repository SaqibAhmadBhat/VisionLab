import numpy as np
import pytest
from src.cv_playground import detection

@pytest.fixture
def dummy_shapes_image():
    """Create a 100x100 grayscale image with a white square."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img

def test_shape_detection(dummy_shapes_image):
    result = detection.detect_shapes(dummy_shapes_image)
    # The result is colored to draw contours (RGB)
    assert result.shape == (100, 100, 3)
    
def test_face_detection_no_crash(dummy_shapes_image):
    # Testing with a blank square shouldn't crash, should return 0 faces
    result, count = detection.detect_faces(dummy_shapes_image)
    assert count == 0
    assert result.shape == (100, 100, 3)
