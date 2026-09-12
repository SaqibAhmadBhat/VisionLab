import numpy as np
import pytest
from src.cv_playground import image_processing

@pytest.fixture
def dummy_rgb_image():
    """Create a 100x100 RGB image with a white square in the center."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]
    return img

def test_grayscale_conversion(dummy_rgb_image):
    gray = image_processing.apply_grayscale(dummy_rgb_image)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)

def test_gaussian_blur(dummy_rgb_image):
    blurred = image_processing.apply_gaussian_blur(dummy_rgb_image, kernel_size=5)
    assert blurred.shape == (100, 100, 3)

def test_canny_edge(dummy_rgb_image):
    edges = image_processing.apply_canny_edge(dummy_rgb_image, 100, 200)
    assert edges.shape == (100, 100)
    # The edges array should have some non-zero pixels because there's a square
    assert np.any(edges > 0)

def test_binary_threshold(dummy_rgb_image):
    thresh = image_processing.apply_binary_threshold(dummy_rgb_image, 127)
    assert thresh.shape == (100, 100)
    # Inside the square should be 255
    assert thresh[50, 50] == 255
    # Outside should be 0
    assert thresh[10, 10] == 0
