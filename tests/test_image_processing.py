import numpy as np
import pytest
from src.cv_playground import image_processing


@pytest.fixture
def dummy_rgb_image():
    """Create a 100x100 RGB image with a white square in the center."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]
    return img


@pytest.fixture
def dummy_gray_image():
    """Create a 100x100 grayscale image with a white square."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img


# --- Grayscale ---
def test_grayscale_conversion(dummy_rgb_image):
    gray = image_processing.apply_grayscale(dummy_rgb_image)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)


def test_grayscale_idempotent(dummy_gray_image):
    """Applying grayscale to a grayscale image should be a no-op."""
    result = image_processing.apply_grayscale(dummy_gray_image)
    assert result.shape == (100, 100)
    np.testing.assert_array_equal(result, dummy_gray_image)


# --- Blur ---
def test_gaussian_blur(dummy_rgb_image):
    blurred = image_processing.apply_gaussian_blur(dummy_rgb_image, kernel_size=5)
    assert blurred.shape == (100, 100, 3)


def test_median_blur(dummy_rgb_image):
    blurred = image_processing.apply_median_blur(dummy_rgb_image, kernel_size=5)
    assert blurred.shape == (100, 100, 3)


def test_bilateral_filter(dummy_rgb_image):
    filtered = image_processing.apply_bilateral_filter(dummy_rgb_image, 9, 75, 75)
    assert filtered.shape == (100, 100, 3)


def test_gaussian_blur_even_kernel(dummy_rgb_image):
    """Even kernel size should be corrected to odd internally."""
    blurred = image_processing.apply_gaussian_blur(dummy_rgb_image, kernel_size=4)
    assert blurred.shape == (100, 100, 3)


# --- Edge Detection ---
def test_canny_edge(dummy_rgb_image):
    edges = image_processing.apply_canny_edge(dummy_rgb_image, 100, 200)
    assert edges.shape == (100, 100)
    assert np.any(edges > 0)


def test_sobel_edge(dummy_rgb_image):
    edges = image_processing.apply_sobel_edge(dummy_rgb_image, ksize=3)
    assert edges.shape == (100, 100)
    assert np.any(edges > 0)


def test_laplacian_edge(dummy_rgb_image):
    edges = image_processing.apply_laplacian_edge(dummy_rgb_image, ksize=3)
    assert edges.shape == (100, 100)
    assert np.any(edges > 0)


# --- Thresholding ---
def test_binary_threshold(dummy_rgb_image):
    thresh = image_processing.apply_binary_threshold(dummy_rgb_image, 127)
    assert thresh.shape == (100, 100)
    assert thresh[50, 50] == 255
    assert thresh[10, 10] == 0


def test_adaptive_threshold(dummy_rgb_image):
    thresh = image_processing.apply_adaptive_threshold(dummy_rgb_image, 11, 2)
    assert thresh.shape == (100, 100)
    assert thresh.dtype == np.uint8


def test_otsu_threshold(dummy_rgb_image):
    thresh = image_processing.apply_otsu_threshold(dummy_rgb_image)
    assert thresh.shape == (100, 100)
    # Otsu should separate the white square from black background
    assert thresh[50, 50] == 255
    assert thresh[10, 10] == 0


# --- Morphology ---
def test_morphology_erosion(dummy_rgb_image):
    result = image_processing.apply_morphology(dummy_rgb_image, "erosion", 3, 1)
    assert result.shape == (100, 100, 3)


def test_morphology_dilation(dummy_rgb_image):
    result = image_processing.apply_morphology(dummy_rgb_image, "dilation", 3, 1)
    assert result.shape == (100, 100, 3)


def test_morphology_opening(dummy_rgb_image):
    result = image_processing.apply_morphology(dummy_rgb_image, "opening", 3, 1)
    assert result.shape == (100, 100, 3)


def test_morphology_closing(dummy_rgb_image):
    result = image_processing.apply_morphology(dummy_rgb_image, "closing", 3, 1)
    assert result.shape == (100, 100, 3)


def test_morphology_unknown_returns_original(dummy_rgb_image):
    """Unknown operation name should return the original image."""
    result = image_processing.apply_morphology(dummy_rgb_image, "invalid", 3, 1)
    np.testing.assert_array_equal(result, dummy_rgb_image)


# --- Transformations ---
def test_rotate_zero(dummy_rgb_image):
    result = image_processing.rotate_image(dummy_rgb_image, 0.0)
    assert result.shape == (100, 100, 3)


def test_rotate_90(dummy_rgb_image):
    result = image_processing.rotate_image(dummy_rgb_image, 90.0)
    assert result.shape == (100, 100, 3)


def test_resize(dummy_rgb_image):
    result = image_processing.resize_image(dummy_rgb_image, 50, 50)
    assert result.shape == (50, 50, 3)


def test_brightness(dummy_rgb_image):
    result = image_processing.adjust_brightness_contrast(dummy_rgb_image, 50, 0)
    assert result.shape == (100, 100, 3)


def test_contrast(dummy_rgb_image):
    result = image_processing.adjust_brightness_contrast(dummy_rgb_image, 0, 50)
    assert result.shape == (100, 100, 3)


def test_sharpening(dummy_rgb_image):
    result = image_processing.apply_sharpening(dummy_rgb_image)
    assert result.shape == (100, 100, 3)
