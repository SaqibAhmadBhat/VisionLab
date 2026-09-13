import pytest
import numpy as np
import io
from PIL import Image
from src.cv_playground import utils


# --- Image info ---
def test_get_image_info_rgb():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    info = utils.get_image_info(img)
    assert info["width"] == 100
    assert info["height"] == 50
    assert info["channels"] == 3
    assert info["dtype"] == "uint8"
    assert info["megapixels"] == 0.01
    assert "aspect_ratio" in info


def test_get_image_info_grayscale():
    img = np.zeros((100, 100), dtype=np.uint8)
    info = utils.get_image_info(img)
    assert info["channels"] == 1
    assert info["width"] == 100
    assert info["height"] == 100


def test_get_image_info_none():
    info = utils.get_image_info(None)
    assert info == {}


def test_get_image_info_1d():
    """1D array should return empty dict."""
    info = utils.get_image_info(np.array([1, 2, 3]))
    assert info == {}


# --- Grayscale conversion ---
def test_ensure_grayscale_rgb():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = utils.ensure_grayscale(img)
    assert len(gray.shape) == 2


def test_ensure_grayscale_already_gray():
    img = np.zeros((10, 10), dtype=np.uint8)
    gray = utils.ensure_grayscale(img)
    assert len(gray.shape) == 2
    np.testing.assert_array_equal(gray, img)


# --- Image encoding ---
def test_encode_png():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    data = utils.encode_image_for_download(img, "png")
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Verify it's valid PNG
    result = Image.open(io.BytesIO(data))
    assert result.size == (50, 50)


def test_encode_jpeg():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    data = utils.encode_image_for_download(img, "jpg")
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_encode_grayscale():
    img = np.zeros((50, 50), dtype=np.uint8)
    data = utils.encode_image_for_download(img, "png")
    assert isinstance(data, bytes)
    assert len(data) > 0


# --- Image validation ---
def test_validate_normal_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    valid, msg = utils.validate_image_size(img)
    assert valid is True
    assert msg == ""


def test_validate_oversized_image():
    img = np.zeros((5000, 5000, 3), dtype=np.uint8)
    valid, msg = utils.validate_image_size(img, max_dim=4096)
    assert valid is False
    assert "4096" in msg


# --- Image loading ---
def test_load_valid_image():
    """Create a valid PNG in memory and load it."""
    pil_img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    result = utils.load_image(buf)
    assert result.shape == (50, 50, 3)
    assert result[25, 25, 0] == 255  # Red channel


def test_load_rgba_image():
    """RGBA should be converted to RGB."""
    pil_img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    result = utils.load_image(buf)
    assert result.shape == (50, 50, 3)


def test_load_empty_file():
    """Empty file should raise ValueError."""
    buf = io.BytesIO(b"")
    with pytest.raises(ValueError, match="empty"):
        utils.load_image(buf)


def test_load_invalid_file():
    """Random bytes should raise ValueError."""
    buf = io.BytesIO(b"not an image at all")
    with pytest.raises(ValueError):
        utils.load_image(buf)
