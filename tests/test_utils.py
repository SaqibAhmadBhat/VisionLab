import pytest
from src.cv_playground import utils
import numpy as np

def test_get_image_info():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    info = utils.get_image_info(img)
    assert info["width"] == 100
    assert info["height"] == 50
    assert info["channels"] == 3
    assert info["dtype"] == "uint8"

def test_ensure_grayscale():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = utils.ensure_grayscale(img)
    assert len(gray.shape) == 2
