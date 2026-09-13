import cv2
import numpy as np
from PIL import Image
import io
from typing import Union, Tuple, Optional, Dict

# Maximum image dimension to prevent UI freezes
MAX_IMAGE_DIMENSION = 4096


def load_image(file_buffer) -> np.ndarray:
    """Load an image from a Streamlit uploaded file buffer safely.

    Converts all images to RGB numpy arrays. Validates that the
    result is a reasonable size for interactive processing.

    Args:
        file_buffer: Streamlit UploadedFile or file-like object.

    Returns:
        RGB numpy array.

    Raises:
        ValueError: If the file cannot be decoded as an image.
    """
    try:
        img_bytes = file_buffer.read()
        file_buffer.seek(0)

        if len(img_bytes) == 0:
            raise ValueError("Uploaded file is empty")

        img = Image.open(io.BytesIO(img_bytes))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        arr = np.array(img)

        if arr.size == 0:
            raise ValueError("Image has zero pixels")

        return arr
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read this file as an image: {str(e)}")


def get_image_info(image: np.ndarray) -> dict:
    """Return metadata about an image array.

    Args:
        image: Input numpy array.

    Returns:
        Dictionary with width, height, channels, dtype, megapixels,
        and aspect ratio. Empty dict if image is invalid.
    """
    if image is None or len(image.shape) < 2:
        return {}

    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    megapixels = round((h * w) / 1_000_000, 2)
    aspect = f"{w}:{h}"

    # Simplify aspect ratio
    from math import gcd
    g = gcd(w, h)
    if g > 0:
        aspect = f"{w // g}:{h // g}"

    return {
        "width": w,
        "height": h,
        "channels": c,
        "dtype": str(image.dtype),
        "megapixels": megapixels,
        "aspect_ratio": aspect
    }


def encode_image_for_download(image: np.ndarray, ext: str = "png") -> bytes:
    """Convert numpy array to bytes for download.

    Args:
        image: Input image (grayscale or RGB).
        ext: Output format ('png', 'jpg', 'jpeg').

    Returns:
        Encoded image bytes.

    Raises:
        ValueError: If encoding fails.
    """
    try:
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(image, 'RGB')

        buf = io.BytesIO()
        format_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}
        fmt = format_map.get(ext.lower(), "PNG")
        pil_img.save(buf, format=fmt)
        return buf.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to 2D grayscale if it has color channels.

    Args:
        image: Input numpy array (2D grayscale or 3D RGB).

    Returns:
        2D grayscale numpy array.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def validate_image_size(image: np.ndarray,
                        max_dim: int = MAX_IMAGE_DIMENSION) -> Tuple[bool, str]:
    """Check if image dimensions are within processing limits.

    Args:
        image: Input numpy array.
        max_dim: Maximum allowed width or height.

    Returns:
        Tuple of (is_valid, message).
    """
    h, w = image.shape[:2]
    if w > max_dim or h > max_dim:
        return (False,
                f"Image is {w}x{h} pixels. Maximum supported dimension "
                f"is {max_dim}px. Please resize before uploading.")
    return True, ""
