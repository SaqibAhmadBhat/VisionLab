import cv2
import numpy as np
from PIL import Image
import io
from typing import Union, Tuple, Optional

def load_image(file_buffer) -> np.ndarray:
    """Load an image from a Streamlit uploaded file buffer safely."""
    try:
        img_bytes = file_buffer.read()
        file_buffer.seek(0) # Reset pointer for potential reuse
        
        # PIL handles many formats safely
        img = Image.open(io.BytesIO(img_bytes))
        # Convert to RGB if necessary (e.g. RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")

def get_image_info(image: np.ndarray) -> dict:
    """Return basic metadata about an image array."""
    if image is None or len(image.shape) < 2:
        return {}
    
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    return {
        "width": w,
        "height": h,
        "channels": c,
        "dtype": str(image.dtype)
    }

def encode_image_for_download(image: np.ndarray, ext: str = "jpg") -> bytes:
    """Convert numpy array back to bytes for downloading."""
    try:
        if len(image.shape) == 2:
            # Grayscale to PIL
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(image, 'RGB')
            
        buf = io.BytesIO()
        format_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}
        pil_img.save(buf, format=format_map.get(ext.lower(), "JPEG"))
        return buf.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure image is 2D grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
