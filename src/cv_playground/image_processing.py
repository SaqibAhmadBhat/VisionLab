import cv2
import numpy as np
from typing import Tuple
from .utils import ensure_grayscale

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to Grayscale."""
    return ensure_grayscale(image)

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """Apply Gaussian Blur."""
    # Ensure kernel size is odd and at least 1
    k = max(1, kernel_size | 1)
    return cv2.GaussianBlur(image, (k, k), sigma)

def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Median Blur."""
    k = max(1, kernel_size | 1)
    return cv2.medianBlur(image, k)

def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply Bilateral Filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_canny_edge(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Apply Canny Edge Detection."""
    gray = ensure_grayscale(image)
    return cv2.Canny(gray, threshold1, threshold2)

def apply_sobel_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Sobel Edge Detection (combined X and Y)."""
    gray = ensure_grayscale(image)
    k = max(1, ksize | 1)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    magnitude = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(magnitude)

def apply_laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Laplacian Edge Detection."""
    gray = ensure_grayscale(image)
    k = max(1, ksize | 1)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(laplacian)

def apply_binary_threshold(image: np.ndarray, thresh_val: int = 127) -> np.ndarray:
    """Apply basic binary thresholding."""
    gray = ensure_grayscale(image)
    _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return result

def apply_adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """Apply adaptive thresholding."""
    gray = ensure_grayscale(image)
    b = max(3, block_size | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, b, c)

def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's thresholding."""
    gray = ensure_grayscale(image)
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def apply_morphology(image: np.ndarray, operation: str = 'erosion', kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Apply morphological operations (erosion, dilation, opening, closing)."""
    k = max(1, kernel_size)
    kernel = np.ones((k, k), np.uint8)

    op_map = {
        'erosion': cv2.erode,
        'dilation': cv2.dilate,
    }

    if operation in op_map:
        return op_map[operation](image, kernel, iterations=iterations)

    cv2_ops = {
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE,
    }
    if operation in cv2_ops:
        return cv2.morphologyEx(image, cv2_ops[operation], kernel, iterations=iterations)

    return image

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to exact dimensions."""
    return cv2.resize(image, (width, height))

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle around its center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Adjust brightness and contrast.
    brightness: -127 to 127
    contrast: -127 to 127
    """
    # Using cv2.addWeighted for contrast and brightness
    # contrast: factor f = 131 * (c + 127) / (127 * (131 - c))
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    if brightness != 0:
        image = cv2.addWeighted(image, 1, image, 0, brightness)

    return image

def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """Apply a standard sharpening filter."""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)
