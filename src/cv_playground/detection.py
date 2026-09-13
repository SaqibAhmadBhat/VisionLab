import cv2
import numpy as np
from typing import Tuple, List, Dict
from .utils import ensure_grayscale


def detect_hough_lines(image: np.ndarray, threshold: int = 150,
                       min_line_length: int = 50,
                       max_line_gap: int = 10) -> np.ndarray:
    """Detect lines using Probabilistic Hough Line Transform.

    Args:
        image: Input image (RGB or grayscale).
        threshold: Accumulator threshold for line detection.
        min_line_length: Minimum length of detected lines in pixels.
        max_line_gap: Maximum gap between line segments to join them.

    Returns:
        Image with detected lines drawn in red.
    """
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            pts = line.flatten()
            x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return output, line_count


def detect_hough_circles(image: np.ndarray, dp: float = 1.2,
                         min_dist: int = 20, param1: int = 50,
                         param2: int = 30, min_radius: int = 0,
                         max_radius: int = 0) -> Tuple[np.ndarray, int]:
    """Detect circles using Hough Circle Transform.

    Args:
        image: Input image (RGB or grayscale).
        dp: Inverse ratio of accumulator resolution to image resolution.
        min_dist: Minimum distance between circle centers.
        param1: Upper Canny edge threshold.
        param2: Accumulator threshold for circle detection.
        min_radius: Minimum circle radius (0 = no minimum).
        max_radius: Maximum circle radius (0 = no maximum).

    Returns:
        Tuple of (image with circles drawn, number of circles found).
    """
    gray = ensure_grayscale(image)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp, min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )

    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    circle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_count = len(circles[0])
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

    return output, circle_count


def detect_shapes(image: np.ndarray,
                  thresh_val: int = 127) -> Tuple[np.ndarray, List[Dict]]:
    """Detect and label basic geometric shapes using contour approximation.

    Uses thresholding and contour approximation to identify triangles,
    squares, rectangles, pentagons, and circles. This is a classical
    heuristic method, not AI-based classification.

    Args:
        image: Input image (RGB or grayscale).
        thresh_val: Binary threshold value for contour extraction.

    Returns:
        Tuple of (annotated image, list of shape info dicts).
    """
    gray = ensure_grayscale(image)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    shapes_found = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        vertices = len(approx)
        shape_name = "Unknown"

        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif vertices == 5:
            shape_name = "Pentagon"
        else:
            shape_name = "Circle"

        cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)

        M = cv2.moments(cnt)
        cx, cy = 0, 0
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, shape_name, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        x, y, w, h = cv2.boundingRect(cnt)
        shapes_found.append({
            "shape": shape_name,
            "vertices": vertices,
            "area": int(area),
            "perimeter": int(perimeter),
            "bounding_box": {"x": x, "y": y, "w": w, "h": h},
            "center": {"x": cx, "y": cy}
        })

    return output, shapes_found


def detect_contours(image: np.ndarray,
                    thresh_val: int = 127) -> Tuple[np.ndarray, int, List[Dict]]:
    """Detect and draw all contours, returning statistics.

    Args:
        image: Input image (RGB or grayscale).
        thresh_val: Binary threshold value for contour extraction.

    Returns:
        Tuple of (annotated image, contour count, list of contour stats).
    """
    gray = ensure_grayscale(image)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    stats = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        stats.append({
            "id": i + 1,
            "area": int(area),
            "perimeter": round(perimeter, 1),
            "bounding_box": f"{w}x{h} at ({x},{y})"
        })

    return output, len(contours), stats


def load_haar_cascade():
    """Load Haar Cascade classifier.

    Returns:
        CascadeClassifier or None if unavailable.
    """
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if cascade.empty():
            return None
        return cascade
    except AttributeError:
        return None


def detect_faces(image: np.ndarray, scale_factor: float = 1.1,
                 min_neighbors: int = 4,
                 face_cascade=None) -> Tuple[np.ndarray, int]:
    """Detect faces using Haar Cascades.

    This is a classical computer vision method using pre-trained Haar
    feature cascades, not a deep learning model.

    Args:
        image: Input image (RGB or grayscale).
        scale_factor: How much the image size is reduced at each scale.
        min_neighbors: Minimum neighbors for a detection to be retained.
        face_cascade: Pre-loaded CascadeClassifier (for caching).

    Returns:
        Tuple of (annotated image, number of faces detected).
    """
    gray = ensure_grayscale(image)

    if face_cascade is None:
        face_cascade = load_haar_cascade()
    if face_cascade is None:
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output, 0

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(output, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return output, len(faces)
