import cv2
import numpy as np
from typing import Tuple, List, Dict
from .utils import ensure_grayscale

def detect_hough_lines(image: np.ndarray, threshold: int = 150, min_line_length: int = 50, max_line_gap: int = 10) -> np.ndarray:
    """Detect lines using Probabilistic Hough Line Transform."""
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
    return output

def detect_hough_circles(image: np.ndarray, dp: float = 1.2, min_dist: int = 20, 
                         param1: int = 50, param2: int = 30, 
                         min_radius: int = 0, max_radius: int = 0) -> np.ndarray:
    """Detect circles using Hough Circle Transform."""
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
        
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    return output

def detect_shapes(image: np.ndarray, thresh_val: int = 127) -> np.ndarray:
    """Detect and label basic shapes (triangle, square, rectangle, pentagon, circle)."""
    gray = ensure_grayscale(image)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        
    for cnt in contours:
        # Ignore very small contours
        if cv2.contourArea(cnt) < 500:
            continue
            
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Determine shape
        shape_name = "Unknown"
        vertices = len(approx)
        
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif vertices == 5:
            shape_name = "Pentagon"
        else:
            shape_name = "Circle"
            
        cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
        
        # Put text
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, shape_name, (cx - 20, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
    return output

def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 4) -> Tuple[np.ndarray, int]:
    """Detect faces using Haar Cascades."""
    gray = ensure_grayscale(image)
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            output = image.copy()
            if len(output.shape) == 2:
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
            return output, 0
            
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
            
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(output, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        return output, len(faces)
    except AttributeError:
        # CascadeClassifier might be missing in some cv2 headless/v5 builds
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output, 0
