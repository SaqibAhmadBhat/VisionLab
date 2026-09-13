# VisionLab — Application Documentation

## Overview

VisionLab is an interactive computer vision studio. It provides six workspaces for exploring classical image processing, edge detection, feature analysis, face detection, video processing, and AI-powered image classification.

The application uses Streamlit as the web interface and OpenCV as the core processing engine. All processing modules are framework-independent and can be reused outside of the Streamlit context.

## Workspaces

### 🖼 Image Studio
Upload an image and apply basic adjustments:
- **Grayscale** — Convert to single-channel grayscale
- **Brightness & Contrast** — Adjust using weighted addition
- **Rotation** — Rotate by any angle around the center
- **Sharpening** — Apply a 3×3 sharpening kernel
- **Histogram** — View pixel intensity distribution

### 🔧 Image Processing
Apply image processing algorithms organized by category:

**Blur & Smoothing:**
- Gaussian Blur (adjustable kernel size)
- Median Blur (good for salt-and-pepper noise)
- Bilateral Filter (edge-preserving smoothing)

**Thresholding:**
- Binary Threshold (manual threshold value)
- Adaptive Threshold (local neighborhood-based)
- Otsu Threshold (automatic optimal threshold)

**Morphological Operations:**
- Erosion, Dilation, Opening, Closing
- Adjustable kernel size and iterations

### 📐 Edge & Feature Analysis
- **Canny Edge Detection** — Dual-threshold edge detector
- **Sobel Edge Detection** — Gradient-based edge detection
- **Laplacian Edge Detection** — Second-derivative edge detection
- **Contour Analysis** — Find and measure all contours (area, perimeter, bounding box)
- **Shape Detection** — Classify contours as Triangle, Square, Rectangle, Pentagon, or Circle using polygon approximation (heuristic, not AI)
- **Hough Lines** — Detect line segments using probabilistic Hough transform
- **Hough Circles** — Detect circles using gradient-based Hough transform

### 👤 Face Detection
Uses OpenCV Haar Cascade classifier for frontal face detection.
- Adjustable scale factor and minimum neighbors
- Cascade is cached for performance
- This is a classical computer vision method, not deep learning

### 🎥 Video Lab
- Upload MP4, AVI, or MOV files
- View video metadata (resolution, FPS, frame count, duration, codec)
- Generate a preview of 6 evenly-spaced frames
- Apply Grayscale or Canny Edge to preview frames
- Process the entire video frame-by-frame and download the result
- Progress bar shows processing status

### 🧠 AI Vision
- Uses MobileNetV3 Small for ImageNet classification
- Returns top-5 predictions with confidence scores
- Requires PyTorch and torchvision (`pip install torch torchvision`)
- Model is cached after first load
- Application works without PyTorch; AI features are optional

## Architecture

```text
app.py (Streamlit)
  ├── Caching layer (@st.cache_resource)
  ├── Workspace functions (one per section)
  └── UI rendering helpers
      ↓
Processing modules (no Streamlit dependency):
  ├── image_processing.py — OpenCV wrappers for blur, edge, threshold, morph
  ├── detection.py — Hough transforms, contours, shapes, Haar faces
  ├── video_processing.py — VideoCapture/VideoWriter pipelines
  ├── ai_tools.py — MobileNetV3 inference
  ├── visualization.py — Matplotlib histogram generation
  └── utils.py — Image I/O, validation, encoding
```

## Running Locally

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
streamlit run src/cv_playground/app.py
```

## Testing

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

## 👨‍💻 Author

**Saqib Ahmad Bhat**

GitHub: [https://github.com/SaqibAhmadBhat](https://github.com/SaqibAhmadBhat)

VisionLab is developed and maintained by Saqib Ahmad Bhat.

## 💡 Acknowledgments

This project was developed independently as a practical computer vision application. Its design and learning direction were informed by publicly available computer vision concepts, documentation, and educational resources.
