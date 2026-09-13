# VisionLab — Interactive Computer Vision Studio

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green.svg)

VisionLab is an interactive computer vision studio built with Python, OpenCV, and Streamlit. Upload images and videos, apply real-time processing algorithms, detect features and faces, and run AI-powered classification — all through a clean browser interface.

## ✨ Features

### 🖼 Image Studio
- Upload and preview images (JPEG, PNG, WebP)
- Image metadata: resolution, channels, megapixels, aspect ratio
- Grayscale conversion, brightness/contrast adjustment
- Rotation and sharpening
- Pixel intensity histograms

### 🔧 Image Processing
- **Blur & Smoothing:** Gaussian, Median, Bilateral filter
- **Thresholding:** Binary, Adaptive (Gaussian), Otsu's automatic
- **Morphology:** Erosion, Dilation, Opening, Closing

### 📐 Edge & Feature Analysis
- Canny, Sobel, and Laplacian edge detection
- Contour analysis with area/perimeter statistics
- Geometric shape detection (heuristic classification)
- Hough line and circle detection

### 👤 Face Detection
- OpenCV Haar Cascade frontal face detection
- Adjustable scale factor and neighbor threshold
- Cached cascade loading for fast repeated use

### 🎥 Video Lab
- Upload video files (MP4, AVI, MOV)
- Video metadata display (resolution, FPS, duration, codec)
- Frame preview extraction
- Full video processing with frame-by-frame filter application
- Processed video download (.mp4)
- Progress bar for long videos

### 🧠 AI Vision (Optional)
- MobileNetV3 Small image classification (ImageNet, 1000 categories)
- Top-5 predictions with confidence scores
- Cached model loading
- Graceful fallback when PyTorch is not installed

## 🏗 Architecture

```text
User
  ↓
Streamlit UI (app.py) — routing, caching, styling
  ↓
Processing modules — framework-independent OpenCV wrappers
  ├── image_processing.py — blur, edges, thresholds, morphology
  ├── detection.py — Hough transforms, shapes, faces
  ├── video_processing.py — frame extraction, full video processing
  ├── ai_tools.py — MobileNetV3 classification
  ├── visualization.py — histograms
  └── utils.py — image I/O, validation, encoding
  ↓
OpenCV / NumPy / Pillow / PyTorch
```

Core processing modules have **no Streamlit dependency** and can be used independently or in other applications.

## 🧰 Tech Stack
- Python 3.11
- OpenCV (`opencv-python-headless`)
- NumPy
- Pillow
- Matplotlib
- Streamlit
- PyTorch & torchvision (optional — AI Vision only)

## 🚀 Installation

```bash
# Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate  # macOS/Linux
# venv311\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: AI Vision support
pip install torch torchvision
```

## ▶️ Running

```bash
streamlit run src/cv_playground/app.py
```

Open `http://localhost:8501` in your browser.

## 🧪 Testing

53 tests covering image processing, detection, video processing, and utilities:

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

CI runs automatically on push via GitHub Actions.

## 📁 Project Structure

```text
VisionLab/
├── src/cv_playground/
│   ├── app.py                 # Streamlit entry point
│   ├── image_processing.py    # Image operations
│   ├── detection.py           # Feature & face detection
│   ├── video_processing.py    # Video analysis & export
│   ├── ai_tools.py            # AI classification
│   ├── visualization.py       # Histogram generation
│   └── utils.py               # I/O and validation
├── tests/
│   ├── test_image_processing.py
│   ├── test_detection.py
│   ├── test_video_processing.py
│   └── test_utils.py
├── .github/workflows/tests.yml
├── requirements.txt
├── LICENSE
└── ReadMe.md
```

## ⚠️ Limitations
- **Image size:** Maximum recommended 4096×4096 pixels for interactive use
- **Video codecs:** Output uses mp4v codec; some systems may have limited codec support
- **AI models:** Classification only (no bounding-box object detection)
- **Webcam:** Not supported (requires HTTPS in browsers)

## 🔮 Roadmap
- Real-time webcam processing via `streamlit-webrtc`
- Lightweight ONNX object detection models
- Batch image processing
- Filter chaining (apply multiple operations sequentially)

## 👨‍💻 Author

**Saqib Ahmad Bhat**

GitHub: [https://github.com/SaqibAhmadBhat](https://github.com/SaqibAhmadBhat)

VisionLab is developed and maintained by Saqib Ahmad Bhat.

## 💡 Acknowledgments

This project was developed independently as a practical computer vision application. Its design and learning direction were informed by publicly available computer vision concepts, documentation, and educational resources.

## 📄 License

This project is open-source under the [MIT License](LICENSE).
