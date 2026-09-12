# VisionLab — Interactive Computer Vision Studio

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

VisionLab is an interactive computer vision studio built with Python, OpenCV, NumPy, Pillow, Matplotlib, and Streamlit. It provides a practical workspace for image processing, feature detection, shape analysis, object detection, video analysis, and optional AI-powered visual classification.

The application is designed to make computer vision techniques accessible through an interactive interface while maintaining a modular Python architecture suitable for further development and experimentation.

## ✨ Features

### 🖼 Image Studio
- Image upload and live rendering
- Image information metadata
- Grayscale conversion
- Brightness adjustment
- Contrast adjustment
- Resize to exact dimensions
- Rotation
- Sharpening

### 🔍 Image Processing
- Gaussian blur (variable kernel)
- Median blur
- Bilateral filtering

### 📏 Edge Detection
- Canny edge detection
- Sobel edge detection
- Laplacian edge detection

### 🎯 Thresholding
- Binary threshold
- Adaptive threshold
- Otsu threshold

### 🧬 Morphological Operations
- Erosion
- Dilation
- Opening
- Closing

### 📐 Feature & Shape Detection
- Contour detection & drawing
- Geometric shape detection (Triangle, Square, Rectangle, Pentagon, Circle)
- Hough line detection
- Hough circle detection

### 🤖 Face Detection
- Lightweight Face Detection using OpenCV Haar Cascades
- Detects bounding boxes around frontal faces with adjustable scale factor and minimum neighbors.
- *Zero additional dependencies required.*

### 🎥 Video Lab
- Video upload (MP4, AVI, MOV)
- Extracts preview frames (up to 10 frames)
- Allows applying quick filters (Grayscale, Edge detection) to extracted frames.

### 🧠 AI Vision
- Uses **MobileNetV3 Small** (via PyTorch/Torchvision) for lightweight Image Classification.
- **Local Requirements:** Requires local PyTorch installation.

## 📸 Screenshots
*(Screenshots and demo media will be added soon)*

## 🏗 Architecture
Functionality is separated into strictly typed modules to ensure scalability and decoupling from the Streamlit UI.

```text
User
  ↓
Streamlit UI (app.py)
  ↓
Processing modules (image_processing.py, detection.py, video_processing.py, ai_tools.py)
  ↓
OpenCV / NumPy / Pillow / PyTorch
  ↓
Visualization (visualization.py) / Export
```

## 🧰 Tech Stack
- Python 3.11
- OpenCV (`opencv-python-headless`)
- NumPy
- Pillow
- Matplotlib
- Streamlit
- PyTorch & Torchvision (Optional)
- Pytest

## 🚀 Installation

**macOS / Linux:**
```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

*(Optional: Install AI dependencies manually with `pip install torch torchvision`)*

## ▶️ Running the Application

```bash
streamlit run src/cv_playground/app.py
```
Access the studio at `http://localhost:8501`.

## 🧪 Testing

The repository uses Pytest for logic and integration testing, backed by a GitHub Actions CI pipeline.

```bash
PYTHONPATH=. pytest tests/
```

## 📁 Project Structure

```text
visionlab/
├── .github/
│   └── workflows/
│       └── tests.yml
├── APP_README.md            # Detailed application documentation
├── requirements.txt
├── src/
│   └── cv_playground/
│       ├── __init__.py
│       ├── app.py
│       ├── image_processing.py
│       ├── video_processing.py
│       ├── detection.py
│       ├── visualization.py
│       ├── utils.py
│       └── ai_tools.py
├── tests/
│   ├── __init__.py
│   ├── test_image_processing.py
│   ├── test_detection.py
│   └── test_utils.py
├── .gitignore
├── LICENSE
└── ReadMe.md
```

## ⚠️ Limitations
- **Processing speed:** Python loop bottlenecks exist on high-resolution video streams.
- **Webcam support:** Browser constraints restrict live webcam features strictly to TLS/HTTPS or localhost.
- **Video limits:** Extracts a maximum of 10 preview frames to prevent Streamlit memory overflow.
- **AI Models:** MobileNet classification lacks advanced bounding-box object detection features.

## 🔮 Roadmap
*Future enhancements under consideration:*
- Real-time webcam processing (via `streamlit-webrtc`)
- Additional lightweight ONNX detection models (e.g., YOLOv8 Nano)
- Object tracking and bounding box persistence across video frames
- Batch image processing endpoints

## 👨‍💻 Author

**Saqib Ahmad Bhat**

GitHub: [https://github.com/SaqibAhmadBhat](https://github.com/SaqibAhmadBhat)

VisionLab is developed and maintained by Saqib Ahmad Bhat.

## 💡 Acknowledgments

This project was developed independently as a practical computer vision application. Its design and learning direction were informed by publicly available computer vision concepts, documentation, and educational resources.
