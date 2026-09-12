# VisionLab — Interactive Computer Vision Studio

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green.svg)

VisionLab is an interactive computer vision studio built with Python, OpenCV, NumPy, Pillow, and Streamlit.

It provides a practical workspace for uploading images and videos, applying computer-vision algorithms, adjusting parameters interactively, visualizing results, detecting shapes and features, and experimenting with AI-powered visual analysis.

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

### 🤖 Object Detection
- Lightweight Face Detection using OpenCV Haar Cascades
- Detects bounding boxes around frontal faces with adjustable scale factor and minimum neighbors.
- *Zero additional dependencies required.*

### 🎥 Video Lab
- Video upload (MP4, AVI, MOV)
- Extracts preview frames (up to 10 frames)
- Allows applying quick filters (Grayscale, Edge detection) to extracted frames.

### 🧠 AI Vision
- Uses **MobileNetV3 Small** (via PyTorch/Torchvision) for lightweight Image Classification.
- **Local Requirements:** Requires local PyTorch installation (`pip install torch torchvision`). 
- **Limitations:** The application gracefully degrades if Torch is absent, keeping the core OpenCV logic functional without requiring huge ML dependencies on standard installations.

## 🖥 Application Overview
**Upload** → **Select Vision Tool** → **Adjust Parameters** → **Process** → **Visualize** → **Download**

## 🧰 Tech Stack
- Python 3.11
- OpenCV (`opencv-python-headless`)
- NumPy
- Pillow
- Matplotlib
- Streamlit
- PyTorch & Torchvision (Optional for AI)

## 📁 Project Structure

```text
visionlab/
├── APP_README.md            # VisionLab application documentation
├── requirements.txt         # VisionLab dependencies
├── src/
│   └── cv_playground/       # Application source
│       ├── __init__.py
│       ├── app.py           # Streamlit entry point
│       ├── image_processing.py
│       ├── video_processing.py
│       ├── detection.py
│       ├── visualization.py
│       ├── utils.py
│       └── ai_tools.py
├── tests/                   # Pytest unit tests
├── .github/
│   └── workflows/           # CI Testing Pipeline
├── .gitignore               # Ignored environments and caches
└── LICENSE                  # Original upstream MIT License
```

## 🚀 Installation

```bash
# 1. Create a Python 3.11 virtual environment
python3.11 -m venv venv311

# 2. Activate (macOS/Linux)
source venv311/bin/activate
# (Windows)
# venv311\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# (Optional) Install AI dependencies
# pip install torch torchvision
```

## ▶️ Running VisionLab

```bash
streamlit run src/cv_playground/app.py
```
Open `http://localhost:8501` in your browser.

## 🧪 Testing

Run the test suite using `pytest`:
```bash
pip install pytest
PYTHONPATH=. pytest tests/
```

## 📸 Screenshots

*(To be added)*
See `docs/screenshots/` for upcoming interface snapshots:
1. Home dashboard
2. Image processing
3. Edge detection
4. Shape detection
5. Object detection
6. Video processing
7. AI Vision

## 🧠 How It Works

```text
User Interface (Streamlit sidebar controls)
      ↓
Streamlit Application (Routing in app.py)
      ↓
Processing Layer (Modular OpenCV wrappers)
      ↓
OpenCV / AI Models (Execute matrix operations)
      ↓
Visualization (Matplotlib histograms & Pillow overlays)
      ↓
Export (In-memory buffers for download)
```

## 🏗 Architecture
Functionality is separated into strictly typed modules:
- `image_processing.py`: Core pixel-level mathematical manipulations (Blur, edges, morphology).
- `detection.py`: Search-based geometric algorithms (Hough transforms, Contours, Haar Cascades).
- `utils.py`: Safe IO operations and buffer decoding.
This structure ensures the application can scale into a larger API/backend without being tied to Streamlit.

## ⚙️ Configuration
Currently, configurations are adjusted in real-time via the Streamlit UI Sidebar. Limits on video frame parsing are defined internally within `video_processing.py` to prevent browser timeouts.

## ⚠️ Limitations
- **Processing speed:** Python loop bottlenecks exist on high-resolution video streams.
- **Webcam support:** Browser constraints restrict `st.camera_input` strictly to TLS/HTTPS or localhost.
- **Video limits:** Extracts a maximum of 10 preview frames to prevent memory overflow in Streamlit.
- **AI Models:** MobileNet classification lacks advanced bounding-box object detection provided by heavier models like YOLO.

## 🔮 Future Improvements
- Real-time webcam processing (via `streamlit-webrtc`)
- Additional lightweight ONNX detection models (YOLOv8 Nano)
- Object tracking and bounding box persistence across video frames
- Batch image processing endpoints

## 📚 Educational Material & Attribution

The `Computer-Vision-main` context was authored originally by Dr-Mushtaq Hussain.
- The interactive **VisionLab** application (`src/`) was independently developed by Saqib Ahmad Bhat using the original educational project strictly as context and inspiration.
- Large educational datasets, copyrighted books, and legacy course materials from the original repository are intentionally omitted from this standalone software repository.
- Applicable original license terms are retained in `LICENSE`.

## 📄 License

1. **VisionLab Application:** 
   Original application code in `src/` and `tests/` is open-sourced under the MIT License.
   *Copyright (c) 2026 Saqib Ahmad Bhat*
2. **Original Project Inheritance:** 
   Authored by Dr-Mushtaq Hussain under MIT License.
3. **Third-Party Dependencies:** 
   OpenCV (Apache 2), Streamlit (Apache 2), PyTorch.

## 👨‍💻 Author

**Saqib Ahmad Bhat**  
GitHub: [https://github.com/SaqibAhmadBhat](https://github.com/SaqibAhmadBhat)
