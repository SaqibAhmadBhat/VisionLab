# PROJECT TITLE
VisionLab — Interactive Computer Vision Studio

# ONE-LINE DESCRIPTION
A modular, interactive computer vision platform built with Python, OpenCV, and Streamlit, featuring real-time image processing, feature extraction, video analysis, and MobileNetV3 classification.

# TECH STACK
**Languages:** Python  
**Frameworks:** Streamlit, OpenCV (`opencv-python-headless`), PyTorch / Torchvision (Optional)  
**Libraries:** NumPy, Matplotlib, Pillow  
**Testing/CI:** Pytest, GitHub Actions  

# KEY ENGINEERING HIGHLIGHTS
- **Decoupled Architecture:** Strictly separated the Streamlit UI presentation layer from the pure Python computer vision logic, enabling 100% testability of the core functions.
- **Robust Resource Management:** Implemented application-layer resource caching for AI models and Haar cascades to prevent memory leaks and ensure rapid UI responsiveness. Designed custom lifecycle managers using explicit `try...finally` teardown for temporary video processing files and `cv2.VideoCapture` objects.
- **Comprehensive Testing:** Developed a robust 53-test suite covering morphological edge cases, input validation, 1D array handling, and synthetic video generation.
- **Graceful Degradation:** Built the AI classification module (MobileNetV3) as an optional dependency. If Torch is unavailable, the application safely disables the AI workspace without crashing the classical CV tools.

# FEATURES
- **Image Studio:** Safe RGB/Grayscale conversions, contrast/brightness equalization, and spatial transformations.
- **Image Processing:** Gaussian, Median, and Bilateral filtering, alongside Canny edge detection and Otsu's adaptive thresholding.
- **Feature Detection:** Contour analysis, Hough circles/lines, and Haar Cascade face detection.
- **Video Lab:** Robust frame extraction, video pipeline execution, and re-encoding for web download.
- **AI Vision:** MobileNetV3 ImageNet classification returning Top-5 predictions with softmax confidence bounds.

# ENGINEERING CONTRIBUTIONS
- Rewrote the monolithic application into a scalable, workspace-driven pattern.
- Resolved memory bloat and thread-hanging bugs related to improper video stream handling.
- Upgraded testing framework from 8 partial tests to 53 comprehensive unit tests, fully automating the suite via GitHub Actions CI.

# GITHUB
[https://github.com/SaqibAhmadBhat/VisionLab](https://github.com/SaqibAhmadBhat/VisionLab)

# SUMMARY PARAGRAPH
VisionLab is a comprehensive interactive computer vision suite I developed to demonstrate scalable, test-driven application design. Built with Python, OpenCV, and Streamlit, the platform allows users to dynamically apply classical algorithms (filtering, edge detection, Hough transforms) and deep learning models (MobileNetV3) to images and video. The most significant engineering challenge I solved was strictly decoupling the Streamlit presentation layer from the core computer vision algorithms. This architecture eliminated framework lock-in, enabling me to build a robust 53-test automated CI pipeline that validates image matrices, video temporary file lifecycles, and edge-case inputs with 100% reliability.
