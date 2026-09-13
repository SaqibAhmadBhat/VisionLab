# Resume Entries for VisionLab

Below are three versions of a resume entry for the VisionLab project, ranging from a single line to a strong three-bullet technical description. Choose the one that best fits your resume's space constraints.

## VERSION 1 — ONE-LINE RESUME ENTRY
**VisionLab** | *Python, OpenCV, Streamlit, PyTorch* | [GitHub](https://github.com/SaqibAhmadBhat/VisionLab)
- Developed an interactive computer vision application with a decoupled architecture, featuring 50+ automated tests, robust video processing lifecycles, and optional MobileNetV3 classification.

## VERSION 2 — 2 BULLET RESUME ENTRY
**VisionLab: Interactive Computer Vision Studio** | *Python, OpenCV, Streamlit, PyTorch, Pytest*
- Engineered a modular computer vision platform with strictly decoupled UI and core processing layers, supporting classical algorithms (contour analysis, Haar cascades) and deep learning classification.
- Implemented robust resource management for video processing and AI model caching, backed by a comprehensive 53-test automated CI pipeline to ensure zero regressions on edge-case inputs.

## VERSION 3 — 3 BULLET STRONG TECHNICAL ENTRY
**VisionLab: Interactive Computer Vision Studio**
*Technologies: Python, OpenCV, Streamlit, NumPy, PyTorch (MobileNetV3), Pytest, GitHub Actions*
- Architected and deployed an interactive computer vision suite allowing dynamic application of image processing, feature detection, and video analysis pipelines.
- Decoupled the Streamlit UI presentation layer from the core vision logic, enabling a highly testable architecture backed by a 53-test automated CI suite.
- Engineered robust, leak-free resource lifecycles using explicit `try...finally` teardowns for `cv2.VideoCapture` objects and implemented application-layer caching for computationally expensive AI models.
