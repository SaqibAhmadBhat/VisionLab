# VisionLab Architecture

VisionLab is structured to ensure a clean separation between the **user interface** and the **core computer vision logic**. This modularity makes the code easy to test, maintain, and reuse.

## High-Level Architecture

```mermaid
flowchart TD
    User([User]) --> UI[Streamlit UI (app.py)]
    
    subgraph "Application Layer"
        UI --> IS[Image Studio Workspace]
        UI --> IP[Image Processing Workspace]
        UI --> EF[Edge & Features Workspace]
        UI --> Det[Detection Workspace]
        UI --> VL[Video Lab Workspace]
        UI --> AIV[AI Vision Workspace]
        
        UI -.->|Caching| Cacher[Resource Caching]
    end
    
    subgraph "Core Vision Layer (Framework Independent)"
        IS --> utils[utils.py]
        IP --> imgProc[image_processing.py]
        EF --> imgProc
        EF --> detect[detection.py]
        Det --> detect
        VL --> vidProc[video_processing.py]
        AIV --> ai[ai_tools.py]
    end
    
    subgraph "External Dependencies"
        utils --> PIL[Pillow]
        utils --> cv2[OpenCV]
        imgProc --> cv2
        detect --> cv2
        vidProc --> cv2
        ai --> torch[PyTorch/Torchvision\n*Optional*]
    end
```

## 1. Application Flow & UI Layer
The entry point is `src/cv_playground/app.py`. It is responsible for:
- Routing the user to different workspaces (Image Studio, Video Lab, etc.).
- Handling file uploads and presenting processed results.
- Catching errors and formatting them for the user safely.
- Caching expensive resources (like Haar Cascades and AI models) using Streamlit's `@st.cache_resource`.

## 2. Core Computer Vision Layer
The core logic resides in dedicated modules that **do not depend on Streamlit**. They take standard Python data structures (NumPy arrays) as input and return NumPy arrays or basic types.
- **`image_processing.py`**: Wraps OpenCV filters, thresholds, and morphological operations.
- **`detection.py`**: Handles Hough transforms, contour analysis, shape heuristics, and Haar cascade face detection.
- **`video_processing.py`**: Manages `cv2.VideoCapture` and `cv2.VideoWriter` lifecycles, guaranteeing clean temporary file teardown via `try...finally` blocks, and exposing progress callbacks.
- **`utils.py`**: Validates image dimensions, limits processing loads, and enforces safe RGB/Grayscale conversions.

## 3. AI Classification
`ai_tools.py` implements an optional dependency pattern.
- If `torch` and `torchvision` are installed, it loads MobileNetV3 Small for ImageNet classification.
- If unavailable, the application gracefully degrades, allowing all classical CV features to function independently.

## 4. Testing & Dependencies
- **Tests**: The suite is located in `tests/` and covers 53 assertions verifying shapes, logic, idempotency, and error states without invoking the UI.
- **Dependencies**: The `requirements.txt` is kept minimal (OpenCV, Streamlit, NumPy, Matplotlib, Pillow) to ensure quick, reliable deployment.
