import streamlit as st
import numpy as np
import cv2
from PIL import Image

import sys
import os
from pathlib import Path

# Add project root to sys.path to allow absolute package imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cv_playground import image_processing
from src.cv_playground import detection
from src.cv_playground import visualization
from src.cv_playground import video_processing
from src.cv_playground import utils
from src.cv_playground import ai_tools

st.set_page_config(
    page_title="VisionLab - Interactive Computer Vision Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_about():
    st.title("About VisionLab")
    st.markdown("""
    **VisionLab** is an interactive, professional-grade computer vision studio designed for exploring classical image and video processing algorithms in real-time.
    
    ### Technologies
    - **Language**: Python 3.11
    - **Framework**: Streamlit
    - **Core Engine**: OpenCV, NumPy, Pillow, Matplotlib
    - **AI Engine (Optional)**: PyTorch & Torchvision
    
    ### Author
    **Saqib Ahmad Bhat**
    - GitHub: [@SaqibAhmadBhat](https://github.com/SaqibAhmadBhat)
    
    *Note: This application is independently developed. It is maintained within the `Computer-Vision-main` repository, which contains separate original educational materials.*
    """)

def render_image_metrics(image: np.ndarray):
    info = utils.get_image_info(image)
    if not info: return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Width", f"{info.get('width')} px")
    col2.metric("Height", f"{info.get('height')} px")
    col3.metric("Channels", str(info.get('channels')))
    col4.metric("Data Type", str(info.get('dtype')))

def render_comparison(original: np.ndarray, processed: np.ndarray, label: str):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)
    with col2:
        st.subheader("Processed Image")
        channels = "GRAY" if len(processed.shape) == 2 else "RGB"
        st.image(processed, use_container_width=True, channels=channels)
        
        try:
            download_data = utils.encode_image_for_download(processed, "jpg")
            st.download_button(
                label="📥 Download Processed Image",
                data=download_data,
                file_name=f"visionlab_{label.lower().replace(' ', '_')}.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Download unavailable: {e}")

def main():
    st.sidebar.title("VisionLab 👁️")
    st.sidebar.markdown("Interactive Computer Vision Studio")
    
    menu = [
        "🖼️ Image Processing",
        "🔍 Feature & Shape Detection",
        "🤖 Object Detection",
        "🎥 Video Lab",
        "🧠 AI Vision",
        "📸 Camera Input",
        "ℹ️ About"
    ]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "ℹ️ About":
        render_about()
        return

    if choice == "📸 Camera Input":
        st.title("Camera Input")
        st.markdown("Take a picture with your webcam and apply effects.")
        cam_image = st.camera_input("Take a picture")
        if cam_image is not None:
            try:
                img = utils.load_image(cam_image)
                render_image_metrics(img)
                st.subheader("Apply Quick Filter")
                filter_choice = st.radio("Filter", ["None", "Grayscale", "Gaussian Blur", "Canny Edge"])
                result = img
                if filter_choice == "Grayscale":
                    result = image_processing.apply_grayscale(img)
                elif filter_choice == "Gaussian Blur":
                    result = image_processing.apply_gaussian_blur(img, 15)
                elif filter_choice == "Canny Edge":
                    result = image_processing.apply_canny_edge(img, 100, 200)
                render_comparison(img, result, "camera_" + filter_choice)
            except Exception as e:
                st.error(f"Error processing image: {e}")
        return

    # Standard File Uploader for Image Modules
    if choice not in ["🎥 Video Lab"]:
        st.title(choice.split(" ", 1)[1])
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is None:
            st.info("Please upload an image to get started.")
            return
            
        try:
            img = utils.load_image(uploaded_file)
            st.expander("Image Metadata").write(utils.get_image_info(img))
            
            result = img
            operation_name = "original"
            
            if choice == "🖼️ Image Processing":
                op = st.sidebar.selectbox("Operation", [
                    "Original", "Grayscale", "Gaussian Blur", "Median Blur", 
                    "Bilateral Filter", "Canny Edge", "Sobel Edge", "Laplacian Edge",
                    "Binary Threshold", "Adaptive Threshold", "Otsu Threshold",
                    "Morphological Operations", "Brightness & Contrast", "Sharpening", "Rotate"
                ])
                operation_name = op
                
                if op == "Grayscale":
                    result = image_processing.apply_grayscale(img)
                elif op == "Gaussian Blur":
                    k = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)
                    result = image_processing.apply_gaussian_blur(img, k)
                elif op == "Median Blur":
                    k = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)
                    result = image_processing.apply_median_blur(img, k)
                elif op == "Bilateral Filter":
                    d = st.sidebar.slider("Diameter", 1, 20, 9)
                    sig = st.sidebar.slider("Sigma (Color/Space)", 10, 150, 75)
                    result = image_processing.apply_bilateral_filter(img, d, sig, sig)
                elif op == "Canny Edge":
                    t1 = st.sidebar.slider("Lower Threshold", 0, 255, 100)
                    t2 = st.sidebar.slider("Upper Threshold", 0, 255, 200)
                    result = image_processing.apply_canny_edge(img, t1, t2)
                elif op == "Sobel Edge":
                    k = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
                    result = image_processing.apply_sobel_edge(img, k)
                elif op == "Laplacian Edge":
                    k = st.sidebar.slider("Kernel Size", 1, 7, 3, step=2)
                    result = image_processing.apply_laplacian_edge(img, k)
                elif op == "Binary Threshold":
                    tv = st.sidebar.slider("Threshold Value", 0, 255, 127)
                    result = image_processing.apply_binary_threshold(img, tv)
                elif op == "Adaptive Threshold":
                    bs = st.sidebar.slider("Block Size", 3, 99, 11, step=2)
                    c = st.sidebar.slider("C Constant", 0, 10, 2)
                    result = image_processing.apply_adaptive_threshold(img, bs, c)
                elif op == "Otsu Threshold":
                    result = image_processing.apply_otsu_threshold(img)
                elif op == "Morphological Operations":
                    m_op = st.sidebar.selectbox("Type", ["erosion", "dilation", "opening", "closing"])
                    k = st.sidebar.slider("Kernel Size", 1, 15, 3)
                    iters = st.sidebar.slider("Iterations", 1, 5, 1)
                    result = image_processing.apply_morphology(img, m_op, k, iters)
                elif op == "Brightness & Contrast":
                    b = st.sidebar.slider("Brightness", -127, 127, 0)
                    c = st.sidebar.slider("Contrast", -127, 127, 0)
                    result = image_processing.adjust_brightness_contrast(img, b, c)
                elif op == "Sharpening":
                    result = image_processing.apply_sharpening(img)
                elif op == "Rotate":
                    angle = st.sidebar.slider("Angle", -180.0, 180.0, 0.0)
                    result = image_processing.rotate_image(img, angle)
                    
            elif choice == "🔍 Feature & Shape Detection":
                op = st.sidebar.selectbox("Detector", [
                    "Hough Lines", "Hough Circles", "Basic Shape Detection"
                ])
                operation_name = op
                
                if op == "Hough Lines":
                    t = st.sidebar.slider("Threshold", 50, 300, 150)
                    ml = st.sidebar.slider("Min Line Length", 10, 200, 50)
                    mg = st.sidebar.slider("Max Line Gap", 1, 50, 10)
                    result = detection.detect_hough_lines(img, t, ml, mg)
                elif op == "Hough Circles":
                    dp = st.sidebar.slider("dp", 1.0, 3.0, 1.2, 0.1)
                    min_dist = st.sidebar.slider("min_dist", 10, 100, 20)
                    p1 = st.sidebar.slider("param1 (Canny)", 10, 150, 50)
                    p2 = st.sidebar.slider("param2 (Accumulator)", 10, 100, 30)
                    result = detection.detect_hough_circles(img, dp, min_dist, p1, p2)
                elif op == "Basic Shape Detection":
                    tv = st.sidebar.slider("Threshold Value", 0, 255, 127)
                    result = detection.detect_shapes(img, tv)
                    
            elif choice == "🤖 Object Detection":
                st.markdown("Uses OpenCV Haar Cascades for lightweight, zero-download face detection.")
                scale = st.sidebar.slider("Scale Factor", 1.01, 1.5, 1.1)
                min_n = st.sidebar.slider("Min Neighbors", 1, 10, 4)
                result, count = detection.detect_faces(img, scale, min_n)
                operation_name = "faces"
                st.success(f"Detected {count} face(s).")
                
            elif choice == "🧠 AI Vision":
                st.markdown("Uses PyTorch and torchvision MobileNetV3 for Image Classification.")
                if st.button("Run AI Intelligence Analysis"):
                    with st.spinner("Analyzing image..."):
                        success, msg, predictions = ai_tools.get_ai_intelligence(img)
                        if success:
                            st.success(msg)
                            for cat, score in predictions:
                                st.metric(cat, f"{score*100:.2f}% Confidence")
                        else:
                            st.warning(msg)
                            st.info("You can install Torch manually using `pip install torch torchvision` to enable this feature.")

            # Display comparison
            render_comparison(img, result, operation_name)
            
            # Histogram section
            if st.checkbox("Show Image Histogram"):
                hist_img = visualization.generate_histogram(result)
                st.image(hist_img, caption="Histogram of Processed Image", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Video Module
    if choice == "🎥 Video Lab":
        st.title("Video Lab")
        st.markdown("Extract and process frames from a video. For performance, we preview up to 10 frames evenly spaced.")
        
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if video_file:
            import tempfile
            import os
            try:
                # Save temporarily for OpenCV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                    
                op = st.sidebar.selectbox("Frame Operation", ["None", "Grayscale", "Canny Edge"])
                
                with st.spinner("Extracting frames..."):
                    frames = video_processing.process_video_preview(tmp_path, max_frames=6)
                    
                if not frames:
                    st.error("Failed to extract frames. Codec may be unsupported.")
                else:
                    st.success(f"Extracted {len(frames)} frames for preview.")
                    cols = st.columns(3)
                    for i, frame in enumerate(frames):
                        processed = frame
                        if op == "Grayscale":
                            processed = image_processing.apply_grayscale(frame)
                        elif op == "Canny Edge":
                            processed = image_processing.apply_canny_edge(frame)
                            
                        channels = "GRAY" if len(processed.shape) == 2 else "RGB"
                        cols[i % 3].image(processed, channels=channels, caption=f"Frame Preview {i+1}", use_container_width=True)
                
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Video processing error: {e}")

if __name__ == "__main__":
    main()
