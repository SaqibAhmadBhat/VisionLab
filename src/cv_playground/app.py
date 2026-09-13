"""VisionLab — Interactive Computer Vision Studio

Streamlit application entry point. Provides workspaces for image
processing, edge detection, feature analysis, face detection,
video processing, and AI-powered classification.
"""
import streamlit as st
import numpy as np
import cv2
import tempfile
import os

from src.cv_playground import image_processing
from src.cv_playground import detection
from src.cv_playground import visualization
from src.cv_playground import video_processing
from src.cv_playground import utils
from src.cv_playground import ai_tools

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionLab — Interactive Computer Vision Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for a professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label {
        color: #b0b0d0 !important;
        font-size: 0.85rem;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Workspace title */
    .workspace-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Info cards */
    .info-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Metric improvements */
    [data-testid="stMetric"] {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources (app-level caching, not in core modules)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_haar_cascade():
    """Load and cache the Haar Cascade for face detection."""
    return detection.load_haar_cascade()


@st.cache_resource
def get_ai_model():
    """Load and cache the AI classification model."""
    return ai_tools.load_ai_model()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 VisionLab</h1>
        <p>Interactive Computer Vision Studio — Explore image processing,
        detection, and AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_image_metrics(image: np.ndarray):
    """Display image metadata in a clean metrics row."""
    info = utils.get_image_info(image)
    if not info:
        return

    cols = st.columns(6)
    cols[0].metric("Width", f"{info['width']} px")
    cols[1].metric("Height", f"{info['height']} px")
    cols[2].metric("Channels", str(info['channels']))
    cols[3].metric("Type", info['dtype'])
    cols[4].metric("Megapixels", str(info['megapixels']))
    cols[5].metric("Aspect", info['aspect_ratio'])


def render_comparison(original: np.ndarray, processed: np.ndarray,
                      label: str):
    """Show original and processed images side by side with download."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(original, width='stretch')
    with col2:
        st.markdown("**Processed**")
        channels = "GRAY" if len(processed.shape) == 2 else "RGB"
        st.image(processed, width='stretch', channels=channels)

    # Download options
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    with dl_col1:
        try:
            png_data = utils.encode_image_for_download(processed, "png")
            st.download_button(
                "📥 Download PNG",
                data=png_data,
                file_name=f"visionlab_{label.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
        except Exception:
            pass
    with dl_col2:
        try:
            jpg_data = utils.encode_image_for_download(processed, "jpg")
            st.download_button(
                "📥 Download JPEG",
                data=jpg_data,
                file_name=f"visionlab_{label.lower().replace(' ', '_')}.jpg",
                mime="image/jpeg"
            )
        except Exception:
            pass


def load_and_validate_image(uploaded_file):
    """Load an uploaded image and validate its size.

    Returns:
        numpy array or None if invalid.
    """
    try:
        img = utils.load_image(uploaded_file)
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        return None

    valid, msg = utils.validate_image_size(img)
    if not valid:
        st.warning(f"⚠️ {msg}")
        return None

    return img


# ---------------------------------------------------------------------------
# Workspace: Image Studio
# ---------------------------------------------------------------------------
def workspace_image_studio(img: np.ndarray):
    """Image Studio workspace: basic adjustments and transformations."""
    op = st.sidebar.selectbox("Tool", [
        "Preview", "Grayscale", "Brightness & Contrast",
        "Rotation", "Sharpening"
    ])

    result = img
    if op == "Grayscale":
        result = image_processing.apply_grayscale(img)
    elif op == "Brightness & Contrast":
        b = st.sidebar.slider("Brightness", -127, 127, 0)
        c = st.sidebar.slider("Contrast", -127, 127, 0)
        result = image_processing.adjust_brightness_contrast(img, b, c)
    elif op == "Rotation":
        angle = st.sidebar.slider("Angle (degrees)", -180.0, 180.0, 0.0,
                                  step=1.0)
        result = image_processing.rotate_image(img, angle)
    elif op == "Sharpening":
        result = image_processing.apply_sharpening(img)

    render_comparison(img, result, f"studio_{op}")

    if st.checkbox("Show histogram"):
        hist_img = visualization.generate_histogram(result)
        st.image(hist_img, caption="Pixel intensity distribution",
                 width='stretch')


# ---------------------------------------------------------------------------
# Workspace: Image Processing
# ---------------------------------------------------------------------------
def workspace_image_processing(img: np.ndarray):
    """Image Processing workspace: filters, thresholds, morphology."""
    category = st.sidebar.selectbox("Category", [
        "Blur & Smoothing", "Thresholding", "Morphological Operations"
    ])

    result = img
    op_name = "original"

    if category == "Blur & Smoothing":
        op = st.sidebar.selectbox("Filter", [
            "Gaussian Blur", "Median Blur", "Bilateral Filter"
        ])
        op_name = op

        if op == "Gaussian Blur":
            k = st.sidebar.slider("Kernel size (odd)", 1, 31, 5, step=2,
                                  help="Larger = more blur. Must be odd.")
            result = image_processing.apply_gaussian_blur(img, k)
        elif op == "Median Blur":
            k = st.sidebar.slider("Kernel size (odd)", 1, 31, 5, step=2,
                                  help="Good for salt-and-pepper noise.")
            result = image_processing.apply_median_blur(img, k)
        elif op == "Bilateral Filter":
            d = st.sidebar.slider("Diameter", 1, 20, 9,
                                  help="Size of pixel neighborhood.")
            sig = st.sidebar.slider("Sigma (color/space)", 10, 150, 75,
                                    help="Higher = more smoothing.")
            result = image_processing.apply_bilateral_filter(img, d, sig, sig)

    elif category == "Thresholding":
        op = st.sidebar.selectbox("Method", [
            "Binary Threshold", "Adaptive Threshold", "Otsu Threshold"
        ])
        op_name = op

        if op == "Binary Threshold":
            tv = st.sidebar.slider("Threshold value", 0, 255, 127)
            result = image_processing.apply_binary_threshold(img, tv)
        elif op == "Adaptive Threshold":
            bs = st.sidebar.slider("Block size (odd)", 3, 99, 11, step=2,
                                   help="Size of local neighborhood.")
            c = st.sidebar.slider("C constant", 0, 20, 2,
                                  help="Subtracted from the mean.")
            result = image_processing.apply_adaptive_threshold(img, bs, c)
        elif op == "Otsu Threshold":
            st.sidebar.info("Otsu's method automatically finds the optimal "
                            "threshold value.")
            result = image_processing.apply_otsu_threshold(img)

    elif category == "Morphological Operations":
        m_op = st.sidebar.selectbox("Operation", [
            "Erosion", "Dilation", "Opening", "Closing"
        ])
        op_name = f"morph_{m_op}"
        k = st.sidebar.slider("Kernel size", 1, 15, 3)
        iters = st.sidebar.slider("Iterations", 1, 5, 1)
        result = image_processing.apply_morphology(
            img, m_op.lower(), k, iters
        )

    render_comparison(img, result, op_name)

    if st.checkbox("Show histogram"):
        hist_img = visualization.generate_histogram(result)
        st.image(hist_img, caption="Pixel intensity distribution",
                 width='stretch')


# ---------------------------------------------------------------------------
# Workspace: Edge & Feature Analysis
# ---------------------------------------------------------------------------
def workspace_edge_features(img: np.ndarray):
    """Edge Detection & Feature Analysis workspace."""
    op = st.sidebar.selectbox("Analysis", [
        "Canny Edge Detection", "Sobel Edge Detection",
        "Laplacian Edge Detection", "Contour Analysis",
        "Shape Detection", "Hough Lines", "Hough Circles"
    ])

    result = img

    if op == "Canny Edge Detection":
        t1 = st.sidebar.slider("Lower threshold", 0, 255, 100)
        t2 = st.sidebar.slider("Upper threshold", 0, 255, 200)
        if t1 >= t2:
            st.sidebar.warning("Lower threshold should be less than upper.")
        result = image_processing.apply_canny_edge(img, t1, t2)

    elif op == "Sobel Edge Detection":
        k = st.sidebar.slider("Kernel size (odd)", 1, 7, 3, step=2)
        result = image_processing.apply_sobel_edge(img, k)

    elif op == "Laplacian Edge Detection":
        k = st.sidebar.slider("Kernel size (odd)", 1, 7, 3, step=2)
        result = image_processing.apply_laplacian_edge(img, k)

    elif op == "Contour Analysis":
        tv = st.sidebar.slider("Threshold value", 0, 255, 127)
        result, count, stats = detection.detect_contours(img, tv)
        st.success(f"Found **{count}** contours")
        if stats:
            st.dataframe(stats, width='stretch')

    elif op == "Shape Detection":
        st.sidebar.info("Uses contour approximation to classify shapes. "
                        "This is a classical heuristic, not AI.")
        tv = st.sidebar.slider("Threshold value", 0, 255, 127)
        result, shapes = detection.detect_shapes(img, tv)
        if shapes:
            st.success(f"Detected **{len(shapes)}** shape(s)")
            st.dataframe(
                [{"Shape": s["shape"], "Area (px)": s["area"],
                  "Perimeter (px)": s["perimeter"],
                  "Vertices": s["vertices"]}
                 for s in shapes],
                width='stretch'
            )
        else:
            st.info("No shapes detected. Try adjusting the threshold.")

    elif op == "Hough Lines":
        t = st.sidebar.slider("Accumulator threshold", 50, 300, 150,
                               help="Higher = fewer but stronger lines.")
        ml = st.sidebar.slider("Min line length (px)", 10, 200, 50)
        mg = st.sidebar.slider("Max line gap (px)", 1, 50, 10)
        result, line_count = detection.detect_hough_lines(img, t, ml, mg)
        st.success(f"Detected **{line_count}** line segment(s)")

    elif op == "Hough Circles":
        dp = st.sidebar.slider("dp (resolution ratio)", 1.0, 3.0, 1.2, 0.1,
                                help="Inverse ratio of accumulator resolution.")
        min_dist = st.sidebar.slider("Min distance between centers", 10, 200,
                                      50)
        p1 = st.sidebar.slider("Canny upper threshold", 10, 300, 100)
        p2 = st.sidebar.slider("Accumulator threshold", 10, 100, 30,
                                help="Lower = more circles detected.")
        result, circle_count = detection.detect_hough_circles(
            img, dp, min_dist, p1, p2
        )
        if circle_count > 0:
            st.success(f"Detected **{circle_count}** circle(s)")
        else:
            st.info("No circles detected. Try lowering the accumulator "
                    "threshold or adjusting parameters.")

    render_comparison(img, result, op)


# ---------------------------------------------------------------------------
# Workspace: Detection
# ---------------------------------------------------------------------------
def workspace_detection(img: np.ndarray):
    """Face Detection workspace using classical Haar Cascades."""
    st.markdown(
        '<div class="info-card">'
        "<strong>Method:</strong> OpenCV Haar Cascades "
        "(classical computer vision, not deep learning). "
        "Best for frontal faces in well-lit conditions."
        "</div>",
        unsafe_allow_html=True
    )

    scale = st.sidebar.slider("Scale factor", 1.05, 2.0, 1.1, 0.05,
                               help="How much to shrink the image at each "
                                    "scale. Lower = more detections but "
                                    "slower.")
    min_n = st.sidebar.slider("Min neighbors", 1, 10, 4,
                               help="Higher = fewer false positives.")

    cascade = get_haar_cascade()
    result, count = detection.detect_faces(img, scale, min_n, cascade)

    if count > 0:
        st.success(f"Detected **{count}** face(s)")
    else:
        st.info("No faces detected. Try adjusting scale factor or min "
                "neighbors, or use a clearer frontal face image.")

    render_comparison(img, result, "face_detection")


# ---------------------------------------------------------------------------
# Workspace: Video Lab
# ---------------------------------------------------------------------------
def workspace_video_lab():
    """Video Lab workspace: upload, inspect, process, and export videos."""
    video_file = st.file_uploader("Upload a video",
                                  type=["mp4", "avi", "mov"])
    if video_file is None:
        st.info("Upload a video file to begin. "
                "Supported formats: MP4, AVI, MOV.")
        return

    # Save to temp file
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        tmp_path = tmp.name
        tmp.close()
    except Exception as e:
        st.error(f"Could not save uploaded video: {e}")
        return

    try:
        # Video metadata
        info = video_processing.get_video_info(tmp_path)
        if info:
            st.markdown("**Video Information**")
            cols = st.columns(5)
            cols[0].metric("Resolution", f"{info['width']}×{info['height']}")
            cols[1].metric("FPS", str(info['fps']))
            cols[2].metric("Frames", str(info['frame_count']))
            cols[3].metric("Duration",
                           f"{info['duration_sec']}s" if info['duration_sec']
                           else "Unknown")
            cols[4].metric("Codec", info['codec'] or "Unknown")
        else:
            st.warning("Could not read video metadata. "
                       "The file may use an unsupported codec.")

        st.divider()

        op = st.sidebar.selectbox("Processing", [
            "None", "Grayscale", "Canny Edge"
        ])

        # Preview
        if st.button("🎬 Generate Preview"):
            with st.spinner("Extracting preview frames..."):
                frames = video_processing.process_video_preview(
                    tmp_path, max_frames=6
                )

            if not frames:
                st.error("Could not extract frames. "
                         "The codec may be unsupported.")
            else:
                st.success(f"Showing {len(frames)} preview frames")
                cols = st.columns(3)
                for i, frame in enumerate(frames):
                    processed = frame
                    if op == "Grayscale":
                        processed = image_processing.apply_grayscale(frame)
                    elif op == "Canny Edge":
                        processed = image_processing.apply_canny_edge(frame)

                    ch = "GRAY" if len(processed.shape) == 2 else "RGB"
                    cols[i % 3].image(
                        processed, channels=ch,
                        caption=f"Frame {i + 1}",
                        width='stretch'
                    )

        # Full processing
        if st.button("⚙️ Process & Download Full Video"):
            progress_bar = st.progress(0, text="Processing video...")

            def update_progress(current, total):
                if total > 0:
                    progress_bar.progress(
                        min(current / total, 1.0),
                        text=f"Processing frame {current}/{total}"
                    )

            out_path = tempfile.mktemp(suffix=".mp4")

            op_func = None
            if op == "Grayscale":
                op_func = image_processing.apply_grayscale
            elif op == "Canny Edge":
                op_func = image_processing.apply_canny_edge

            success = video_processing.process_full_video(
                tmp_path, out_path, op_func,
                progress_callback=update_progress
            )

            if success and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    video_bytes = f.read()

                progress_bar.progress(1.0, text="Complete!")
                st.success("Video processing complete!")
                st.download_button(
                    "📥 Download Processed Video",
                    data=video_bytes,
                    file_name=(
                        f"visionlab_{op.lower().replace(' ', '_')}.mp4"
                    ),
                    mime="video/mp4"
                )
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
            else:
                st.error("Video processing failed. The codec may be "
                         "unsupported on this system.")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Workspace: AI Vision
# ---------------------------------------------------------------------------
def workspace_ai_vision(img: np.ndarray):
    """AI Vision workspace: MobileNetV3 image classification."""
    st.markdown(
        '<div class="info-card">'
        "<strong>Model:</strong> MobileNetV3 Small (torchvision) — "
        "a lightweight classifier trained on ImageNet (1000 categories).<br>"
        "<strong>Requirement:</strong> <code>pip install torch "
        "torchvision</code>"
        "</div>",
        unsafe_allow_html=True
    )

    if not ai_tools.TORCH_AVAILABLE:
        st.warning("PyTorch is not installed. Install with:\n\n"
                   "```\npip install torch torchvision\n```")
        st.image(img, caption="Uploaded image", width='stretch')
        return

    if st.button("🧠 Classify Image"):
        with st.spinner("Running inference..."):
            model, weights = get_ai_model()
            success, msg, predictions = ai_tools.get_ai_intelligence(
                img, model=model, weights=weights, top_k=5
            )

        if success and predictions:
            st.success(msg)

            # Top prediction highlight
            top_cat, top_score = predictions[0]
            st.markdown(f"### {top_cat}")
            st.progress(top_score,
                        text=f"{top_score * 100:.1f}% confidence")

            # Remaining predictions
            if len(predictions) > 1:
                st.markdown("**Other possibilities:**")
                for cat, score in predictions[1:]:
                    col1, col2 = st.columns([3, 1])
                    col1.text(cat)
                    col2.text(f"{score * 100:.1f}%")
        else:
            st.error(msg)

    st.image(img, caption="Input image", width='stretch')


# ---------------------------------------------------------------------------
# Workspace: About
# ---------------------------------------------------------------------------
def workspace_about():
    """About workspace with project information."""
    st.markdown("""
    ## About VisionLab

    **VisionLab** is an interactive computer vision studio for exploring
    classical and AI-powered image analysis in real time.

    ### Workspaces
    - **Image Studio** — Basic adjustments: grayscale, brightness,
      contrast, rotation, sharpening
    - **Image Processing** — Filters, thresholds, and morphological
      operations
    - **Edge & Features** — Canny, Sobel, Laplacian edge detection;
      contour and shape analysis; Hough transforms
    - **Face Detection** — Classical Haar Cascade face detection
    - **Video Lab** — Video upload, frame preview, and full-video
      processing with export
    - **AI Vision** — MobileNetV3 image classification (optional,
      requires PyTorch)

    ### Tech Stack
    - **Language:** Python 3.11
    - **UI:** Streamlit
    - **Vision:** OpenCV, NumPy, Pillow, Matplotlib
    - **AI:** PyTorch & torchvision (optional)

    ### Author
    **Saqib Ahmad Bhat**
    — [github.com/SaqibAhmadBhat](https://github.com/SaqibAhmadBhat)

    VisionLab is developed and maintained by Saqib Ahmad Bhat.
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_header()

    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    workspace = st.sidebar.radio(
        "Workspace",
        [
            "🖼️ Image Studio",
            "🔧 Image Processing",
            "📐 Edge & Features",
            "👤 Face Detection",
            "🎥 Video Lab",
            "🧠 AI Vision",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    # Workspaces that don't need an image upload
    if workspace == "ℹ️ About":
        workspace_about()
        return

    if workspace == "🎥 Video Lab":
        st.markdown('<div class="workspace-title">Video Lab</div>',
                    unsafe_allow_html=True)
        workspace_video_lab()
        return

    # All other workspaces need an image
    workspace_name = workspace.split(" ", 1)[1]
    st.markdown(f'<div class="workspace-title">{workspace_name}</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported: JPEG, PNG, WebP. Max recommended: 4096×4096 px."
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="info-card">'
            "Upload an image to get started. Use the sidebar to select "
            "processing tools after uploading."
            "</div>",
            unsafe_allow_html=True
        )
        return

    img = load_and_validate_image(uploaded_file)
    if img is None:
        return

    # Image metadata
    with st.expander("📊 Image Information", expanded=False):
        render_image_metrics(img)

    # Route to workspace
    try:
        if workspace == "🖼️ Image Studio":
            workspace_image_studio(img)
        elif workspace == "🔧 Image Processing":
            workspace_image_processing(img)
        elif workspace == "📐 Edge & Features":
            workspace_edge_features(img)
        elif workspace == "👤 Face Detection":
            workspace_detection(img)
        elif workspace == "🧠 AI Vision":
            workspace_ai_vision(img)
    except Exception as e:
        st.error(
            "VisionLab encountered an error processing this image. "
            "Please try a different image or adjust the parameters."
        )
        with st.expander("Technical details"):
            st.code(str(e))


if __name__ == "__main__":
    main()
