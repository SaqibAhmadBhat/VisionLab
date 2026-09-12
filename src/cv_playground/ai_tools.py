import numpy as np
import cv2
from typing import Tuple, List, Optional
import streamlit as st

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@st.cache_resource
def load_ai_model():
    """Load and cache the PyTorch MobileNetV3 model."""
    if not TORCH_AVAILABLE:
        return None, None
    try:
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        model.eval()
        return model, weights
    except Exception as e:
        return None, None

def get_ai_intelligence(image: np.ndarray) -> Tuple[bool, str, List[Tuple[str, float]]]:
    """
    Optional AI feature using torchvision's MobileNetV3 Small.
    Returns: (success_bool, message, list_of_predictions)
    """
    if not TORCH_AVAILABLE:
        return False, "PyTorch and Torchvision are not installed. Please install them to use AI Features: pip install torch torchvision", []

    model, weights = load_ai_model()
    if model is None:
        return False, "Failed to load the AI model. Ensure dependencies are correct.", []

    try:
        # Preprocess image
        preprocess = weights.transforms()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(image)
        batch = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)

        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]

        return True, "Analysis Complete", [(category_name, score)]

    except Exception as e:
        return False, f"Failed to run AI model: {str(e)}", []
