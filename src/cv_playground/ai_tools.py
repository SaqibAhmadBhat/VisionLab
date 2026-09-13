import numpy as np
import cv2
from typing import Tuple, List, Optional

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_ai_model():
    """Load the PyTorch MobileNetV3 Small model.

    Returns:
        Tuple of (model, weights) or (None, None) if unavailable.
    """
    if not TORCH_AVAILABLE:
        return None, None
    try:
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        model.eval()
        return model, weights
    except Exception:
        return None, None


def get_ai_intelligence(image: np.ndarray,
                        model=None,
                        weights=None,
                        top_k: int = 5) -> Tuple[bool, str, List[Tuple[str, float]]]:
    """Run image classification using MobileNetV3 Small.

    This is an optional feature requiring PyTorch and torchvision.
    The model classifies images into ImageNet categories.

    Args:
        image: Input image as numpy array (RGB).
        model: Pre-loaded model (for caching). If None, loads fresh.
        weights: Pre-loaded weights. If None, loads fresh.
        top_k: Number of top predictions to return.

    Returns:
        Tuple of (success, message, list of (category, confidence) tuples).
    """
    if not TORCH_AVAILABLE:
        return (False,
                "PyTorch and torchvision are not installed. "
                "Install with: pip install torch torchvision",
                [])

    if model is None or weights is None:
        model, weights = load_ai_model()
    if model is None:
        return (False,
                "Failed to load the AI model. "
                "Ensure torch and torchvision are correctly installed.",
                [])

    try:
        preprocess = weights.transforms()

        img = image.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(img)
        batch = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)

        categories = weights.meta["categories"]

        top_values, top_indices = prediction.topk(min(top_k, len(categories)))
        results = []
        for val, idx in zip(top_values, top_indices):
            results.append((categories[idx.item()], val.item()))

        return True, "Classification complete", results

    except Exception as e:
        return False, f"Classification failed: {str(e)}", []
