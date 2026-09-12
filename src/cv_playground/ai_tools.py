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

def get_ai_intelligence(image: np.ndarray) -> Tuple[bool, str, List[Tuple[str, float]]]:
    """
    Optional AI feature using torchvision's MobileNetV3 Small.
    Returns: (success_bool, message, list_of_predictions)
    """
    if not TORCH_AVAILABLE:
        return False, "PyTorch and Torchvision are not installed. Please install them to use AI Features: pip install torch torchvision", []
        
    try:
        # Load a pre-trained MobileNetV3 small model
        # Using weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        model.eval()
        
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
