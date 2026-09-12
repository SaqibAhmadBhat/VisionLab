import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

def generate_histogram(image: np.ndarray) -> np.ndarray:
    """Generate a histogram plot image from the input image."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if len(image.shape) == 2:
        # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
    else:
        # RGB
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
            ax.set_xlim([0, 256])
            
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Read the image back as numpy array
    hist_img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    hist_img = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)
    
    return hist_img
