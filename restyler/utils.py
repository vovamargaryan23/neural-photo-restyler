import numpy as np
from PIL import Image
from config import DEVICE


def load_image(image_path, transform=None, max_size=None, shape=None):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)
    
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.Resampling.LANCZOS)
    
    if shape:
        image = image.resize(shape, Image.Resampling.LANCZOS)
    
    if transform:
        image = transform(image).unsqueeze(0)
    
    return image.to(DEVICE)