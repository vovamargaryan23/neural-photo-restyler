from typing import Tuple, Union

import os
import math

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms


_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage() 


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def read_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    return _ensure_rgb(img)


def save_image(image: Image.Image, path: str) -> bool:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png"]:
        image.save(path, format="PNG")
    else:
        image.save(path, format="JPEG")
    

def pil_to_tensor(image: Image.Image) -> Tensor:
    t = _to_tensor(image)
    return t
    

def tensor_to_pil(tensor: Tensor) -> Image.Image:
    i = _to_pil(tensor)
    return i