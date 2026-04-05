import torchvision.transforms as T
import numpy as np
from PIL import Image

import cv2
import numpy as np

from config import config

base_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

def meteor_stretch(img, Bmin, Bmax):
    if Bmax == Bmin:
        # Binary image: pixels equal to Bmax -> 255, rest -> 0
        out = np.zeros_like(img, dtype=np.uint8)
        out[img == Bmax] = 255
        return out

    stretched = (img - Bmin) * (255.0 / (Bmax - Bmin))
    stretched = 255 - np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)

def global_threshold(img, T):
    return (img >= T).astype(np.uint8) * 255

def min_max_stretch(img): 
    x_min = img.min() 
    x_max = img.max()   
    
    # Avoid division by zero if the image is flat 
    if x_max == x_min: 
        return np.zeros_like(img, dtype=np.uint8) 
    
    stretched = (img - x_min) * (255.0 / (x_max - x_min)) 
    return stretched.astype(np.uint8)

def percentile_stretch(img, low=2, high=98):
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    # Avoid division by zero
    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.uint8)
    stretched = np.clip((img - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    return stretched

def cv2_equalizer(img):
    return cv2.equalizeHist(img)

class NumpyEnhance:
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def __call__(self, img):
        # PIL -> numpy
        arr = np.array(img)

        out = self.fn(arr, **self.kwargs)

        # numpy -> PIL
        return Image.fromarray(out.astype(np.uint8))
    
class GlobalThresholdEnhance:
    def __call__(self, img, bmin):
        arr = np.array(img)
        out = global_threshold(arr, T=int(bmin))
        return Image.fromarray(out.astype(np.uint8))

class MeteorStretchEnhance:
    def __call__(self, img, bmin, bmax):
        arr = np.array(img)
        out = meteor_stretch(arr, int(bmin), int(bmax))
        return Image.fromarray(out.astype(np.uint8))