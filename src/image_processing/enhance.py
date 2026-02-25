import cv2
import numpy as np

from config import config

def meteor_stretch(img, Bmin, Bmax):
    if Bmax == Bmin:
        # Binary image: pixels equal to Bmax -> 255, rest -> 0
        out = np.zeros_like(img, dtype=np.uint8)
        out[img == Bmax] = 255
        return out

    stretched = (img - Bmin) * (255.0 / (Bmax - Bmin))
    stretched = np.clip(stretched, 0, 255)
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

def percentile_stretch(img, low=config.enhancement.percentile.low, high=config.enhancement.percentile.high):
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    stretched = np.clip((img - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    return stretched

def cv2_equalizer(img):
    return cv2.equalizeHist(img)