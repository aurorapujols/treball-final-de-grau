import torch
import torchvision.transforms as T
import numpy as np
import cv2

from PIL import Image
from sklearn.decomposition import PCA

from config import config

# cpu
base_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

# # gpu
# base_transform = T.Compose([
#     T.Resize((128,128)),
#     T.ToDtype(torch.float32, scale=True)    # [0,1]
# ])

def meteor_stretch(img, Bmin, Bmax):
    if Bmax == Bmin:
        # Binary image: pixels equal to Bmax -> 255, rest -> 0
        out = np.zeros_like(img, dtype=np.uint8)
        out[img == Bmax] = 255
        return out

    stretched = (img - Bmin) * (255.0 / (Bmax - Bmin))
    stretched = 255 - np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)

def meteor_stretch_t(x, bmin, bmax):
    # x: (B, 1, H, W) in [0,1], bmin/bmax: (B,)
    bmin = (bmin / 255.0).view(-1, 1, 1, 1)
    bmax = (bmax / 255.0).view(-1, 1, 1, 1)
    denom = (bmax - bmin).clamp(min=1e-6)

    stretched = (x - bmin) * (1.0 / denom)  # [0,1]
    stretched = stretched.clamp(0.0, 1.0)
    return stretched

def global_threshold(img, T):
    return (img >= T).astype(np.uint8) * 255

def global_threshold_t(x, bmin):
    # x: (B,1,H,W) in [0,1], bmin in original [0,255] scale
    thr = (bmin / 255.0).view(-1,1,1,1)
    return (x >= thr).float()

def min_max_stretch(img): 
    x_min = img.min() 
    x_max = img.max()   
    
    # Avoid division by zero if the image is flat 
    if x_max == x_min: 
        return np.zeros_like(img, dtype=np.uint8) 
    
    stretched = (img - x_min) * (255.0 / (x_max - x_min)) 
    return stretched.astype(np.uint8)

def min_max_stretch_t(x):
    # x: (B, 1, H, W)
    x_min = x.amin(dim=(2,3), keepdim=True)
    x_max = x.amax(dim=(2,3), keepdim=True)
    denom = (x_max - x_min).clamp(min=1e-6)
    stretched = (x - x_min) * (1.0/denom)
    return stretched.clamp(0.0, 1.0)

def percentile_stretch(img, low=2, high=98):
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    # Avoid division by zero
    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.uint8)
    stretched = np.clip((img - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    return stretched

def percentile_stretch_t(x, low=2.0, high=98.0):
    # x: (B,1,H,W)
    B, C, H, W = x.shape
    flat = x.view(B, -1)
    k_low = int(low / 100.0 * (H*W - 1))
    k_high = int(high / 100.0 * (H*W - 1))

    vals, _ = flat.sort(dim=1)
    p_low = vals[torch.arange(B), k_low].view(B,1,1,1)
    p_high = vals[torch.arange(B), k_high].view(B,1,1,1)

    denom = (p_high - p_low).clamp(min=1e-6)
    stretched = (x - p_low) * (1.0 / denom)
    return stretched.clamp(0.0, 1.0)

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
    
def project_2d_hypersphere(Z):
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    Z_2d_norm = Z_2d / np.linalg.norm(Z_2d, axis=1, keepdims=True)
    return Z_2d_norm