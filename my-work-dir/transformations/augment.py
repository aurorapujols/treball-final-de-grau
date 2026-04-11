import os
import random
import torch
import math
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ColorJitter
from transformations.transform import base_transform
from PIL import Image

from transformations.transform import (
    NumpyEnhance, 
    MeteorStretchEnhance, 
    GlobalThresholdEnhance, 
    min_max_stretch, 
    percentile_stretch,
    min_max_stretch_t,
    global_threshold_t,
    percentile_stretch_t,
    meteor_stretch_t)


class RandomAffineMeanFill:
    def __init__(self, degrees=(-90, 90), translate=(0.05, 0.05), scale=(0.9, 1.1)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, img):
        # Convert PIL → tensor
        t = TF.to_tensor(img)  # shape (1, H, W)

        # Compute mean pixel value in 0–255 range
        mean_val = float(t.mean().item() * 255.0)

        # Sample affine params
        angle = random.uniform(*self.degrees)
        translate_px = (
            self.translate[0] * img.size[0],
            self.translate[1] * img.size[1]
        )
        scale = random.uniform(*self.scale)

        # Apply affine with mean fill
        out = TF.affine(
            img,
            angle=angle,
            translate=translate_px,
            scale=scale,
            shear=[0.0, 0.0],
            fill=int(mean_val)
        )

        return out

class RandomAffineCropSquare:
    def __init__(self, degrees=(-90, 90), out_size=128):
        self.degrees = degrees
        self.out_size = out_size

    def __call__(self, img):
        # 1. Sample angle
        angle = random.uniform(*self.degrees)

        # 2. Rotate with expand=True
        rotated = img.rotate(angle, expand=True, fillcolor=0)

        # 3. Compute crop size (Mathematica "SameRatioCropping")
        L = img.size[0]  # original side
        theta = math.radians(abs(angle) % 90)

        denom = abs(math.cos(theta)) + abs(math.sin(theta))
        side = L / denom

        # 4. Crop centered in the rotated bounding box
        W, H = rotated.size
        cx, cy = W // 2, H // 2

        half = side / 2
        left = int(cx - half)
        top = int(cy - half)
        right = int(cx + half)
        bottom = int(cy + half)

        cropped = rotated.crop((left, top, right, bottom))

        # 5. Resize back to output size
        cropped = cropped.resize((self.out_size, self.out_size)) #, resample=Image.BILINEAR)

        return cropped

class RandomAffineCropSquareGPU(torch.nn.Module):
    def __init__(self, degrees=(-90, 90), out_size=128):
        super().__init__()
        self.degrees = degrees
        self.out_size = out_size

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Sample angles and convert to radians
        angles = torch.empty(B, device=device).uniform_(self.degrees[0], self.degrees[1])
        rad = angles * (math.pi / 180.0)
        
        # 2. Rotation components
        cos = torch.cos(rad)
        sin = torch.sin(rad)

        # 3. The largest axis-aligned square inscribed in a rotated unit square
        #    has side length 1 / (|cos| + |sin|).
        #    To crop to it, we DIVIDE the rotation matrix by this factor
        #    (equivalent to multiplying by |cos| + |sin|), which zooms the
        #    grid INTO the inscribed region.
        scale = cos.abs() + sin.abs()  # == 1 / inscribed_side_length

        # 4. Build affine matrix: pure rotation scaled to sample only the
        #    inscribed square. Dividing by scale shrinks the sampling window.
        thetas = torch.zeros((B, 2, 3), device=device)
        thetas[:, 0, 0] =  cos / scale   # was cos * scale
        thetas[:, 0, 1] = -sin / scale   # was -sin * scale
        thetas[:, 1, 0] =  sin / scale   # was sin * scale
        thetas[:, 1, 1] =  cos / scale   # was cos * scale
        # translation column stays 0 (crop is centered)

        # 5. Generate grid and sample into desired output size
        grid = F.affine_grid(thetas, size=(B, C, self.out_size, self.out_size), align_corners=False)
        out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return out

class ControlledAugment:
    """
    Applies each augmentation independently with probability p = 1/N. 
    Ensures each view gets at least one augmentation. 
    """
    def __init__(self, augs_idx=None, use_enhanced=False): 

        # Geometric / photometric basic augmentations
        self.augs = [
            T.ColorJitter(brightness=1.5, contrast=1.5),
            RandomAffineCropSquare(degrees=(-90,90), out_size=128),
            T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),            
            T.RandomResizedCrop(size=128, scale=(0.3, 1.0)),    # resize to 128x128 image crop up to 50% in the image
        ] # list of PIL→PIL transforms 

        # Enhanced-image augmentations
        self.enhance_fns = {
            "min_max_stretch": NumpyEnhance(min_max_stretch),
            "global_threshold": GlobalThresholdEnhance(),
            "percentile_stretch": NumpyEnhance(percentile_stretch),
            "meteors_stretch": MeteorStretchEnhance()
        }

        self.img_types = list(self.enhance_fns.keys())
        self.use_enhanced = use_enhanced

        if augs_idx is not None:

            if not use_enhanced:
                # Only geometric augs
                try:
                    self.augs = [self.augs[i] for i in augs_idx]
                except TypeError:
                    self.augs = [self.augs[augs_idx]]

                print("Using geometric augs:", self.augs)
                
            else:
                # Geometric + enhanced (keep all geometric, filter only enhanced augs)
                try:
                    self.img_types = [self.img_types[i] for i in augs_idx]
                except TypeError:
                    self.img_types = [self.img_types[augs_idx]]

                print("Using enhanced augs:", self.img_types)

        self.N = len(self.augs)
        self.p = 1.0 / self.N # equal probability 

    def apply_aug(self, aug, img, bmin, bmax):
        # Case of color jitter, use one of the enhanced images
        if isinstance(aug, ColorJitter) and self.use_enhanced:
            enh = random.choice(self.img_types)
            fn = self.enhance_fns[enh]

            if isinstance(fn, MeteorStretchEnhance):
                return fn(img, bmin, bmax)
            if isinstance(self.enhance_fns[enh], GlobalThresholdEnhance):
                return fn(img, bmin)
            return fn(img)
        
        return aug(img)
        
    def one_view(self, img_tensor, img_name, bmin, bmax): 
        img = T.ToPILImage()(img_tensor)
        applied = False

        for aug in self.augs:
            if random.random() <= self.p:
                applied = True
                img = self.apply_aug(aug, img, bmin, bmax)                
            
        if not applied: # if no augmentation was applied, ensure one is
            aug = random.choice(self.augs)
            img = self.apply_aug(aug, img, bmin, bmax)
        
        return T.ToTensor()(img)
     
    def __call__(self, batch, fnames, bmins, bmaxs): 
        x_i = torch.stack([self.one_view(img, fname, bmin, bmax) for img, fname, bmin, bmax in zip(batch, fnames, bmins, bmaxs)]) 
        x_j = torch.stack([self.one_view(img, fname, bmin, bmax) for img, fname, bmin, bmax in zip(batch, fnames, bmins, bmaxs)]) 
        return x_i, x_j


class ControlledAugmentGPU(nn.Module):
    def __init__(self, augs_idx=None, use_enhanced=False, img_types=None):
        super().__init__()

        # Geometric / photometric augmentations
        self.augs = nn.ModuleList([
            T.ColorJitter(brightness=1.5, contrast=1.5),
            RandomAffineCropSquareGPU(degrees=(-90, 90), out_size=128),
            T.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 2.0)),
            T.RandomResizedCrop(size=(128,128), scale=(0.3, 1.0)),
        ])

        # Enhanced transforms
        self.img_types = img_types or [
            "min_max_stretch",
            # "global_threshold",
            "percentile_stretch",
            "meteor_stretch"
        ]

        self.use_enhanced = use_enhanced

        if augs_idx is not None:
            if not use_enhanced:
                # Filter geometric augs
                if isinstance(augs_idx, int):
                    augs_idx = [augs_idx]
                self.augs = nn.ModuleList([self.augs[i] for i in augs_idx])
                print("Using geometric augs:", self.augs)
            else:
                # Filter enhanced transforms
                if isinstance(augs_idx, int):
                    augs_idx = [augs_idx]
                self.img_types = [self.img_types[i] for i in augs_idx]
                print("Using enhanced augs:", self.img_types)

        self.N = len(self.augs)
        self.p = 1.0 / self.N

    # --- Enhanced transforms ---
    def apply_enhance(self, x, bmin, bmax):
        choice = random.choice(self.img_types)
        if choice == "min_max_stretch":
            return min_max_stretch_t(x)
        elif choice == "global_threshold":
            return global_threshold_t(x, bmin)
        elif choice == "percentile_stretch":
            return percentile_stretch_t(x)
        elif choice == "meteor_stretch":
            return meteor_stretch_t(x, bmin, bmax)
        return x

    # --- Apply one view ---
    def one_view(self, x, bmin, bmax):
        B = x.size(0)
        out = x.clone()
        applied_mask = torch.zeros(B, dtype=torch.bool, device=x.device)

        for aug in self.augs:
            mask = torch.rand(B, device=x.device) <= self.p
            if mask.any():

                # If ColorJitter AND enhanced mode → replace with enhanced transform
                if self.use_enhanced and isinstance(aug, T.ColorJitter):
                    out[mask] = self.apply_enhance(out[mask], bmin[mask], bmax[mask])
                else:
                    out[mask] = aug(out[mask])

                applied_mask |= mask

        # Fallback: ensure at least one augmentation
        if (~applied_mask).any():
            idx = torch.nonzero(~applied_mask).squeeze(1)
            aug = random.choice(self.augs)

            if self.use_enhanced and isinstance(aug, T.ColorJitter):
                out[idx] = self.apply_enhance(out[idx], bmin[idx], bmax[idx])
            else:
                out[idx] = aug(out[idx])

        return out

    def forward(self, batch, fnames, bmins, bmaxs):
        bmins = bmins.to(batch.device).float()
        bmaxs = bmaxs.to(batch.device).float()
        x_i = self.one_view(batch, bmins, bmaxs)
        x_j = self.one_view(batch, bmins, bmaxs)
        return x_i, x_j


class Augment:
    """
    Stochastic augmentation returning two correlated views per image.
    """
    def __init__(self):
        self.train_transform = T.Compose([
            RandomAffineMeanFill(degrees=(-90,90), scale=(0.9, 1.1)),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=2.0, contrast=2.0)], p=0.5),
            T.RandomResizedCrop(size=64, scale=(0.8, 1.0)),

            T.ToTensor()
        ])

    def __call__(self, batch):
        # batch: tensor (B, 1, H, W)
        x_i = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        x_j = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        return x_i, x_j
