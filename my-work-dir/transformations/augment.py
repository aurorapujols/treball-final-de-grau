import os
import random
import torch
import math
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms import ColorJitter
from transformations.transform import base_transform
from PIL import Image

from transformations.transform import NumpyEnhance, MeteorStretchEnhance, GlobalThresholdEnhance, min_max_stretch, percentile_stretch


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
            # "meteors_stretch": MeteorStretchEnhance()
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
