import os
import random
import torch
import math
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from transformations.transform import base_transform
from PIL import Image


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
    def __init__(self, img_transforms_path=None): 
        self.augs = [
            RandomAffineCropSquare(degrees=(-90,90), out_size=128),
            # T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),
            T.ColorJitter(brightness=1.0, contrast=1.0),
            T.RandomResizedCrop(size=128, scale=(0.5, 1.0)),    # resize to 128x128 image crop up to 50% in the image
        ] # list of PIL→PIL transforms 
        self.img_types = ["global_threshold", "percentile_stretch"]
        self.img_path = img_transforms_path

        self.all_augs = self.augs + (self.img_types if self.img_path is not None else [])

        self.N = len(self.all_augs)
        self.p = 1.0 / self.N # equal probability 



    def get_enhanced_image(self, fname, img_type):

        filename = f"{os.path.basename(fname)}{'_CROP_ENHANCED' if img_type != 'original' else '_CROP_SUMIMG'}.png"     # in case can't not open file, use original image
        path = os.path.join(self.img_path, img_type, filename)
        try:
            img = Image.open(path)
        except:
            return None
        return img.resize((128, 128)) 
        
    def one_view(self, img_tensor, img_name): 
        img = T.ToPILImage()(img_tensor)
        applied = False

        for aug in self.all_augs:
            if random.random() <= self.p:
                applied = True

                if isinstance(aug, str):    # if it is the name of the folder in img_types
                    img_enhanced = self.get_enhanced_image(img_name, aug)
                    if img_enhanced is not None:
                        img = img_enhanced
                    else: #keep the original image
                        applied = False
                else:   # normal augmentations
                    img = aug(img)
            
        if not applied: # if no augmentation was applied, ensure one is
            while (not applied):
                aug = random.choice(self.all_augs)
                if isinstance(aug, str):
                    img_enhanced = self.get_enhanced_image(img_name, aug) 
                    if img_enhanced is not None:
                        img = img_enhanced
                        applied = True
                else: 
                    img = aug(img)
                    applied = True
        
        return T.ToTensor()(img)
     
    def __call__(self, batch, fnames): 
        x_i = torch.stack([self.one_view(img, fname) for img, fname in zip(batch, fnames)]) 
        x_j = torch.stack([self.one_view(img, fname) for img, fname in zip(batch, fnames)]) 
        return x_i, x_j


class Augment:
    """
    Stochastic augmentation returning two correlated views per image.
    """
    def __init__(self):
        self.train_transform = T.Compose([
            RandomAffineMeanFill(degrees=(-90,90), scale=(0.9, 1.1)),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8)], p=0.5),
            T.RandomResizedCrop(size=64, scale=(0.8, 1.0)),

            T.ToTensor()
        ])

    def __call__(self, batch):
        # batch: tensor (B, 1, H, W)
        x_i = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        x_j = torch.stack([self.train_transform(T.ToPILImage()(img)) for img in batch])
        return x_i, x_j
