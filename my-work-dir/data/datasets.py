import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

VERSIONS_ORIGINAL_IMAGES = ["0.", "1.", "3."]
ENHANCED_SUFFIX = "_CROP_ENHANCED"
DEFAULT_VERSION = "0.0"

class MyMeteorDatasetLabeled(Dataset):
    def __init__(self, image_dir, version=DEFAULT_VERSION, csv_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.version = version

        df = pd.read_csv(csv_file, sep=";")

        # Keep only files that exist
        ending = "_CROP_SUMIMG" if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        valid_files = set(os.path.splitext(f)[0].replace(ending, "") for f in os.listdir(image_dir) if f.lower().endswith(".png"))
        df = df[df["filename"].isin(valid_files)]

        self.df = df.sort_values("filename").reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ending = "_CROP_SUMIMG" if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        img_path = os.path.join(self.image_dir, row['filename'] + f"{ending}.png")
        img = Image.open(img_path).convert('L')
        label = row['class']

        if self.transform:
            img = self.transform(img)

        return img, label

class MyMeteorDataset(Dataset):
    def __init__(self, folder, version=DEFAULT_VERSION, transform=None):
        self.folder = folder
        self.transform = transform
        self.version = version

        # Sort filenames
        ending = "_CROP_SUMIMG" if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        self.files = sorted([os.path.splitext(f)[0].replace(ending, "") for f in os.listdir(folder) if f.lower().endswith(('.png'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        ending = "_CROP_SUMIMG" if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        img_path = os.path.join(self.folder, fname + f"{ending}.png")
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, fname     # return filename as label