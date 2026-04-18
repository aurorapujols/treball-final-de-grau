import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from transformations.augment import ControlledAugmentGPU

VERSIONS_ORIGINAL_IMAGES = ["1.", "2.", "3.", "4."]
ENHANCED_SUFFIX = "_CROP_ENHANCED"
DEFAULT_VERSION = "1.0"

def get_df_from_csv(filepath, sep=";"):
    if os.path.exists(filepath):
        return pd.read_csv(filepath, sep=sep)
    return None

class MyMeteorDataset(Dataset):
    def __init__(self, img_folder, dataset, version=DEFAULT_VERSION, transform=None):
        self.folder = img_folder
        self.transform = transform
        self.version = version

        # Sort filenames
        ending = "_CROP_SUMIMG" # if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        self.files = sorted([
            f for f in dataset["filename"].tolist()
            if os.path.isfile(os.path.join(img_folder, f + f"{ending}.png"))
        ])

        self.dataset = dataset[dataset["filename"].isin(self.files)].sort_values("filename").reset_index(drop=True)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        fname = self.files[idx]
        bmin = row["bmin"]
        bmax = row["bmax"]
        label = row["class"]

        ending = "_CROP_SUMIMG" # if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        img_path = os.path.join(self.folder, fname + f"{ending}.png")

        img = Image.open(img_path).convert('L')
        # img = np.array(img, dtype=np.uint8)         # GPU
        # img = torch.from_numpy(img).unsqueeze(0)    # (1, H, W)

        if self.transform:
            img = self.transform(img)

        return img, fname, bmin, bmax, label    

class TwoViewDataset(Dataset):
    def __init__(self, img_folder, dataset, version=DEFAULT_VERSION, transform=None):
        self.folder = img_folder
        self.transform = transform
        self.version = version

        # Sort filenames
        ending = "_CROP_SUMIMG" # if (self.version[:2] in VERSIONS_ORIGINAL_IMAGES) else ENHANCED_SUFFIX
        self.files = sorted([
            f for f in dataset["filename"].tolist()
            if os.path.isfile(os.path.join(img_folder, f + f"{ending}.png"))
        ])

        self.dataset = dataset[dataset["filename"].isin(self.files)].sort_values("filename").reset_index(drop=True)

        self.augmentfn = ControlledAugmentGPU(use_enhanced=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        fname = self.files[idx]
        # Ensure these are tensors for the GPU augmentation logic
        bmin = torch.tensor([row["bmin"]])
        bmax = torch.tensor([row["bmax"]])
        label = row["class"]

        ending = "_CROP_SUMIMG"
        img_path = os.path.join(self.folder, fname + f"{ending}.png")

        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img) # Now img is [1, H, W]

        # 1. Add batch dimension: [1, 1, H, W]
        img_batch = img.unsqueeze(0)

        # 2. Apply augmentation and then squeeze back to [1, H, W]
        x_i = self.augmentfn.one_view(img_batch, bmin, bmax).squeeze(0)
        x_j = self.augmentfn.one_view(img_batch, bmin, bmax).squeeze(0)

        return x_i, x_j, label
    
class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, imgs_folder, transform=None):
        self.df = df
        self.imgs_folder = imgs_folder
        self.transform = transform

        self.label_map = {"meteor": 1, "non-meteor": 0, "unknown": 0, 0: 0, 1: 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.imgs_folder, f"{row['filename']}_CROP_SUMIMG.png")
        raw_label = row["class"]   # assumes 0/1 labels
        label = self.label_map[raw_label]   # now an integer

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def split_dataset(dataset, output_path, test_frac=0.1, val_frac=0.1, seed=42):

    # Shuffle to make the splits random
    dataset = dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_total = len(dataset)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)

    test_set = dataset.iloc[:n_test]
    val_set = dataset.iloc[n_test:n_test + n_val]
    train_set = dataset.iloc[n_test + n_val:]

    train_set.to_csv(f"{output_path}/dataset_train.csv", sep=";", index=False)
    val_set.to_csv(f"{output_path}/dataset_val.csv", sep=";", index=False)
    test_set.to_csv(f"{output_path}/dataset_test.csv", sep=";", index=False)

    print(f"Saved: dataset splits.")
    print(f"Train: {len(train_set)} | Test: {len(test_set)} | Val: {len(val_set)}")

    return train_set, val_set, test_set

def get_dataset_split(full_dataset_csv_path, output_path):
    
    # Full dataset exists
    if os.path.isfile(full_dataset_csv_path):

        dataset = pd.read_csv(full_dataset_csv_path, sep=";")

        if os.path.isfile(f"{output_path}/dataset_train.csv") and os.path.isfile(f"{output_path}/dataset_val.csv") and os.path.isfile(f"{output_path}/dataset_test.csv"):
            print("Uploading split.")
            train_set = pd.read_csv(f"{output_path}/dataset_train.csv", sep=";")
            val_set = pd.read_csv(f"{output_path}/dataset_val.csv", sep=";")
            test_set = pd.read_csv(f"{output_path}/dataset_test.csv", sep=";")
            print(f"Train: {len(train_set)} | Test: {len(test_set)} | Val: {len(val_set)}")
            return train_set, val_set, test_set
        
        else:
            print("Creating split.")
            return split_dataset(dataset, output_path)
    else:
        print("Full dataset not found.")
        return None