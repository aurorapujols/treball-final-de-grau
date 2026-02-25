from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .datasets import MyMeteorDataset, MyMeteorDatasetLabeled
from .collate import pad_collate


def get_ssl_loader(data_root, batch_size, transform, version=None):
    dataset = MyMeteorDataset(folder=data_root, version=version, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    return dataset, loader


def get_labeled_loaders(data_root, csv_file, batch_size, transform, val_split=0.2, version=None):
    dataset = MyMeteorDatasetLabeled(
        image_dir=data_root,
        version=version,
        csv_file=csv_file,
        transform=transform
    )

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=dataset.df["class"]
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return dataset, train_loader, val_loader


def get_full_loader_for_features(dataset, batch_size):
    """Used for feature extraction (SSL, SupCon, clustering)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
