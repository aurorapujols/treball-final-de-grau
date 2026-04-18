import os


from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

from .datasets import MyMeteorDataset, TwoViewDataset, CSVImageDataset
from .collate import pad_collate


def get_ssl_loader(data_root, dataframe, batch_size, transform, version=None, shuffle=True):
    dataset = MyMeteorDataset(img_folder=data_root, dataset=dataframe, version=version, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )
    return dataset, loader

def get_two_view_loader(data_root, dataframe, batch_size, transform, version=None, shuffle=True):
    dataset = TwoViewDataset(img_folder=data_root, dataset=dataframe, version=version, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )
    return dataset, loader

def get_csv_loader(data_root, dataframe, transform):
    dataset = CSVImageDataset(df=dataframe, imgs_folder=data_root, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False
    )
    return dataset, loader

# def get_ssl_loaders(data_root, csv_file, batch_size, transform, val_split=0.2, version=None):
#     dataset = MyMeteorDataset(folder=data_root, csv_file=csv_file, version=version, transform=transform)
    
#     indices = list(range(len(dataset)))
#     train_idx, val_idx = train_test_split(indices, test_size=val_split, shuffle=True)
    
#     train_set = Subset(dataset, train_idx)
#     val_set = Subset(dataset, val_idx)
    
#     ssl_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     ssl_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
#     return dataset, ssl_train_loader, ssl_val_loader, train_idx, val_idx
    

# def get_labeled_loaders(data_root, csv_file, batch_size, transform, val_split=0.2, version=None, train_idx=None, val_idx=None):
#     dataset = MyMeteorDatasetLabeled(
#         image_dir=data_root,
#         version=version,
#         csv_file=csv_file,
#         transform=transform
#     )
    
#     if (train_idx is None) or (val_idx is None):
#         indices = list(range(len(dataset)))
#         train_idx, val_idx = train_test_split(
#             indices,
#             test_size=val_split,
#             stratify=dataset.df["class"]
#         )

#     train_set = Subset(dataset, train_idx)
#     val_set = Subset(dataset, val_idx)

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_set,
#         batch_size=batch_size,
#         shuffle=False,
#         pin_memory=True
#     )

#     return dataset, train_loader, val_loader

def get_full_loader_for_features(dataset, batch_size):
    """Used for feature extraction (SSL, SupCon, clustering)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
