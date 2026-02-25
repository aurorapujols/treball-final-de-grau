import os
import time
import torch
import optuna
import numpy as np

from data.dataloaders import get_ssl_loader, get_labeled_loaders, get_full_loader_for_features
from transformations.transform import base_transform
from models.ssl_model import SSLModel, SSLResNet
from training.ssl_training import train_ssl, extract_backbone_features
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


def run_ssl_experiment(cfg, trial=None):
    """
    Runs a full SSL experiment:
      1. Load unlabeled dataset
      2. Load labeled dataset for linear probe
      3. Train SSL model
      4. Extract features
      5. Save results
    """

    print("\n========== SSL EXPERIMENT ==========")
    print(f"Experiment: {cfg['experiment_name']}")

    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    VERSION = cfg["experiment_version"]
    print(f"VERSION={VERSION}")

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------
    print("\nLoading datasets...")

    # Unlabeled dataset for SSL
    unlabeled_dataset, ssl_loader = get_ssl_loader(
        data_root=cfg["paths"]["data_root"],
        batch_size=cfg["training"]["batch_size"],
        transform=base_transform,
        version=VERSION
    )

    # Labeled dataset for linear probe
    labeled_dataset, train_loader, val_loader = get_labeled_loaders(
        data_root=cfg["paths"]["data_root"],
        csv_file=cfg["paths"]["labels_csv"],
        batch_size=cfg["training"]["batch_size"],
        transform=base_transform,
        version=VERSION
    )

    print(f"Unlabeled samples: {len(unlabeled_dataset)}")
    print(f"Labeled samples:   {len(labeled_dataset)}")


    # -------------------------------------------------
    # Initialize model
    # -------------------------------------------------
    print("\nInitializing SSL model...")

    # model = SSLModel(
    #     backbone_dim=cfg["model"]["backbone_dim"],
    #     hidden_dim=cfg["model"]["hidden_dim"],
    #     projection_dim=cfg["model"]["projection_dim"]
    # ).to(device)

    model = SSLResNet(
        backbone_dim=cfg["model"]["backbone_dim"],   # 512 or 2048 (ResNet-18 vs ResNet-50)
        hidden_dim=cfg["model"]["hidden_dim"],
        projection_dim=cfg["model"]["projection_dim"]
    ).to(device)
    
    # -------------------------------------------------
    # Train SSL model
    # -------------------------------------------------
    start_time = time.time()

    model, history, stop_epoch = train_ssl(
        model=model,
        batch_size=cfg["training"]["batch_size"],
        num_epochs=cfg["training"]["num_epochs"],
        patience=cfg["training"]["patience"],
        cutoff_ratio=cfg["training"]["cutoff_ratio"],
        lr=cfg["training"]["learning_rate"],
        temperature=cfg["training"]["temperature"],
        loader=ssl_loader,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        version=cfg["experiment_version"],
        output_path=cfg["paths"]["output_dir"],
        use_image_augs=cfg["training"]["use_image_augs"],
        trial=trial
    )

    training_time = time.time() - start_time

    # -------------------------------------------------
    # Extract features
    # -------------------------------------------------
    print("\nExtracting backbone features...")
    full_loader = get_full_loader_for_features(unlabeled_dataset, batch_size=cfg["training"]["batch_size"])
    features, filenames = extract_backbone_features(model, full_loader, device)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    print("\nSaving results...")

    # Save training history
    history_path = os.path.join(output_dir, f"ssl_history_{cfg['experiment_name']}_{VERSION}.csv")
    history.to_csv(history_path, sep=";", index=False)
    
    # Save features
    feat_path = os.path.join(output_dir, f"ssl_features_{cfg['experiment_name']}_{VERSION}.npy")
    name_path = os.path.join(output_dir, f"ssl_filenames_{cfg['experiment_name']}_{VERSION}.npy")
    np.save(feat_path, features)
    np.save(name_path, filenames)

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"ssl_model_{cfg['experiment_name']}_{VERSION}.pt")
    save_checkpoint(model, ckpt_path)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("\n========== SSL SUMMARY ==========")
    print(f"Experiment:       {cfg['experiment_name']}")
    print(f"Stop epoch:       {stop_epoch}")
    print(f"Final accuracy:   {history['accuracy'].iloc[-1]:.4f}")
    print(f"Final loss:       {history['contrastive_loss'].iloc[-1]:.4f}")
    print(f"Training time:    {training_time:.2f} sec")
    print(f"Features saved:   {feat_path}")
    print(f"Model saved:      {ckpt_path}")
    print("=================================\n")

    return model, history, stop_epoch
