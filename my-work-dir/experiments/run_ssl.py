import os
import time
import torch
import optuna
import numpy as np

from data.dataloaders import get_ssl_loader, get_ssl_loaders, get_labeled_loaders, get_full_loader_for_features
from transformations.transform import base_transform
from models.ssl_model import SSLModel, SSLResNet
from training.ssl_training import train_ssl, extract_backbone_features
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


def run_ssl_experiment(cfg, add_version=None, augs_idx=None, trial=None):
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

    if add_version is not None:
        VERSION = f"{cfg['experiment_version'][:2]}{add_version}"
    else:
        VERSION = cfg['experiment_version']
    print(f"VERSION={VERSION}")

    lr = trial.params["lr"] if trial else cfg["training"]["learning_rate"]
    temperature = trial.params["temperature"] if trial else cfg["training"]["temperature"]
    batch_size = trial.params["batch_size"] if trial else cfg["training"]["batch_size"]


    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------
    print("\nLoading datasets...")

    # Unlabeled dataset for SSL (ALL or split)
    train_idx = None
    val_idx = None
    ssl_val_loader = None
    
    if cfg['training']['train_with_all_data']:
        unlabeled_dataset, ssl_train_loader = get_ssl_loader(
            data_root=cfg["paths"]["data_root"],
            csv_file=cfg["paths"]["labels_csv"],
            batch_size=batch_size,
            transform=base_transform,
            version=VERSION
        )
    else:
        unlabeled_dataset, ssl_train_loader, ssl_val_loader, train_idx, val_idx = get_ssl_loaders(
            data_root=cfg["paths"]["data_root"],
            csv_file=cfg["paths"]["labels_csv"],
            batch_size=batch_size,
            transform=base_transform,
            val_split=0.2,
            version=VERSION
        )

    # Labeled dataset for linear probe
    labeled_dataset, lp_train_loader, lp_val_loader = get_labeled_loaders(
        data_root=cfg["paths"]["data_root"],
        csv_file=cfg["paths"]["labels_csv"],
        batch_size=batch_size,
        transform=base_transform,
        version=VERSION,
        train_idx=train_idx,
        val_idx=val_idx
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
        res_net_dim=cfg["model"]["res_net_dim"],
        projection_dim=cfg["model"]["projection_dim"]
    ).to(device)
    
    # -------------------------------------------------
    # Train SSL model
    # -------------------------------------------------
    start_time = time.time()

    model, best_acc, best_state, history, stop_epoch = train_ssl(
        model=model,
        num_epochs=cfg["training"]["num_epochs"],
        patience=cfg["training"]["patience"],
        cutoff_ratio=cfg["training"]["cutoff_ratio"],
        lr=lr,
        temperature=temperature,
        ssl_train_loader=ssl_train_loader,
        ssl_val_loader=ssl_val_loader,
        lp_train_loader=lp_train_loader,
        lp_val_loader=lp_val_loader,
        device=device,
        version=VERSION,
        output_path=cfg["paths"]["output_dir"],
        augs_idx=augs_idx,
        use_enhanced=cfg["training"]["use_enhanced"],
        trial=trial
    )

    training_time = time.time() - start_time

    # -------------------------------------------------
    # Extract features
    # -------------------------------------------------
    print("\nExtracting backbone features...")
    full_loader = get_full_loader_for_features(unlabeled_dataset, batch_size=batch_size)
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

    print("Loading best model (val_accuracy = {:.4f})".format(best_acc))
    model.load_state_dict(best_state)
    best_model_path = os.path.join(output_dir, f"ssl_best_model_{VERSION}.pt")
    save_checkpoint(model, best_model_path)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("\n========== SSL SUMMARY ==========")
    print(f"Experiment:       {cfg['experiment_name']}")
    print(f"Stop epoch:       {stop_epoch}")
    print(f"Final accuracy:   {history['val_accuracy'].iloc[-1]:.4f}")
    print(f"Final loss:       {history['contrastive_loss'].iloc[-1]:.4f}")
    print(f"Training time:    {training_time:.2f} sec")
    print(f"Features saved:   {feat_path}")
    print(f"Model saved:      {ckpt_path}")
    print("=================================\n")

    return model, history, stop_epoch
